import torch
import torch.nn as nn
import torch.nn.functional as F

class RadiologicalRasterizer(nn.Module):
    def __init__(self):
        super(RadiologicalRasterizer, self).__init__()

    def forward(self, gaussian_params, pose, slice_shape=(200, 200)):
        """
        CPU-based Rasterizer to definitively avoid GPU OOM.
        We perform the projection and accumulation on CPU and return to GPU.
        """
        device = gaussian_params['means'].device
        B = gaussian_params['means'].shape[0]
        G = gaussian_params['means'].shape[1]
        H, W = slice_shape
        
        means_cpu = gaussian_params['means'].detach().cpu()
        scales_cpu = gaussian_params['scales'].detach().cpu()
        if scales_cpu.dim() == 2: scales_cpu = scales_cpu.unsqueeze(0).expand(B, -1, -1)
        
        # Use opacities directly (already sigmoid in forward pass)
        opacities_cpu = gaussian_params['opacities'].detach().cpu()
        intensities_cpu = gaussian_params['intensities'].detach().cpu()
        oi_cpu = (opacities_cpu * intensities_cpu).squeeze(-1)
        if oi_cpu.dim() == 1: oi_cpu = oi_cpu.unsqueeze(0).expand(B, -1)
        
        with torch.no_grad():
            u = torch.linspace(0, W-1, W)
            v = torch.linspace(0, H-1, H)
            grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
            
            pts_img = torch.zeros(H, W, 3)
            axis = pose['axis']
            slice_idx = pose['slice_idx']
            
            if axis == 0:
                pts_img[..., 0] = grid_u
                pts_img[..., 1] = slice_idx
                pts_img[..., 2] = grid_v
            elif axis == 1:
                pts_img[..., 0] = slice_idx
                pts_img[..., 1] = grid_u
                pts_img[..., 2] = grid_v
            else:
                pts_img[..., 0] = grid_u
                pts_img[..., 1] = grid_v
                pts_img[..., 2] = slice_idx

            pts_h = torch.cat([pts_img, torch.ones(H, W, 1)], dim=-1)
            pts_world = (pts_h @ pose['affine'].cpu().T)[..., :3]
            
            # Apply Pose Deltoid (R_delta, T_delta) if present
            if 'R_delta' in pose:
                # pts_world: (H, W, 3)
                R = pose['R_delta'].cpu()
                T = pose['T_delta'].cpu()
                # Apply rotation relative to center_world
                center = pose['center_world'].cpu()
                pts_world = (pts_world - center) @ R.T + center + T
            
            pts_flat = pts_world.view(-1, 3)
            N = pts_flat.shape[0]

        # Accumulate on CPU
        # Even on CPU, large tensors are bad, so we use a simple loop
        final_signal = torch.zeros(B, N)
        
        # We can use a slightly larger chunk on CPU
        chunk_p = 1000
        for i in range(0, N, chunk_p):
            p_chunk = pts_flat[i:i+chunk_p].unsqueeze(1) # (Np, 1, 3)
            
            for b in range(B):
                # (Np, G, 3)
                diff = p_chunk - means_cpu[b].unsqueeze(0)
                dist_sq = torch.sum((diff / (scales_cpu[b].unsqueeze(0) + 1e-8))**2, dim=-1)
                weights = torch.exp(-0.5 * dist_sq)
                final_signal[b, i:i+chunk_p] = torch.sum(weights * oi_cpu[b].unsqueeze(0), dim=-1)
                
        # Return to original device
        # Note: detach makes it non-differentiable. 
        # For a truly differentiable CPU rasterizer, we'd need to keep gradients.
        # But let's get the forward pass working first.
        return final_signal.view(B, H, W).to(device)
