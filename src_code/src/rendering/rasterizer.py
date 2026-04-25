import torch
import torch.nn as nn
import torch.nn.functional as F

class RadiologicalRasterizer(nn.Module):
    def __init__(self):
        super(RadiologicalRasterizer, self).__init__()

    def forward(self, gaussian_params, ray_bundle):
        """
        gaussian_params: dict containing means, scales, rotations, opacities, intensities
        ray_bundle: dict containing origins (B, H, W, 3) and directions (B, H, W, 3)
        """
        B, H, W, _ = ray_bundle['origins'].shape
        device = ray_bundle['origins'].device
        
        # This is a simplified differentiable radiological rasterizer.
        # It accumulates density and intensity along rays.
        
        # 1. Sample points along rays
        num_samples_per_ray = 64
        t_vals = torch.linspace(0., 1., steps=num_samples_per_ray).to(device)
        # Assuming near/far clip of [-150, 150] for heart volume
        near, far = -150., 150. 
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals = z_vals.expand(B, H, W, num_samples_per_ray)
        
        # (B, H, W, num_samples, 3)
        pts = ray_bundle['origins'].unsqueeze(-2) + ray_bundle['directions'].unsqueeze(-2) * z_vals.unsqueeze(-1)
        
        # 2. Evaluate Gaussian contributions at these points
        # For memory efficiency, we might need to chunk this or use a more optimized kernel.
        # pts: (B, H*W*num_samples, 3)
        pts_flat = pts.view(B, -1, 3)
        
        # Re-use the evaluate_occupancy logic but for intensity
        # occupancy = evaluate_occupancy(pts_flat, gaussian_params)
        
        means = gaussian_params['means'].unsqueeze(1) # (B, 1, num_gaussians, 3)
        pts_exp = pts_flat.unsqueeze(2) # (B, pts_flat, 1, 3)
        
        diff = pts_exp - means # (B, pts_flat, num_gaussians, 3)
        scales = gaussian_params['scales'].unsqueeze(0).unsqueeze(0)
        
        # (B, pts_flat, num_gaussians)
        exponent = -0.5 * torch.sum((diff / (scales + 1e-8))**2, dim=-1)
        weights = torch.exp(exponent)
        
        opacities = gaussian_params['opacities'].unsqueeze(0).transpose(-1, -2) # (B, 1, num_gaussians)
        intensities = gaussian_params['intensities'].unsqueeze(0).transpose(-1, -2) # (B, 1, num_gaussians)
        
        # Local density at each point
        densities = torch.sum(weights * opacities, dim=-1) # (B, pts_flat)
        # Local intensity (radiological signal)
        signal = torch.sum(weights * opacities * intensities, dim=-1) # (B, pts_flat)
        
        # 3. Accumulate along rays
        densities = densities.view(B, H, W, num_samples_per_ray)
        signal = signal.view(B, H, W, num_samples_per_ray)
        
        # Radiological accumulation is typically additive density
        # (X-ray/CT/MRI intensity is often an integral of local properties)
        accumulated_signal = torch.sum(signal, dim=-1) # (B, H, W)
        
        return accumulated_signal
