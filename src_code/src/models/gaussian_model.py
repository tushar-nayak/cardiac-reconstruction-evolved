import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianModel(nn.Module):
    def __init__(self, num_gaussians=10000, latent_dim=128):
        super(GaussianModel, self).__init__()
        self.num_gaussians = num_gaussians
        
        # Initial parameters for Gaussians (can be initialized from sparse points)
        self.means = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * 5.0)
        self.rotations = nn.Parameter(torch.tensor([1., 0., 0., 0.]).repeat(num_gaussians, 1))
        self.opacities = nn.Parameter(torch.ones(num_gaussians, 1) * 1.0)
        self.intensities = nn.Parameter(torch.ones(num_gaussians, 1) * 0.5)

        # MLP to predict deformations from latent state
        self.deformation_net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_gaussians * 7) # Predicts delta_means (3), delta_scales (3), delta_opacity (1)
        )

    def get_covariance(self):
        # Placeholder for building covariance matrix from scales and rotations (quaternions)
        # For now, just use scales as axis-aligned for simplicity
        return self.scales

    def initialize_from_voxels(self, occupancy_volumes, affines):
        """
        Initialize Gaussian means from occupied voxels.
        occupancy_volumes: List of (D, H, W) tensors
        affines: (B, 4, 4)
        """
        device = self.means.device
        all_pts_world = []
        for i in range(len(occupancy_volumes)):
            occ_indices = torch.nonzero(occupancy_volumes[i])
            if len(occ_indices) > 0:
                # Subsample
                num_to_sample = min(len(occ_indices), self.num_gaussians // len(occupancy_volumes))
                selected_indices = occ_indices[torch.randint(0, len(occ_indices), (num_to_sample,))]
                pts_img = selected_indices[:, [2, 1, 0]].float().to(device)
                pts_h = torch.cat([pts_img, torch.ones(len(pts_img), 1).to(device)], dim=-1)
                pts_world = (pts_h @ affines[i].to(device).T)[:, :3]
                all_pts_world.append(pts_world)
        
        if len(all_pts_world) > 0:
            init_pts = torch.cat(all_pts_world, dim=0)
            if len(init_pts) < self.num_gaussians:
                # Pad with random
                extra = torch.randn(self.num_gaussians - len(init_pts), 3).to(device) * 10.0
                init_pts = torch.cat([init_pts, extra], dim=0)
            else:
                init_pts = init_pts[:self.num_gaussians]
            
            self.means.data.copy_(init_pts)
            print(f"Initialized {self.num_gaussians} Gaussians from occupancy volumes.")

    def forward(self, z):
        """
        z: latent state from ODE
        Returns the deformed Gaussian parameters
        """
        B = z.shape[0]
        params = self.deformation_net(z).view(B, self.num_gaussians, 7)
        
        delta_means = params[..., :3] * 10.0
        delta_scales = params[..., 3:6]
        delta_opacities = params[..., 6:7]

        current_means = self.means.unsqueeze(0) + delta_means
        current_scales = torch.exp(torch.log(self.scales.unsqueeze(0) + 1e-8) + delta_scales)
        current_opacities = torch.sigmoid(self.opacities.unsqueeze(0) + delta_opacities + 10.0)
        
        return {
            'means': current_means,
            'scales': current_scales,
            'rotations': self.rotations,
            'opacities': current_opacities,
            'intensities': torch.sigmoid(self.intensities)
        }

    def evaluate_occupancy(self, points, gaussian_params, chunk_size=10000):
        """
        points: (B, N, 3) query points
        gaussian_params: dict of params from forward()
        """
        B, N, _ = points.shape
        device = points.device
        all_occupancies = []

        for i in range(0, N, chunk_size):
            chunk_pts = points[:, i:i+chunk_size] # (B, chunk_N, 3)
            
            # (B, 1, num_gaussians, 3) - (B, chunk_N, 1, 3) -> (B, chunk_N, num_gaussians, 3)
            means = gaussian_params['means'].unsqueeze(1)
            pts_exp = chunk_pts.unsqueeze(2)
            
            diff = pts_exp - means
            scales = gaussian_params['scales']
            if scales.dim() == 3: # (B, num_gaussians, 3)
                scales = scales.unsqueeze(1) # (B, 1, num_gaussians, 3)
            else:
                scales = scales.unsqueeze(0).unsqueeze(0)
            
            # (B, chunk_N, num_gaussians)
            exponent = -0.5 * torch.sum((diff / (scales + 1e-8))**2, dim=-1)
            
            opacities = gaussian_params['opacities']
            if opacities.dim() == 3: # (B, num_gaussians, 1)
                opacities = opacities.transpose(-1, -2) # (B, 1, num_gaussians)
            else:
                opacities = opacities.unsqueeze(0).transpose(-1, -2)
            
            # occupancy = sum(opacity * exp(exponent))
            weights = torch.exp(exponent)
            occupancy_chunk = torch.sum(weights * opacities, dim=-1)
            
            all_occupancies.append(occupancy_chunk)
            
        occupancy = torch.cat(all_occupancies, dim=1)
        return torch.clamp(occupancy, 0, 1)
