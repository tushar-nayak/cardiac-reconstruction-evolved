import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianModel(nn.Module):
    def __init__(self, num_gaussians=10000, latent_dim=128):
        super(GaussianModel, self).__init__()
        self.num_gaussians = num_gaussians
        
        # Initial parameters for Gaussians (can be initialized from sparse points)
        self.means = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * 0.01)
        self.rotations = nn.Parameter(torch.tensor([1., 0., 0., 0.]).repeat(num_gaussians, 1))
        self.opacities = nn.Parameter(torch.ones(num_gaussians, 1) * 0.1)
        self.intensities = nn.Parameter(torch.ones(num_gaussians, 1) * 0.5)

        # MLP to predict deformations from latent state
        self.deformation_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_gaussians * 3) # Predicts delta_means
        )

    def get_covariance(self):
        # Placeholder for building covariance matrix from scales and rotations (quaternions)
        # For now, just use scales as axis-aligned for simplicity
        return self.scales

    def forward(self, z):
        """
        z: latent state from ODE
        Returns the deformed Gaussian parameters
        """
        delta_means = self.deformation_net(z).view(-1, self.num_gaussians, 3)
        current_means = self.means.unsqueeze(0) + delta_means
        
        return {
            'means': current_means,
            'scales': self.scales,
            'rotations': self.rotations,
            'opacities': torch.sigmoid(self.opacities),
            'intensities': torch.sigmoid(self.intensities)
        }

    def evaluate_occupancy(self, points, gaussian_params):
        """
        points: (B, N, 3) query points
        gaussian_params: dict of params from forward()
        """
        # Simplified occupancy calculation: sum of RBFs
        # (B, 1, num_gaussians, 3) - (B, N, 1, 3) -> (B, N, num_gaussians, 3)
        means = gaussian_params['means'].unsqueeze(1)
        points = points.unsqueeze(2)
        
        diff = points - means
        scales = gaussian_params['scales'].unsqueeze(0).unsqueeze(0)
        
        # (B, N, num_gaussians)
        exponent = -0.5 * torch.sum((diff / (scales + 1e-8))**2, dim=-1)
        weights = torch.exp(exponent)
        
        opacities = gaussian_params['opacities'].unsqueeze(0).transpose(-1, -2)
        occupancy = torch.sum(weights * opacities, dim=-1)
        
        return torch.clamp(occupancy, 0, 1)
