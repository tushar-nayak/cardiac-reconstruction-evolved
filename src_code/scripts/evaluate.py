import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder, CardiacODE
from src_code.src.models.gaussian_model import GaussianModel

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"
    checkpoint_path = "runs/smoke_run_01/checkpoint_epoch_50.pth"
    output_dir = "runs/smoke_run_01/eval_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    latent_dim = 128
    num_gaussians = 5000
    encoder = LatentEncoder(input_channels=3, latent_dim=latent_dim).to(device)
    ode = CardiacODE(latent_dim=latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians=num_gaussians, latent_dim=latent_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    ode.load_state_dict(checkpoint['ode_state_dict'])
    gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
    
    encoder.eval()
    ode.eval()
    gaussian_model.eval()

    # Load a validation sample
    dataset = MITEADataset(data_dir, split='val')
    sample = dataset[0]
    input_ed = sample['sparse_slices_ed'].unsqueeze(0).to(device)
    gt_occ_ed = sample['occupancy_ed']
    affine = sample['affine']

    with torch.no_grad():
        # Predict
        z_ed = encoder(input_ed)
        params_ed = gaussian_model(z_ed)
        
        # Create a grid for visualization (128x128x128)
        # We'll sample points in the volume to reconstruct it
        D, H, W = gt_occ_ed.shape
        # Create grid in image coordinates
        z_range = torch.linspace(0, D-1, 64)
        y_range = torch.linspace(0, H-1, 64)
        x_range = torch.linspace(0, W-1, 64)
        grid_z, grid_y, grid_x = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
        pts_img = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device) # (N, 3)
        
        # Transform to world coordinates
        pts_h = torch.cat([pts_img, torch.ones(pts_img.shape[0], 1).to(device)], dim=-1)
        affine_th = affine.float().to(device) if torch.is_tensor(affine) else torch.from_numpy(affine).float().to(device)
        pts_world = (pts_h @ affine_th.T)[:, :3]
        
        # Evaluate occupancy
        occ_pred = gaussian_model.evaluate_occupancy(pts_world.unsqueeze(0), params_ed)
        occ_pred = occ_pred.view(64, 64, 64).cpu().numpy()

    # Plot middle slices
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_occ_ed[D//2].cpu().numpy(), cmap='gray')
    axes[0].set_title("Ground Truth (Middle Slice)")
    axes[1].imshow(occ_pred[32], cmap='gray')
    axes[1].set_title("Predicted (Middle Slice)")
    
    plt.savefig(os.path.join(output_dir, "comparison_ed.png"))
    print(f"Evaluation complete. Result saved to {output_dir}/comparison_ed.png")

if __name__ == "__main__":
    evaluate()
