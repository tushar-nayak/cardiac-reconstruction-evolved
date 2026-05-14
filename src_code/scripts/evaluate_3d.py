import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder
from src_code.src.models.gaussian_model import GaussianModel

DEFAULT_DATA_DIR = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"

def evaluate_3d(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval_3d")
    os.makedirs(output_dir, exist_ok=True)

    encoder = LatentEncoder(3, args.latent_dim).to(device)
    gaussian_model = GaussianModel(args.num_gaussians, args.latent_dim).to(device)

    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
    encoder.eval(); gaussian_model.eval()

    dataset = MITEADataset(args.data_dir, split=args.split)
    sample = dataset[args.sample_index]
    gt_vol = sample['occupancy_ed'].cpu().numpy() # (D, H, W)
    affine = sample['affine']
    D, H, W = gt_vol.shape

    test_idx = torch.tensor([D//2, H//2, W//2]).float()
    test_pts_img = test_idx[[2, 1, 0]].view(1, 3)
    pts_h = torch.cat([test_pts_img, torch.ones(1, 1)], dim=-1)
    pts_world = (pts_h @ affine.T)[:, :3]
    
    inv_affine = torch.inverse(affine)
    pts_h_world = torch.cat([pts_world, torch.ones(1, 1)], dim=-1)
    back_img = (pts_h_world @ inv_affine.T)[:, :3]
    print(f"Coord Test - Original: {test_pts_img.tolist()}, Recovered: {back_img.tolist()}")

    z_idx = D // 2
    u = torch.linspace(0, W - 1, args.slice_resolution)
    v = torch.linspace(0, H - 1, args.slice_resolution)
    grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
    pts_img_axial = torch.stack([grid_u, grid_v, torch.ones_like(grid_u) * z_idx], dim=-1).reshape(-1, 3)
    
    y_idx = H // 2
    u = torch.linspace(0, W - 1, args.slice_resolution)
    w = torch.linspace(0, D - 1, args.slice_resolution)
    grid_w, grid_u = torch.meshgrid(w, u, indexing='ij')
    pts_img_coronal = torch.stack([grid_u, torch.ones_like(grid_u) * y_idx, grid_w], dim=-1).reshape(-1, 3)

    with torch.no_grad():
        input_ed = sample['sparse_slices_ed'].unsqueeze(0).to(device)
        z_ed = encoder(input_ed)
        params_ed = gaussian_model(z_ed)
        
        pts_axial_world = (torch.cat([pts_img_axial, torch.ones(pts_img_axial.shape[0], 1)], dim=-1) @ affine.T)[:, :3].to(device)
        occ_axial = gaussian_model.evaluate_occupancy(pts_axial_world.unsqueeze(0), params_ed).view(args.slice_resolution, args.slice_resolution).cpu().numpy()
        
        pts_coronal_world = (torch.cat([pts_img_coronal, torch.ones(pts_img_coronal.shape[0], 1)], dim=-1) @ affine.T)[:, :3].to(device)
        occ_coronal = gaussian_model.evaluate_occupancy(pts_coronal_world.unsqueeze(0), params_ed).view(args.slice_resolution, args.slice_resolution).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(gt_vol[D//2, :, :], cmap='gray')
    axes[0, 0].set_title("GT Axial")
    axes[0, 1].imshow(occ_axial, cmap='magma', vmin=0, vmax=1)
    axes[0, 1].set_title("Pred Axial (Stabilized)")
    
    axes[1, 0].imshow(gt_vol[:, H//2, :], cmap='gray')
    axes[1, 0].set_title("GT Coronal")
    axes[1, 1].imshow(occ_coronal, cmap='magma', vmin=0, vmax=1)
    axes[1, 1].set_title("Pred Coronal (Stabilized)")

    for ax in axes.ravel(): ax.axis('off')
    
    save_path = os.path.join(output_dir, f"comparison_v01.png")
    plt.savefig(save_path)
    print(f"Evaluation saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=1000)
    parser.add_argument("--slice-resolution", type=int, default=128)
    evaluate_3d(parser.parse_args())
