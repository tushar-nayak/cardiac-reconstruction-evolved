import sys
import os
import argparse
import torch

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder
from src_code.src.models.gaussian_model import GaussianModel

DEFAULT_DATA_DIR = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"

def check_donut(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = LatentEncoder(3, args.latent_dim).to(device)
    gaussian_model = GaussianModel(args.num_gaussians, args.latent_dim).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    gaussian_model.load_state_dict(checkpoint['gaussian_model_state_dict'])
    encoder.eval(); gaussian_model.eval()

    dataset = MITEADataset(args.data_dir, split=args.split)
    sample = dataset[args.sample_index]
    gt_vol = sample['occupancy_ed']
    affine = sample['affine']
    
    occ_indices = torch.nonzero(gt_vol)
    if len(occ_indices) == 0:
        raise ValueError("Selected sample has no occupied voxels.")
    centroid_img = occ_indices.float().mean(dim=0)[[2, 1, 0]] # (W, H, D)
    
    # Points in cavity (near centroid)
    pts_neg_img = centroid_img.view(1, 3) + torch.randn(100, 3) * 5.0
    pts_neg_h = torch.cat([pts_neg_img, torch.ones(100, 1)], dim=-1)
    pts_neg_world = (pts_neg_h @ affine.T)[:, :3].to(device)
    
    # Points on wall (known occupied)
    pts_pos_img = occ_indices[torch.randint(0, len(occ_indices), (100,))][:, [2, 1, 0]].float()
    pts_pos_h = torch.cat([pts_pos_img, torch.ones(100, 1)], dim=-1)
    pts_pos_world = (pts_pos_h @ affine.T)[:, :3].to(device)

    with torch.no_grad():
        z_ed = encoder(sample['sparse_slices_ed'].unsqueeze(0).to(device))
        params_ed = gaussian_model(z_ed)
        
        occ_neg = gaussian_model.evaluate_occupancy(pts_neg_world.unsqueeze(0), params_ed)
        occ_pos = gaussian_model.evaluate_occupancy(pts_pos_world.unsqueeze(0), params_ed)
        
        print(f"--- Quant Verification (Stabilized Model) ---")
        print(f"Mean Occupancy in Cavity (Target: Low): {occ_neg.mean().item():.4f}")
        print(f"Mean Occupancy on Wall (Target: High): {occ_pos.mean().item():.4f}")
        
        if occ_pos.mean() > occ_neg.mean() * 1.5:
            print("SUCCESS: Model shows distinct 'donut' structure (Wall > Cavity).")
        else:
            print("FAILURE: Model is still saturated or collapsed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-gaussians", type=int, default=1000)
    check_donut(parser.parse_args())
