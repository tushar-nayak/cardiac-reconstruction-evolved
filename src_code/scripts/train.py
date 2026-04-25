import sys
import os

# Add the project root to sys.path to allow imports from src_code
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src_code.src.data_loaders.mitea_loader import MITEADataset
from src_code.src.models.ode_model import LatentEncoder, CardiacODE
from src_code.src.models.gaussian_model import GaussianModel
from src_code.src.utils.coord_utils import sample_volume_at_points
import tqdm

def collate_fn(batch):
    """
    Custom collate to handle varying occupancy volume shapes.
    """
    return {
        'subject': [d['subject'] for d in batch],
        'sparse_slices_ed': torch.stack([d['sparse_slices_ed'] for d in batch]),
        'sparse_slices_es': torch.stack([d['sparse_slices_es'] for d in batch]),
        'occupancy_ed': [d['occupancy_ed'] for d in batch],
        'occupancy_es': [d['occupancy_es'] for d in batch],
        'affine': torch.stack([d['affine'] for d in batch])
    }

import argparse

def train(args):
    # Config
    data_dir = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"
    run_dir = "runs/smoke_run_01"
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    latent_dim = 128
    num_gaussians = 5000
    lr = 1e-4
    epochs = args.epochs

    # Data
    dataset = MITEADataset(data_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Models
    encoder = LatentEncoder(input_channels=3, latent_dim=latent_dim).to(device)
    ode = CardiacODE(latent_dim=latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians=num_gaussians, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(ode.parameters()) + list(gaussian_model.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
        pbar = tqdm.tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader))
        for batch_idx, batch in pbar:
            if epoch == 0 and batch_idx == 0:
                gaussian_model.initialize_from_voxels(batch['occupancy_ed'], batch['affine'])
            
            # Input is (B, 3, 200, 200) - 3 sparse slices stacked
            input_ed = batch['sparse_slices_ed'].to(device)
            
            # 1. Encode to latent space
            z_ed = encoder(input_ed)
            
            # 2. Evolve latent space over time
            # t=0 is ED, t=1 is ES
            t = torch.tensor([0.0, 1.0]).to(device)
            zt = ode(z_ed, t) # (2, B, latent_dim)
            z_es = zt[1]
            
            # 3. Predict Gaussian parameters
            params_ed = gaussian_model(z_ed)
            params_es = gaussian_model(z_es)
            
            # 4. Calculate Loss
            # Supervise against ground truth occupancy labels
            B = z_ed.shape[0]
            num_samples = 2000

            # Sample points within a bounding box
            # Typical heart volume is not uniformly distributed in [-150, 150]
            # Let's sample points more intelligently: some random, some near GT occupied regions
            B = z_ed.shape[0]
            num_samples = 2000
            
            # Mixture of uniform and near-GT points
            query_points = []
            for i in range(B):
                # 50% Uniform random in world space
                # World coordinates seem to be in range [-200, 200] roughly
                pts_uni = (torch.rand(num_samples // 2, 3).to(device) * 400.0) - 200.0
                
                # 50% Near ground truth occupied voxels
                vol = batch['occupancy_ed'][i]
                occ_indices = torch.nonzero(vol)
                if len(occ_indices) > 0:
                    selected_indices = occ_indices[torch.randint(0, len(occ_indices), (num_samples // 2,))]
                    # Convert to world coords
                    pts_img = selected_indices[:, [2, 1, 0]].float().to(device) # (W, H, D)
                    # Add small noise
                    pts_img += torch.randn_like(pts_img) * 2.0
                    
                    pts_h = torch.cat([pts_img, torch.ones(len(pts_img), 1).to(device)], dim=-1)
                    affine_i = batch['affine'][i].to(device)
                    pts_world = (pts_h @ affine_i.T)[:, :3]
                else:
                    pts_world = (torch.rand(num_samples // 2, 3).to(device) * 300.0) - 150.0
                
                query_points.append(torch.cat([pts_uni, pts_world], dim=0))
            
            query_points = torch.stack(query_points)

            # Evaluate model occupancy
            occ_pred_ed = gaussian_model.evaluate_occupancy(query_points, params_ed)
            occ_pred_es = gaussian_model.evaluate_occupancy(query_points, params_es)

            # Get ground truth occupancy at those points
            target_occ_ed = sample_volume_at_points(batch['occupancy_ed'], query_points, batch['affine'].to(device))
            target_occ_es = sample_volume_at_points(batch['occupancy_es'], query_points, batch['affine'].to(device))

            # Simple BCE Loss
            loss_ed = F.binary_cross_entropy(occ_pred_ed, target_occ_ed)
            loss_es = F.binary_cross_entropy(occ_pred_es, target_occ_es)
            loss = loss_ed + loss_es

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Save checkpoint periodically
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'ode_state_dict': ode.state_dict(),
                'gaussian_model_state_dict': gaussian_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args)
