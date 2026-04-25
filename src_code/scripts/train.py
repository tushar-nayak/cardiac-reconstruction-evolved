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
from src_code.src.models.pose_model import PoseOptimizer
from src_code.src.utils.coord_utils import sample_volume_at_points
from src_code.src.rendering.rasterizer import RadiologicalRasterizer
import tqdm
import argparse

def collate_fn(batch):
    """
    Custom collate to handle varying occupancy volume shapes.
    """
    return {
        'subject': [d['subject'] for d in batch],
        'index': torch.tensor([d['index'] for d in batch]),
        'sparse_slices_ed': torch.stack([d['sparse_slices_ed'] for d in batch]),
        'sparse_slices_es': torch.stack([d['sparse_slices_es'] for d in batch]),
        'poses_ed': [d['poses_ed'] for d in batch],
        'occupancy_ed': [d['occupancy_ed'] for d in batch],
        'occupancy_es': [d['occupancy_es'] for d in batch],
        'affine': torch.stack([d['affine'] for d in batch])
    }

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
    rasterizer = RadiologicalRasterizer().to(device)
    pose_optimizer = PoseOptimizer(num_subjects=len(dataset)).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(ode.parameters()) + 
        list(gaussian_model.parameters()) + list(pose_optimizer.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
        pbar = tqdm.tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader))
        for batch_idx, batch in pbar:
            if epoch == 0 and batch_idx == 0:
                gaussian_model.initialize_from_voxels(batch['occupancy_ed'], batch['affine'])
            
            # Input is (B, 3, 200, 200) - 3 sparse slices stacked
            input_ed = batch['sparse_slices_ed'].to(device)
            subject_indices = batch['index'].to(device)
            
            # 1. Encode to latent space
            z_ed = encoder(input_ed)
            
            # 2. Evolve latent space over time
            t = torch.tensor([0.0, 1.0]).to(device)
            zt = ode(z_ed, t)
            z_es = zt[1]
            
            # 3. Predict Gaussian parameters
            params_ed = gaussian_model(z_ed)
            params_es = gaussian_model(z_es)
            
            # 4. Refine Poses
            refined_poses_ed = pose_optimizer(subject_indices, batch['poses_ed'])

            # 5. Calculate Loss
            B = z_ed.shape[0]
            num_samples = 2000
            query_points = []
            for i in range(B):
                pts_uni = (torch.rand(num_samples // 2, 3).to(device) * 400.0) - 200.0
                vol = batch['occupancy_ed'][i]
                occ_indices = torch.nonzero(vol)
                if len(occ_indices) > 0:
                    selected_indices = occ_indices[torch.randint(0, len(occ_indices), (num_samples // 2,))]
                    pts_img = selected_indices[:, [2, 1, 0]].float().to(device)
                    pts_img += torch.randn_like(pts_img) * 2.0
                    pts_h = torch.cat([pts_img, torch.ones(len(pts_img), 1).to(device)], dim=-1)
                    affine_i = batch['affine'][i].to(device)
                    pts_world = (pts_h @ affine_i.T)[:, :3]
                else:
                    pts_world = (torch.rand(num_samples // 2, 3).to(device) * 400.0) - 200.0
                query_points.append(torch.cat([pts_uni, pts_world], dim=0))
            query_points = torch.stack(query_points)

            occ_pred_ed = gaussian_model.evaluate_occupancy(query_points, params_ed)
            occ_pred_es = gaussian_model.evaluate_occupancy(query_points, params_es)
            target_occ_ed = sample_volume_at_points(batch['occupancy_ed'], query_points, batch['affine'].to(device))
            target_occ_es = sample_volume_at_points(batch['occupancy_es'], query_points, batch['affine'].to(device))
            loss_occ = F.binary_cross_entropy(occ_pred_ed, target_occ_ed) + F.binary_cross_entropy(occ_pred_es, target_occ_es)

            # 6. Radiological Intensity Supervision
            loss_img = torch.tensor(0.0).to(device)
            for b in range(B):
                for s in range(3):
                    pose = refined_poses_ed[b][s]
                    params_b = {k: v[b:b+1] if v.dim() > 2 else v for k, v in params_ed.items()}
                    img_pred = rasterizer(params_b, pose)
                    img_target = batch['sparse_slices_ed'][b, s].to(device).unsqueeze(0)
                    loss_img += F.mse_loss(img_pred, img_target)
            
            # 7. Semantic Consistency Loss (over time)
            # Ensure embeddings for the same Gaussians stay similar during deformation
            loss_sem = F.mse_loss(params_ed['semantics'], params_es['semantics'])

            loss = loss_occ + 1.0 * loss_img + 0.1 * loss_sem

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'img_l': f"{loss_img.item():.4f}", 'sem_l': f"{loss_sem.item():.4f}"})

        # Save checkpoint periodically
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'ode_state_dict': ode.state_dict(),
                'gaussian_model_state_dict': gaussian_model.state_dict(),
                'pose_optimizer_state_dict': pose_optimizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train(args)
