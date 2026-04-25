import sys
import os

# Add the project root to sys.path to allow relative imports from code/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from code.src.data_loaders.mitea_loader import MITEADataset
from code.src.models.ode_model import LatentEncoder, CardiacODE
from code.src.models.gaussian_model import GaussianModel
import tqdm

def train():
    # Config
    data_dir = "/home/sofa/host_dir/cardiac_reconstruction_project/cap-mitea/mitea"
    run_dir = "runs/smoke_run_01"
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    latent_dim = 128
    num_gaussians = 5000
    lr = 1e-4
    epochs = 50

    # Data
    dataset = MITEADataset(data_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Models
    encoder = LatentEncoder(input_channels=1, latent_dim=latent_dim).to(device)
    ode = CardiacODE(latent_dim=latent_dim).to(device)
    gaussian_model = GaussianModel(num_gaussians=num_gaussians, latent_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(ode.parameters()) + list(gaussian_model.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            subject = batch['subject']
            # Take the first sparse slice for ED as input
            # (In reality, we'd combine all sparse slices)
            input_ed = batch['sparse_slices_ed'][0].unsqueeze(1).to(device) # (B, 1, H, W)
            
            # 1. Encode to latent space
            z_ed = encoder(input_ed)
            
            # 2. Evolve latent space over time
            # t=0 is ED, t=1 is ES
            t = torch.tensor([0.0, 1.0]).to(device)
            zt = ode(z_ed, t) # (2, B, latent_dim)
            z_es = zt[1]
            
            # 3. Deform Gaussians
            params_ed = gaussian_model(z_ed)
            params_es = gaussian_model(z_es)
            
            # 4. Calculate Loss
            # Supervise against ground truth occupancy labels
            # Sample random points in the volume
            B = z_ed.shape[0]
            num_samples = 1000
            
            # Random points in [-1, 1] range
            query_points = (torch.rand(B, num_samples, 3).to(device) * 2.0) - 1.0
            
            # Evaluate model occupancy
            occ_pred_ed = gaussian_model.evaluate_occupancy(query_points, params_ed)
            
            # Get ground truth occupancy at those points
            # (Need to map query_points back to voxel indices)
            # For now, let's just use a dummy target or simplify the supervision
            target_occ_ed = torch.ones_like(occ_pred_ed) * 0.5 # Placeholder
            
            loss = F.binary_cross_entropy(occ_pred_ed, target_occ_ed)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})

if __name__ == "__main__":
    train()
