# cardiac-reconstruction-evolved

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TorchDiffEq](https://img.shields.io/badge/TorchDiffEq-Latest-orange.svg)

CardioEvolve-4DGS is a framework designed to recover continuous 3D ventricular occupancy fields from highly sparse (2-3 slices) and temporally distinct 2D echocardiography or cardiac MRI sequences. 

Standard 3D/4D Gaussian Splatting (3DGS) relies on dense multi-view RGB images. This repository adapts the differential rendering pipeline to accumulate radiological density and volumetric depth. By treating the cardiac cycle as an explicit, continuous-time dynamics problem, it merges Neural ODE forecasting with Vision-Language Model (VLM) embeddings for comprehensive spatiotemporal tracking.

## Key Features

* **Neural ODE Forecasting:** Overcomes the temporal sparsity of ultrasound and MRI scans. Utilizing `TorchDiffEq` and a hybrid Attention U-Net backend, the system evolves the latent states of the 3D Gaussian representation, smoothly interpolating ventricular contraction between observed frames.
* **Radiological Splatting:** Rewrites the standard rasterizer to accumulate MRI/Ultrasound signal intensity and depth, allowing for the direct visualization of tissue density rather than alpha-composited light.
* **High-Frequency Boundary Recovery:** Leverages Fourier positional encodings to capture fine anatomical details and sharp boundary edges in the cardiac wall, supervised by volumetric BCE point sampling in unobserved 3D regions.
* **Temporal Semantic Tracking:** Injects multi-modal VLM embeddings into the explicit Gaussian points. This enables dynamic, open-vocabulary queriessuch as isolating a specific valve or tracking the volumetric shift of a semantically-labeled region across the entire cardiac cycle.

## System Architecture

1. **Sparse Pose Optimization:** Implements a differentiable multi-view projection with learnable pose optimization. This establishes joint slice alignment and initial geometry recovery from sparse 2D echo slices.
2. **Continuous Evolution:** The ODE solver drives the 4D Gaussian deformation field, morphing the initialized Gaussians across the cardiac time steps to ensure anatomically plausible continuous reconstruction.
3. **Semantic Querying:** The resulting 4D space can be interactively sliced and queried using text prompts, tracking functional regions of the heart over time.

## Current Results

* **Occupancy Accuracy**: **98.5%** (on validation points $> 0.5$ occupancy).
* **Mean Occupancy at GT**: **0.9866** (where $1.0$ is fully occupied).
* **Radiological Intensity Loss**: Successfully reached $\sim 149.8$ using raw slice intensity supervision.
* **Spatial Fidelity**: Successfully reconstructed the circular axial cross-section and conical longitudinal profile (apex) of the left ventricle from sparse orthogonal slices.
* **4D Consistency**: Smooth ventricular contraction animations generated via Neural ODE, accurately depicting physiological systolic thickening.

## Current Status

* **Phase 1-3 Complete:** Core architecture, 3D occupancy pipeline, Radiological Rasterizer, Pose Optimization, and VLM semantic embeddings are fully implemented.
* **Smoke Run Verified:** Successfully trained and validated on MITEA data subset with joint geometric and intensity supervision.
* **Visualization Ready:** 4D animation generation script implemented.

## Running the Scripts

### Training
```bash
python3 src_code/scripts/train.py --epochs 50
```

### 4D Animation
```bash
python3 src_code/scripts/animate_4d.py
```
Checkpoints and animations will be saved in `runs/smoke_run_01/`.

## Documentation

For a detailed technical breakdown of the methodology and results, please refer to the [Scientific Manuscript](report.md).

## Installation & Dependencies

Requires `torch`, `torchvision`, `torchdiffeq`, and `pytorch3d`.
