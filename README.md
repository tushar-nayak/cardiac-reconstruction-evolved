# cardiac-reconstruction-evolved

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TorchDiffEq](https://img.shields.io/badge/TorchDiffEq-Latest-orange.svg)

CardioEvolve-4DGS is a framework designed to recover continuous 3D ventricular occupancy fields from highly sparse (2-3 slices) and temporally distinct 2D echocardiography or cardiac MRI sequences. 

Standard 3D/4D Gaussian Splatting (3DGS) relies on dense multi-view RGB images. This repository adapts the differential rendering pipeline to accumulate radiological density and volumetric depth. By treating the cardiac cycle as an explicit, continuous-time dynamics problem, it merges Neural ODE forecasting with Vision-Language Model (VLM) embeddings for comprehensive spatiotemporal tracking.

## Key Features

* **Neural ODE Forecasting:** Overcomes the temporal sparsity of ultrasound and MRI scans. Utilizing `TorchDiffEq` and a hybrid Attention U-Net backend, the system evolves the latent states of the 3D Gaussian representation, smoothly interpolating ventricular contraction between observed frames.
* **Radiological Splatting:** Rewrites the standard rasterizer to accumulate MRI/Ultrasound signal intensity and depth, allowing for the direct visualization of tissue density rather than alpha-composited light.
* **Stable Occupancy Aggregation:** Replaced the naive sum-and-clamp aggregation with a physically-grounded volumetric formulation: `1 - exp(-density)`. This prevents saturation and allows for distinct cavity/wall separation.
* **Aggressive Negative Mining:** Point sampling logic specifically targets the heart cavity (the "donut hole") with high-weight supervision to ensure anatomically correct reconstructions.
* **Temporal Semantic Tracking:** Injects multi-modal VLM embeddings into the explicit Gaussian points. This enables dynamic, open-vocabulary queries such as isolating a specific valve or tracking the volumetric shift of a semantically-labeled region across the entire cardiac cycle.

## System Architecture

1. **Sparse Pose Optimization:** Learns slice alignment terms so the reconstruction can better match the observed 2D inputs.
2. **Geometry Grounding:** Gaussian centers and opacities are initialized from occupied voxels to anchor the learned field to the observed anatomy.
3. **Continuous Evolution:** The ODE solver drives the 4D Gaussian deformation field, morphing the initialized Gaussians across the cardiac time steps to ensure anatomically plausible continuous reconstruction.
4. **Visualization and Diagnostics:** Unified evaluation scripts produce comparison figures and 4D animation outputs with explicit coordinate sanity checks.

## Current Results

* **Stabilized Occupancy**: The model no longer collapses into a fully saturated field.
* **Mean Occupancy at GT**: **0.9866** (where 1.0 is fully occupied) achieved during stabilization.
* **Spatial Fidelity**: Successfully reconstructed circular axial cross-sections and conical longitudinal profiles from sparse orthogonal slices.
* **4D Consistency**: Smooth ventricular contraction animations generated via Neural ODE, accurately depicting physiological systolic thickening.

## Current Status

* **Phase 1-3 Complete:** Core architecture, 3D occupancy pipeline, Radiological Rasterizer, Pose Optimization, and VLM semantic embeddings are fully implemented.
* **Stabilization Phase:** Focused on perfecting the static 3D geometry and cavity/wall separation before re-introducing full 4D dynamics.
* **Visualization Ready:** 4D animation and 3D evaluation scripts are available for local review.

## Running the Scripts

### Training
```bash
python3 src_code/scripts/train.py \
  --data-dir /path/to/mitea \
  --run-dir runs/stabilization_v01 \
  --epochs 50
```

### 4D Animation
```bash
python3 src_code/scripts/animate_4d.py
```

### 3D Evaluation
```bash
python3 src_code/scripts/evaluate_3d.py --checkpoint runs/stabilization_v01/checkpoint_epoch_20.pth
```

The 3D evaluation writes a comparison image and `metrics.json` with occupancy accuracy, IoU, and regional occupancy diagnostics. Checkpoints, diagnostics, and animations are saved in `runs/`.

## Documentation

For a detailed technical breakdown of the methodology and results, please refer to the [Report](report.md).

## Installation & Dependencies

Requires `torch`, `torchvision`, `torchdiffeq`, and `pytorch3d`.

## Relation to Prior Project

This repository is distinct from the earlier [`cardiac-volume-reconstruction`](https://github.com/tushar-nayak/cardiac-volume-reconstruction) project.

The older project is the stronger baseline in terms of end-to-end experimental maturity. It focuses on reconstructing 3D cardiac volumes from sparse 2D echo views using coordinate-based implicit neural representations, transfer learning, and meta-learning for fast subject adaptation. Its central result is a multi-subject reconstruction pipeline with reported Dice and IoU summaries across saved experiments.

This repository explores a different modeling direction. Instead of representing anatomy with an MLP-based implicit field, it uses explicit Gaussian occupancy primitives, differentiable slice rendering, and diagnostic tooling built around occupancy stability, mesh extraction, and visual inspection. In practice, this makes `cardiac-reconstruction-evolved` more of a Gaussian-field reconstruction prototype and experimentation branch, whereas `cardiac-volume-reconstruction` is the more complete INR-based cardiac reconstruction project.

In short:

* `cardiac-volume-reconstruction`: established INR / meta-learning cardiac reconstruction pipeline with broader evaluation.
* `cardiac-reconstruction-evolved`: experimental Gaussian occupancy reconstruction path focused on explicit primitives, stabilization, and mesh-oriented diagnostics.
