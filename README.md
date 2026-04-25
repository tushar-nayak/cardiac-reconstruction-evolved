# CardioEvolve-4DGS: Continuous-Time 4D Semantic Splatting for Echocardiography

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TorchDiffEq](https://img.shields.io/badge/TorchDiffEq-Latest-orange.svg)

CardioEvolve-4DGS is a framework designed to recover continuous 3D ventricular occupancy fields from highly sparse (2-3 slices) and temporally distinct 2D echocardiography or cardiac MRI sequences. 

Standard 3D/4D Gaussian Splatting (3DGS) relies on dense multi-view RGB images. This repository adapts the differential rendering pipeline to accumulate radiological density and volumetric depth. By treating the cardiac cycle as an explicit, continuous-time dynamics problem, it merges Neural ODE forecasting with Vision-Language Model (VLM) embeddings for comprehensive spatiotemporal tracking.

## 🚀 Key Features

* **Neural ODE Forecasting:** Overcomes the temporal sparsity of ultrasound and MRI scans. Utilizing `TorchDiffEq` and a hybrid Attention U-Net backend, the system evolves the latent states of the 3D Gaussian representation, smoothly interpolating ventricular contraction between observed frames.
* **Radiological Splatting:** Rewrites the standard rasterizer to accumulate MRI/Ultrasound signal intensity and depth, allowing for the direct visualization of tissue density rather than alpha-composited light.
* **High-Frequency Boundary Recovery:** Leverages Fourier positional encodings to capture fine anatomical details and sharp boundary edges in the cardiac wall, supervised by volumetric BCE point sampling in unobserved 3D regions.
* **Temporal Semantic Tracking:** Injects multi-modal VLM embeddings into the explicit Gaussian points. This enables dynamic, open-vocabulary queries—such as isolating a specific valve or tracking the volumetric shift of a semantically-labeled region across the entire cardiac cycle.

## 🛠️ System Architecture

1. **Sparse Pose Optimization:** Implements a differentiable multi-view projection with learnable pose optimization. This establishes joint slice alignment and initial geometry recovery from sparse 2D echo slices.
2. **Continuous Evolution:** The ODE solver drives the 4D Gaussian deformation field, morphing the initialized Gaussians across the cardiac time steps to ensure anatomically plausible continuous reconstruction.
3. **Semantic Querying:** The resulting 4D space can be interactively sliced and queried using text prompts, tracking functional regions of the heart over time.

## 📦 Installation & Dependencies

Requires `torch`, `torchvision`, `torchdiffeq`, and `pytorch3d`.

```bash
git clone [https://github.com/tushar-nayak/CardioEvolve-4DGS.git](https://github.com/tushar-nayak/CardioEvolve-4DGS.git)
cd CardioEvolve-4DGS

# Install standard dependencies
pip install -r requirements.txt

# Compile the custom radiological Gaussian rasterizer
cd submodules/diff-radiological-rasterization
pip install .
