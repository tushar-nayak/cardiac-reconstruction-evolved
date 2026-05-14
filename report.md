# Cardiac Reconstruction from Sparse Imaging via Stabilized Gaussian Occupancy Fields

## Abstract
Accurate 3D reconstruction from sparse 2D imaging remains a useful testbed for evaluating latent-field models, occupancy aggregation, and coordinate handling. This project now focuses on stabilized Gaussian occupancy fields and qualitative diagnostics rather than a full temporal reconstruction claim. The current codebase includes a probabilistic occupancy aggregator, voxel-grounded Gaussian initialization, and comparison figures generated from the latest runs.

## Introduction
Standard sparse imaging setups often lack enough context for direct reconstruction. The current implementation uses a 3D occupancy field inferred from sparse slices, then validates the output with direct slice comparisons and cavity-versus-wall diagnostics. The emphasis is on making the representation numerically stable and easy to inspect.

## Materials and Methods

### 1. Gaussian Occupancy Fields
The geometry is represented by a set of anisotropic Gaussian kernels. Each Gaussian is parameterized by a mean position, scale, rotation, and opacity. The current stabilization pass reduces the number of Gaussians and grounds initial positions from occupied voxels.

### 2. Latent Dynamics
The earlier ODE-based motion model is still present in the repository, but the latest stabilization path focuses on static reconstruction. This reduces one major source of instability while preserving a path for future temporal experiments.

### 3. Differentiable Rasterization
The project retains a custom rasterizer for slice supervision and intensity matching. The present emphasis is on verifying whether rendered slices align with the observed input rather than matching a full clinical rendering pipeline.

### 4. Multi-view Pose Optimization
A differentiable pose optimizer remains part of the training stack. It is used to reduce slice misalignment and improve correspondence between sparse observations and the occupancy field.

### 5. Semantic Embedding
Semantic embeddings remain in the model but are currently a secondary concern relative to occupancy stability and geometry grounding.

## Results

### 1. Quantitative Status
The latest run produces stabilized checkpoints and a 3D comparison artifact. The diagnostic emphasis is now on whether the model separates cavity from wall and whether the predicted occupancy is no longer uniformly saturated.

### 2. Spatial Fidelity
The current figures show a visible target-shaped envelope in the predicted slices, but the result remains smoother and thicker than the ground truth. This indicates partial success: the occupancy field is no longer collapsed, yet it still lacks a sharp internal cavity structure.

### 3. Temporal Dynamics
The animation artifact is currently best interpreted as a diagnostic output. It is useful for checking continuity in the latent evolution path, but it should not yet be treated as a validated motion model.

## Discussion
The stabilization pass improved the numerical behavior of the occupancy field, but the model is still underconstrained relative to the complexity of the reconstruction problem. The main remaining challenge is to sharpen the cavity/wall separation without reintroducing saturation or instability. Future work should focus on stronger geometric priors, tighter coordinate validation, and more explicit supervision of the hollow interior structure.

## Conclusion
The current state of the project is best described as a stabilized sparse reconstruction prototype. It produces more meaningful outputs than the earlier saturated version, but it still does not fully recover a crisp internal cavity structure or a validated temporal evolution.
