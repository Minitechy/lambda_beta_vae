# λβ-VAE Disentanglement Experiments
This repository implements β-VAE and λβ-VAE models to evaluate disentanglement metrics on the dSprites, Shapes3D, and MPI3D datasets. It includes both nonlinear (convolutional) and linear implementations. The β-VAE baselines are trained first, followed by continued training with an additional λ term in the loss function to assess its impact on reconstruction quality and disentanglement performance.

## Features
- **Models**: Convolutional encoder-decoder VAE architecture (nonlinear) and linear encoder-decoder (linear).
- **Loss Functions**: β-VAE (reconstruction + β × KL divergence) and λβ-VAE (reconstruction + β × KL divergence + λ × L2 loss).
- **Datasets**: dSprites, Shapes3D, and MPI3D (included in the `data` folder).
- **Metrics**: Mutual Information Gap (MIG), Separated Attribute Predictability (SAP), and $I_m$ score. For nonlinear: also includes Negative Log-Likelihood (NLL).
- **Visualizations**: Image reconstruction grids, latent traversal GIFs, mutual information heatmaps, boxplots for β-VAE results, heatmaps for λβ-VAE results, and interactive Plotly heatmaps for weighted scores.
- **Reproducibility**: Multi-seed experiments with fixed random seeds.
- **Linear Methods**: Fixed-point iteration and AdamW optimization for linear models.

## Installation
1. Download and unzip the repository archive from GitHub.
2. Ensure the `data` folder contains the datasets.
3. Install dependencies using Python 3.8+:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Running Linear β-VAE and λβ-VAE Experiments (Method 1: Direct Optimization)
Train and evaluate linear β-VAE and λβ-VAE models using fixed-point iteration and AdamW optimization:
```
python linear_betavae_optimizer.py
```
- Results are saved to `./linear_betavae_results/`.
- Includes metrics (reconstruction error, SAP, Im), boxplots, and heatmaps.
- Aggregated visualizations: `./linear_betavae_results/boxplots_betavae_n*.png` and `./linear_betavae_results/heatmap_lambda_betavae_n*.png`.
- Pre-computed results: Download `linear_betavae_results.zip` from the repository attachments.

### Running Linear β-VAE and λβ-VAE Experiments (Method 2: Alternative Approach)
For an alternative linear implementation, refer to the additional script:
```
python linear_betavae.py
```
- This provides another method to run linear experiments before proceeding to nonlinear datasets.
- Results are compatible with the visualization tools below.

### Running Nonlinear β-VAE Experiments
Train and evaluate nonlinear β-VAE models:
```
python main_beta.py
```
- Results are saved to `./betavae_results/<dataset>/`.
- Includes trained models, metrics, visualizations, and summary files.
- Aggregated boxplots: `./betavae_results/metric_boxplots_across_datasets.png`.

### Running Nonlinear λβ-VAE Experiments
Continue training from β-VAE checkpoints (run `main_beta.py` first):
```
python main_lambda.py
```
- Results are saved to `./lambda_betavae_results/<dataset>/`.
- Includes updated models, metrics, visualizations, and summary files.
- Aggregated heatmaps: `./lambda_betavae_results/metric_heatmaps_across_datasets.png`.
- Pre-computed results: Download `iterative_heatmap_results.zip` from the repository attachments (includes interactive heatmaps).

### Plotting Interactive Heatmaps
After running nonlinear experiments, visualize weighted scores with an interactive slider to balance reconstruction (NLL) and disentanglement (MIG), highlighting optimal β-λ pairs:
```
python interactive_heatmap_visualizer.py
```
- Results are saved to `./lambda_betavae_results/` as HTML files (e.g., `interactive_weighted_score_heatmap_<dataset>.html`).
- Use the slider to adjust weights and find the best β-λ pair for different priorities.

## Configuration
Customize hyperparameters (e.g., β values, λ values, number of seeds, training steps) in `config.py`.

## Datasets
We do not redistribute datasets. Please download them from the official sources below and place the files in the `data` folder so loaders work correctly.
- `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`: dSprites (shapes with variations in shape, scale, orientation, position).
  - Source: https://github.com/deepmind/dsprites-dataset
- `3dshapes.h5`: Shapes3D (3D shapes with variations in hue, shape, scale, orientation).
  - Source: https://github.com/deepmind/3d-shapes
- `mpi3d_real.npz`: MPI3D (realistic 3D objects with variations in color, shape, size, camera position).
  - Source: https://github.com/rr-learning/disentanglement_dataset
Expected layout:
```text
data/
├─ dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
├─ 3dshapes.h5
└─ mpi3d_real.npz
```

## Results
- **Metrics**: Reconstruction loss (NLL for nonlinear, error for linear), KL divergence, L2 loss (for λβ-VAE), MIG, SAP, $I_m$.
- **Visualizations**:
  - Original and reconstructed image grids (nonlinear).
  - Latent traversal animations (GIFs, nonlinear).
  - Mutual information heatmaps.
  - Boxplots and heatmaps for metrics.
  - Interactive weighted score heatmaps (nonlinear).
- Summary statistics (mean ± std across seeds) in text files per dataset.

### Pre-trained Models and Results
Pre-trained models, metrics, and visualizations from training runs are available for download. These can be used to reproduce or analyze results without re-training.
- β-VAE (nonlinear): https://drive.google.com/file/d/1ofhdRkt4kaD7aLhTqq9FE2KIMQksmbyi/view?usp=drive_link
- λβ-VAE (nonlinear): https://drive.google.com/file/d/1AMGOE_MrNIQjRhlMHL73AQMQe4R7Qd9B/view?usp=drive_link

## Project Structure
- `config.py`: Configuration settings.
- `datasets.py`: Dataset loading.
- `models.py`: VAE model definitions.
- `losses.py`: Loss functions and training utilities.
- `metrics.py`: Disentanglement metric calculations.
- `utils.py`: Utility functions for seeding and data handling.
- `visualizations.py`: Functions for generating plots and images.
- `main_beta.py`: Script for nonlinear β-VAE training and evaluation.
- `main_lambda.py`: Script for nonlinear λβ-VAE continuation and evaluation.
- `linear_betavae_optimizer.py`: Script for linear β-VAE and λβ-VAE experiments.
- `linear_betavae.py`: Alternative script for linear experiments.
- `interactive_heatmap_visualizer.py`: Script for interactive weighted score heatmaps.
- `requirements.txt`: List of dependencies.

## Acknowledgments
- Datasets and metrics inspired by the Disentanglement Library: https://github.com/google-research/disentanglement_lib.
