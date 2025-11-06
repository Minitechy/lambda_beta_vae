# config.py
"""Configuration constants for the β-VAE and λβ-VAE experiments."""

import torch

CONFIG = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'latent_dim': 10,
    'total_steps': 150000,
    'additional_steps': 50000,
    'log_interval': 5000,
    'final_window': 5000,
    'num_train_samples': 10000,
    'num_test_samples': 5000,
    'num_bins': 20,
    'loss_type': 'bce',
    'continuous_factors': False,
    'betas': [1, 4, 8, 16, 32],
    'lambdas': [0, 2, 4, 6, 8],
    'num_seeds': 10,
    'learning_rate': 1e-4,
    'adam_betas': (0.9, 0.999),
    'adam_eps': 1e-08,
    'num_distinct_images': 5,
}