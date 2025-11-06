# utils.py
"""Utility functions for seeding, data loading, and index selection."""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CONFIG

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_train_loader(dataset, batch_size, seed):
    """Create a DataLoader with reproducible shuffling."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

def select_distinct_indices(dataset_len, num, rng):
    """Select distinct indices for visualization."""
    if num <= 0:
        raise ValueError("Number of distinct images must be positive")
    edges = np.linspace(0, dataset_len, num + 1, dtype=int)
    return [min(s + rng.integers(0, max(1, e - s)), dataset_len - 1) for s, e in zip(edges[:-1], edges[1:])]