# datasets.py
"""Dataset classes for dSprites, Shapes3D, and MPI3D."""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class DSpritesDataset(Dataset):
    """Dataset class for dSprites dataset."""
    def __init__(self, file_path="data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"dSprites dataset not found at {file_path}")
        self.data = np.load(file_path, mmap_mode='r')
        self.images = self.data['imgs'].astype(np.float32)
        self.factors = self.data['latents_classes'][:, 1:].astype(np.float32)
        self.channels = 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)
        factors = torch.from_numpy(self.factors[idx])
        return image, factors

class Shapes3DDataset(Dataset):
    """Dataset class for Shapes3D dataset."""
    def __init__(self, file_path="data/3dshapes.h5"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Shapes3D dataset not found at {file_path}")
        with h5py.File(file_path, 'r') as f:
            self.images = f['images'][:].astype(np.float32) / 255.0
            self.factors = f['labels'][:].astype(np.float32)
        self.channels = 3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        factors = torch.from_numpy(self.factors[idx])
        return image, factors

class MPI3DDataset(Dataset):
    """Dataset class for MPI3D dataset."""
    def __init__(self, file_path="data/mpi3d_real.npz"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MPI3D dataset not found at {file_path}")
        data = np.load(file_path, mmap_mode='r')
        factor_sizes = [6, 6, 2, 3, 3, 40, 40]
        total_samples = np.prod(factor_sizes)
        self.images = data['images']
        self.total_samples = total_samples
        factor_grids = np.meshgrid(*[range(size) for size in factor_sizes], indexing='ij')
        self.factors = np.stack([grid.ravel() for grid in factor_grids], axis=1).astype(np.float32)
        self.channels = 3

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        image = self.images[idx].reshape(64, 64, 3).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        factors = torch.from_numpy(self.factors[idx])
        return image, factors