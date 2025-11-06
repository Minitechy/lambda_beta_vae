# losses.py
"""Loss functions and training utilities for β-VAE and λβ-VAE."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG

def beta_vae_loss(recon_x, x, mu, logvar, beta, loss_type):
    """Compute β-VAE loss (reconstruction + beta * KL divergence)."""
    batch_size = x.size(0)
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + beta * kl_loss
    return recon_loss, kl_loss, total_loss

def lambda_beta_vae_loss(recon_x, x, mu, logvar, beta, lambda_, loss_type):
    """Compute λβ-VAE loss (reconstruction + beta * KL divergence + lambda * L2 loss)."""
    batch_size = x.size(0)
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    l2_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    total_loss = recon_loss + beta * kl_loss + lambda_ * l2_loss
    return recon_loss, kl_loss, l2_loss, total_loss

def train_beta_vae(model, train_loader, optimizer, beta):
    """Train β-VAE model."""
    model.to(CONFIG['device'])
    model.train()
    recon_window, kl_window, total_window = [], [], []
    train_iter = iter(train_loader)
    global_step = 0
    pbar = tqdm(total=CONFIG['total_steps'], desc="Training β-VAE", dynamic_ncols=True)
    while global_step < CONFIG['total_steps']:
        try:
            batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, _ = next(train_iter)
        batch = batch.to(CONFIG['device'], non_blocking=True)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        recon_loss, kl_loss, total_loss = beta_vae_loss(recon_batch, batch, mu, logvar, beta, model.loss_type)
        total_loss.backward()
        optimizer.step()
        recon_window.append(recon_loss.item())
        kl_window.append(kl_loss.item())
        total_window.append(total_loss.item())
        recon_window = recon_window[-CONFIG['final_window']:]
        kl_window = kl_window[-CONFIG['final_window']:]
        total_window = total_window[-CONFIG['final_window']:]
        global_step += 1
        if global_step % CONFIG['log_interval'] == 0 or global_step == CONFIG['total_steps']:
            pbar.set_description(
                f"Step [{global_step}/{CONFIG['total_steps']}] | "
                f"NLL: {np.mean(recon_window):.2f} | KL: {np.mean(kl_window):.2f} | "
                f"Total: {np.mean(total_window):.2f}"
            )
        pbar.update(1)
    pbar.close()
    return {
        "NLL": np.mean(recon_window),
        "KL": np.mean(kl_window),
        "Total loss": np.mean(total_window)
    }

def continue_train_lambda_beta_vae(model, train_loader, optimizer, beta, lambda_):
    """Continue training the model with λβ-VAE loss."""
    model.to(CONFIG['device'])
    model.train()
    recon_window, kl_window, l2_window, total_window = [], [], [], []
    train_iter = iter(train_loader)
    global_step = 0
    pbar = tqdm(total=CONFIG['additional_steps'], desc="Continuing Training λβ-VAE", dynamic_ncols=True)
    while global_step < CONFIG['additional_steps']:
        try:
            batch, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch, _ = next(train_iter)
        batch = batch.to(CONFIG['device'], non_blocking=True)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        recon_loss, kl_loss, l2_loss, total_loss = lambda_beta_vae_loss(
            recon_batch, batch, mu, logvar, beta, lambda_, model.loss_type
        )
        total_loss.backward()
        optimizer.step()
        recon_window.append(recon_loss.item())
        kl_window.append(kl_loss.item())
        l2_window.append(l2_loss.item())
        total_window.append(total_loss.item())
        recon_window = recon_window[-CONFIG['final_window']:]
        kl_window = kl_window[-CONFIG['final_window']:]
        l2_window = l2_window[-CONFIG['final_window']:]
        total_window = total_window[-CONFIG['final_window']:]
        global_step += 1
        if global_step % CONFIG['log_interval'] == 0 or global_step == CONFIG['additional_steps']:
            pbar.set_description(
                f"Step [{global_step}/{CONFIG['additional_steps']}] | "
                f"NLL: {np.mean(recon_window):.2f} | KL: {np.mean(kl_window):.2f} | "
                f"L2: {np.mean(l2_window):.2f} | Total: {np.mean(total_window):.2f}"
            )
        pbar.update(1)
    pbar.close()
    return {
        "NLL": np.mean(recon_window),
        "KL": np.mean(kl_window),
        "L2": np.mean(l2_window),
        "Total loss": np.mean(total_window)
    }