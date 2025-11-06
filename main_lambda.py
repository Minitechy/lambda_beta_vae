# main_lambda.py
"""Main script for continuing β-VAE training with λ term and evaluating disentanglement."""

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from config import CONFIG
from datasets import DSpritesDataset, Shapes3DDataset, MPI3DDataset
from models import VAE
from losses import continue_train_lambda_beta_vae
from metrics import get_representations_and_factors, compute_metrics
from utils import seed_everything, make_train_loader, select_distinct_indices
from visualizations import (save_distinct_images, visualize_originals_grid, visualize_reconstructions_grid,
                            visualize_latent_traversal_combined, visualize_mi_heatmap, plot_metric_heatmaps,
                            save_results_summary)

def main():
    """Main function to load β-VAE baselines and continue training with λ term."""
    datasets = {
        "dSprites": DSpritesDataset(),
        "Shapes3D": Shapes3DDataset(),
        "MPI3D": MPI3DDataset()
    }
    factor_names_dict = {
        "dSprites": ['Shape', 'Scale', 'Orient', 'Pos X', 'Pos Y'],
        "Shapes3D": ['Floor Hue', 'Wall Hue', 'Obj Hue', 'Scale', 'Shape', 'Orient'],
        "MPI3D": ['Obj Color', 'Obj Shape', 'Obj Size', 'Cam Height', 'Bg Color', 'Azimuth', 'Elevation']
    }
    results_dict = {}
    for ds_name, dataset in datasets.items():
        print(f"\n==== Dataset: {ds_name} ====")
        beta_save_path = f"./betavae_results/{ds_name}/"
        lambda_save_path = f"./lambda_betavae_results/{ds_name}/"
        os.makedirs(lambda_save_path, exist_ok=True)
        results = {beta: {lambda_: [] for lambda_ in CONFIG['lambdas']} for beta in CONFIG['betas']}
        fixed_indices_cache = {}
        for seed in range(CONFIG['num_seeds']):
            print(f"\n-- Seed {seed} --")
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            distinct_indices = select_distinct_indices(len(dataset), CONFIG['num_distinct_images'], rng)
            eval_indices = rng.choice(len(dataset), size=CONFIG['num_train_samples'] + CONFIG['num_test_samples'], replace=False)
            fixed_indices_cache[seed] = {'distinct': distinct_indices, 'eval': eval_indices}
            save_distinct_images(dataset, distinct_indices, lambda_save_path, dataset.channels, seed)
            train_loader = make_train_loader(dataset, CONFIG['batch_size'], seed)
            fixed_batch, _ = next(iter(train_loader))
            fixed_batch = fixed_batch.to(CONFIG['device'])
            visualize_originals_grid(fixed_batch, lambda_save_path, dataset.channels, seed)
            for beta in CONFIG['betas']:
                print(f"==== Processing beta = {beta} ====")
                beta_model_path = os.path.join(beta_save_path, f"betavae_beta_{beta}_seed_{seed}.pth")
                if not os.path.exists(beta_model_path):
                    print(f"Pre-trained β-VAE model not found at {beta_model_path}. Skipping.")
                    continue
                for lambda_ in CONFIG['lambdas']:
                    print(f"------ Continuing with lambda = {lambda_} ------")
                    seed_everything(seed)
                    train_loader.generator.manual_seed(seed)
                    model = VAE(CONFIG['latent_dim'], dataset.channels, CONFIG['loss_type']).to(CONFIG['device'])
                    model.load_state_dict(torch.load(beta_model_path))
                    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], betas=CONFIG['adam_betas'], eps=CONFIG['adam_eps'])
                    train_results = continue_train_lambda_beta_vae(model, train_loader, optimizer, beta, lambda_)
                    lambda_model_path = os.path.join(lambda_save_path, f"lambda_betavae_beta_{beta}_lambda_{lambda_}_seed_{seed}.pth")
                    if not os.path.exists(lambda_model_path):
                        torch.save(model.state_dict(), lambda_model_path)
                    mus, factors = get_representations_and_factors(model, dataset, fixed_indices_cache[seed]['eval'])
                    mus_train, ys_train = mus[:, :CONFIG['num_train_samples']], factors[:, :CONFIG['num_train_samples']]
                    mus_test, ys_test = mus[:, CONFIG['num_train_samples']:], factors[:, CONFIG['num_train_samples']:]
                    metric_results, mi_matrix = compute_metrics(mus_train, ys_train, mus_test, ys_test,
                                                                CONFIG['num_bins'], CONFIG['continuous_factors'], seed)
                    visualize_reconstructions_grid(model, fixed_batch, beta, lambda_save_path, dataset.channels, seed, lambda_=lambda_)
                    traversal_path = os.path.join(lambda_save_path, f"latent_traversal_beta_{beta}_lambda_{lambda_}_seed_{seed}")
                    visualize_latent_traversal_combined(model, traversal_path,
                                                        fixed_indices_cache[seed]['distinct'], dataset, dataset.channels, CONFIG['latent_dim'])
                    visualize_mi_heatmap(mi_matrix, factor_names_dict[ds_name], lambda_save_path, beta, seed, lambda_=lambda_)
                    metrics_path = os.path.join(lambda_save_path, f"metrics_beta_{beta}_lambda_{lambda_}_seed_{seed}.npz")
                    saved_metrics = {
                        "NLL": train_results["NLL"],
                        "KL": train_results["KL"],
                        "L2": train_results["L2"],
                        "Total loss": train_results["Total loss"],
                        "MIG": metric_results["MIG"],
                        "SAP": metric_results["SAP"],
                        "Im": metric_results["Im"]
                    }
                    if not os.path.exists(metrics_path):
                        np.savez(metrics_path, **saved_metrics, mi_matrix=mi_matrix)
                    combined_results = {**train_results, **metric_results}
                    results[beta][lambda_].append(combined_results)
        results_dict[ds_name] = results
        print(f"\n==== Summary Results for {ds_name} (Mean ± Std) ====")
        for beta in CONFIG['betas']:
            for lambda_ in CONFIG['lambdas']:
                lambda_results = results[beta][lambda_]
                means = {key: np.mean([r[key] for r in lambda_results]) for key in lambda_results[0]}
                stds = {key: np.std([r[key] for r in lambda_results]) for key in lambda_results[0]}
                print(f"beta = {beta}, lambda = {lambda_}:")
                for key in ["NLL", "KL", "L2", "Total loss", "MIG", "SAP", "Im"]:
                    print(f" {key}: {means[key]:.4f} ± {stds[key]:.4f}")
        save_results_summary(results, ds_name, lambda_save_path, is_lambda=True)
    plot_metric_heatmaps(results_dict, list(datasets.keys()), "./lambda_betavae_results/")

if __name__ == "__main__":
    main()