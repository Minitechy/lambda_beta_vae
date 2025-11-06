# main_beta.py
"""Main script for β-VAE training and evaluation."""

import os
import numpy as np
import torch
import torch.optim as optim

from config import CONFIG
from datasets import DSpritesDataset, Shapes3DDataset, MPI3DDataset
from models import VAE
from losses import train_beta_vae
from metrics import get_representations_and_factors, compute_metrics
from utils import seed_everything, make_train_loader, select_distinct_indices
from visualizations import (save_distinct_images, visualize_originals_grid, visualize_reconstructions_grid,
                            visualize_latent_traversal_combined, visualize_mi_heatmap, plot_metric_boxplots,
                            save_results_summary)

def main():
    """Main function to run β-VAE experiments on multiple datasets."""
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
        save_path = f"./betavae_results/{ds_name}/"
        os.makedirs(save_path, exist_ok=True)
        results = {beta: [] for beta in CONFIG['betas']}
        fixed_indices_cache = {}
        for seed in range(CONFIG['num_seeds']):
            print(f"\n-- Seed {seed} --")
            seed_everything(seed)
            rng = np.random.default_rng(seed)
            distinct_indices = select_distinct_indices(len(dataset), CONFIG['num_distinct_images'], rng)
            eval_indices = rng.choice(len(dataset), size=CONFIG['num_train_samples'] + CONFIG['num_test_samples'], replace=False)
            fixed_indices_cache[seed] = {'distinct': distinct_indices, 'eval': eval_indices}
            save_distinct_images(dataset, distinct_indices, save_path, dataset.channels, seed)
            train_loader = make_train_loader(dataset, CONFIG['batch_size'], seed)
            fixed_batch, _ = next(iter(train_loader))
            fixed_batch = fixed_batch.to(CONFIG['device'])
            visualize_originals_grid(fixed_batch, save_path, dataset.channels, seed)
            for beta in CONFIG['betas']:
                print(f"==== Processing beta = {beta} ====")
                seed_everything(seed)
                train_loader.generator.manual_seed(seed)
                model_save_path = os.path.join(save_path, f"betavae_beta_{beta}_seed_{seed}.pth")
                metrics_path = os.path.join(save_path, f"metrics_beta_{beta}_seed_{seed}.npz")
                if os.path.exists(model_save_path) and os.path.exists(metrics_path):
                    model = VAE(CONFIG['latent_dim'], dataset.channels, CONFIG['loss_type']).to(CONFIG['device'])
                    model.load_state_dict(torch.load(model_save_path))
                    loaded_data = np.load(metrics_path)
                    train_results = {
                        "NLL": float(loaded_data["NLL"]),
                        "KL": float(loaded_data["KL"]),
                        "Total loss": float(loaded_data["Total loss"])
                    }
                    metric_results = {
                        "MIG": float(loaded_data["MIG"]),
                        "SAP": float(loaded_data["SAP"]),
                        "Im": float(loaded_data["Im"])
                    }
                    mi_matrix = loaded_data["mi_matrix"]
                    print(f"Loaded pre-trained model and metrics for beta={beta}, seed={seed}")
                else:
                    model = VAE(CONFIG['latent_dim'], dataset.channels, CONFIG['loss_type']).to(CONFIG['device'])
                    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], betas=CONFIG['adam_betas'], eps=CONFIG['adam_eps'])
                    train_results = train_beta_vae(model, train_loader, optimizer, beta)
                    torch.save(model.state_dict(), model_save_path)
                    mus, factors = get_representations_and_factors(model, dataset, fixed_indices_cache[seed]['eval'])
                    mus_train, ys_train = mus[:, :CONFIG['num_train_samples']], factors[:, :CONFIG['num_train_samples']]
                    mus_test, ys_test = mus[:, CONFIG['num_train_samples']:], factors[:, CONFIG['num_train_samples']:]
                    metric_results, mi_matrix = compute_metrics(mus_train, ys_train, mus_test, ys_test,
                                                                CONFIG['num_bins'], CONFIG['continuous_factors'], seed)
                    saved_metrics = {
                        "NLL": train_results["NLL"],
                        "KL": train_results["KL"],
                        "Total loss": train_results["Total loss"],
                        "MIG": metric_results["MIG"],
                        "SAP": metric_results["SAP"],
                        "Im": metric_results["Im"]
                    }
                    np.savez(metrics_path, **saved_metrics, mi_matrix=mi_matrix)
                visualize_reconstructions_grid(model, fixed_batch, beta, save_path, dataset.channels, seed)
                traversal_path = os.path.join(save_path, f"latent_traversal_beta_{beta}_seed_{seed}")
                visualize_latent_traversal_combined(model, traversal_path,
                                                    fixed_indices_cache[seed]['distinct'], dataset, dataset.channels, CONFIG['latent_dim'])
                visualize_mi_heatmap(mi_matrix, factor_names_dict[ds_name], save_path, beta, seed)
                combined_results = {**train_results, **metric_results}
                results[beta].append(combined_results)
        results_dict[ds_name] = results
        print(f"\n==== Summary Results for {ds_name} (Mean ± Std) ====")
        for beta in CONFIG['betas']:
            beta_results = results[beta]
            means = {key: np.mean([r[key] for r in beta_results]) for key in beta_results[0]}
            stds = {key: np.std([r[key] for r in beta_results]) for key in beta_results[0]}
            print(f"beta = {beta}:")
            for key in ["NLL", "KL", "Total loss", "MIG", "SAP", "Im"]:
                print(f" {key}: {means[key]:.4f} ± {stds[key]:.4f}")
        save_results_summary(results, ds_name, save_path)
    plot_metric_boxplots(results_dict, list(datasets.keys()), "./betavae_results/")

if __name__ == "__main__":
    main()