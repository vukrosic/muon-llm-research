import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_jsonl(path):
    data = []
    if not os.path.exists(path): return data
    with open(path, 'r') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue
    return data

def compute_entropy(singular_values):
    s = np.array(singular_values)
    if len(s) == 0: return 0
    s_norm = s / (np.sum(s) + 1e-12)
    return -np.sum(s_norm * np.log(s_norm + 1e-12))

def plot_heatmaps(records, exp_name, output_dir, metric='entropy', seed='137'):
    projections = ['q', 'k', 'v', 'o', 'up', 'down']
    
    if metric == 'entropy':
        for r in records:
            if 'singular_values' in r:
                r['entropy'] = compute_entropy(r['singular_values'])

    data_dict = {(r['step'], r['layer'], r['proj']): r for r in records}
    steps = sorted(list(set([k[0] for k in data_dict.keys()])))
    layers = sorted(list(set([k[1] for k in data_dict.keys()])))

    fig, axes = plt.subplots(len(projections), 1, figsize=(10, 4 * len(projections)))
    
    for idx, proj in enumerate(projections):
        ax = axes[idx]
        heatmap = np.zeros((len(layers), len(steps)))
        for sid, s in enumerate(steps):
            for lid, l in enumerate(layers):
                key = (s, l, proj)
                if key in data_dict:
                    heatmap[lid, sid] = data_dict[key].get(metric, 0)
        
        im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis',
                      extent=[steps[0], steps[-1], layers[0], layers[-1]])
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{exp_name} - {proj.upper()} - {metric}")
        ax.set_ylabel("Layer Depth")
        if idx == len(projections) - 1: ax.set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{metric}_seed{seed}.png")
    plt.close()

def plot_single_experiment(metrics_path, output_dir, seed='137'):
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    steps = history['steps']
    losses = history['val_losses']
    accs = history['val_accuracies']
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', color='tab:blue', label='Val Loss')
    plt.title(f"Validation Loss - Muon Seed {seed}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"loss_seed{seed}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(steps, accs, marker='o', color='tab:green', label='Val Accuracy')
    plt.title(f"Validation Accuracy - Muon Seed {seed}")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"accuracy_seed{seed}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metrics/manifold heatmaps for a Muon seed run.")
    parser.add_argument("--seed", default="137", help="Seed id to plot, e.g. 42 or 137")
    args = parser.parse_args()
    seed = str(args.seed)

    exp_dir = Path(f"experiments/01_muon_vs_adamw_baseline/muon_seed{seed}")
    out_dir = Path(f"experiments/01_muon_vs_adamw_baseline/plots_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard plots
    metrics_file = exp_dir / "metrics.json"
    if metrics_file.exists():
        plot_single_experiment(metrics_file, out_dir, seed=seed)
    
    # Heatmaps
    stats_file = exp_dir / "metrics" / "manifold_stats.jsonl"
    if stats_file.exists():
        records = load_jsonl(stats_file)
        for m in ['entropy', 'update_alignment', 'effective_rank']:
            print(f"Generating heatmap for {m}...")
            plot_heatmaps(records, f"Muon Seed {seed}", out_dir, metric=m, seed=seed)
    
    print(f"Done. Plots saved to {out_dir}")
