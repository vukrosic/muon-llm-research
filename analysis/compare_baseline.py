import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def compute_entropy(singular_values):
    s = np.array(singular_values)
    if len(s) == 0: return 0
    s_norm = s / (np.sum(s) + 1e-12)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    return entropy

def get_experiment_data(exp_dir):
    stats_file = Path(exp_dir) / "metrics" / "manifold_stats.jsonl"
    if not stats_file.exists():
        # Fallback to any other possible location
        stats_file = Path(exp_dir) / "raw_metrics" / "manifold_stats.jsonl"
        if not stats_file.exists():
            stats_file = Path(exp_dir) / "manifold_stats.jsonl"
            
    records = load_jsonl(stats_file)
    if not records:
        print(f"Warning: No stats found for {exp_dir}")
        return None
    
    # Process entropy if requested metric is entropy
    for r in records:
        if 'singular_values' in r:
            r['entropy'] = compute_entropy(r['singular_values'])
            # Don't keep the full singular values in memory for all records if we don't need them
            # del r['singular_values'] 
            
    return records

def plot_heatmaps(all_data, metric='entropy'):
    # all_data: {'muon': {42: {...}}, 'adamw': {42: {...}}}
    
    # We will plot one figure per optimizer.
    # The figure will have rows=projections, cols=seeds
    projections = ['q', 'k', 'v', 'o', 'up', 'down']
    opts = list(all_data.keys())
    
    for opt in opts:
        seeds = list(all_data[opt].keys())
        if not seeds: continue
        
        fig, axes = plt.subplots(len(projections), len(seeds), figsize=(5 * len(seeds), 4 * len(projections)))
        
        for r_idx, proj in enumerate(projections):
            for c_idx, seed in enumerate(seeds):
                ax = axes[r_idx, c_idx] if len(projections) > 1 else (axes[c_idx] if len(seeds) > 1 else axes)
                
                # Build heatmap matrix: (layers, steps)
                records_dict = all_data[opt][seed]
                if not records_dict: continue
                
                # Extract unique steps and layers
                steps = sorted(list(set([k[0] for k in records_dict.keys()])))
                layers = sorted(list(set([k[1] for k in records_dict.keys()])))
                
                if not steps or not layers: continue
                
                heatmap = np.zeros((len(layers), len(steps)))
                for sid, s in enumerate(steps):
                    for lid, l in enumerate(layers):
                        key = (s, l, proj)
                        if key in records_dict:
                            heatmap[lid, sid] = records_dict[key].get(metric, 0)
                            
                im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis', 
                          extent=[steps[0], steps[-1], layers[0], layers[-1]])
                fig.colorbar(im, ax=ax)
                
                if r_idx == 0:
                    ax.set_title(f"{opt.upper()} (Seed {seed})", fontsize=14)
                if c_idx == 0:
                    ax.set_ylabel(f"Proj {proj.upper()}\nLayer Depth", fontsize=12)
                if r_idx == len(projections) - 1:
                    ax.set_xlabel("Step")
                    
        plt.tight_layout()
        os.makedirs("results/analysis", exist_ok=True)
        plt.savefig(f"results/analysis/heatmap_{opt}_{metric}.png")
        plt.close(fig)
        print(f"Saved {opt} heatmap to results/analysis/heatmap_{opt}_{metric}.png")

def main():
    base_path = Path("experiments/01_muon_vs_adamw_baseline")
    opts = ['muon', 'adamw']
    seeds = [42, 137, 256]
    
    all_data = {'muon': {}, 'adamw': {}}
    
    for opt in opts:
        for seed in seeds:
            exp_dir = base_path / f"{opt}_seed{seed}"
            print(f"Loading {exp_dir}...")
            data = get_experiment_data(exp_dir)
            if data:
                all_data[opt][seed] = {(r['step'], r['layer'], r['proj']): r for r in data}
                
    if not any(all_data['muon'].values()) and not any(all_data['adamw'].values()):
        print("No data found to plot. Ensure experiments completed.")
        return

    metrics = ['entropy', 'update_alignment', 'left_alignment', 'right_alignment', 'grad_norm', 'weight_norm', 'effective_rank']
    for metric in metrics:
        print(f"Plotting heatmaps for {metric}...")
        try:
            plot_heatmaps(all_data, metric=metric)
        except Exception as e:
            print(f"Failed to plot {metric}: {e}")

if __name__ == "__main__":
    main()
