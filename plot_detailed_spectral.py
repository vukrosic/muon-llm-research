import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_metrics(filepath):
    print(f"Loading metrics from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_detailed_spectral(files, labels, output_dir="plots/spectral"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = [load_metrics(f) for f in files]
    
    # 1. Plot Effective Rank over time
    plt.figure(figsize=(15, 10))
    for p_idx, proj in enumerate(['Q', 'K', 'V', 'O']):
        plt.subplot(2, 2, p_idx + 1)
        for d_idx, data in enumerate(all_data):
            label_prefix = labels[d_idx]
            layer_logs = data.get('history', {}).get('layer_logs', [])
            if not layer_logs: continue
            
            steps = [log['step'] for log in layer_logs]
            num_layers = len(layer_logs[0]['layers'])
            
            # For multiple data sources, only plot a subset of layers or average to avoid clutter
            # Let's plot average rank across layers for comparison
            avg_ranks = []
            for log in layer_logs:
                ranks = [layer['projections'][proj]['effective_rank'] for layer in log['layers']]
                avg_ranks.append(np.mean(ranks))
            
            style = '-' if d_idx == 0 else '--'
            plt.plot(steps, avg_ranks, marker='o', linestyle=style, markersize=5, label=f'{label_prefix} (Avg)')
            
        comparison_title = " vs ".join(labels)
        plt.title(f'{comparison_title}\nEffective Rank (Avg) - {proj}')
        plt.xlabel('Step')
        plt.ylabel('Rank')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "effective_rank_comparison.png")
    plt.close()
    
    # 2. Plot Alignment over time
    plt.figure(figsize=(15, 10))
    for p_idx, proj in enumerate(['Q', 'K', 'V', 'O']):
        plt.subplot(2, 2, p_idx + 1)
        for d_idx, data in enumerate(all_data):
            label_prefix = labels[d_idx]
            layer_logs = data.get('history', {}).get('layer_logs', [])
            if not layer_logs: continue
            
            steps = [log['step'] for log in layer_logs]
            avg_aligns = []
            for log in layer_logs:
                aligns = [layer['projections'][proj]['alignment'] for layer in log['layers']]
                avg_aligns.append(np.mean(aligns))
            
            style = '-' if d_idx == 0 else '--'
            plt.plot(steps, avg_aligns, marker='o', linestyle=style, markersize=5, label=f'{label_prefix} (Avg)')
            
        comparison_title = " vs ".join(labels)
        plt.title(f'{comparison_title}\nUpdate-Weight Alignment (Avg) - {proj}')
        plt.xlabel('Step')
        plt.ylabel('Cosine Sim')
        plt.ylim(-0.1, 1.1)
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "alignment_comparison.png")
    plt.close()
    
    # 3. Plot SVD Spectrum (Last point, middle layer)
    plt.figure(figsize=(15, 10))
    for p_idx, proj in enumerate(['Q', 'K', 'V', 'O']):
        plt.subplot(2, 2, p_idx + 1)
        for d_idx, data in enumerate(all_data):
            label_prefix = labels[d_idx]
            layer_logs = data.get('history', {}).get('layer_logs', [])
            if not layer_logs: continue
            
            last_log = layer_logs[-1]
            num_layers = len(last_log['layers'])
            mid_layer = num_layers // 2
            
            s_vals = last_log['layers'][mid_layer]['projections'][proj]['singular_values']
            plt.plot(s_vals, label=f'{label_prefix} L{mid_layer}')
            
        comparison_title = " vs ".join(labels)
        plt.title(f'{comparison_title}\nSVD Spectrum (Mid Layer) - {proj}')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.yscale('log')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "svd_spectrum_comparison.png")
    plt.close()

    # 4. Plot Grad Norm over time (Avg)
    plt.figure(figsize=(15, 10))
    for p_idx, proj in enumerate(['Q', 'K', 'V', 'O']):
        plt.subplot(2, 2, p_idx + 1)
        for d_idx, data in enumerate(all_data):
            label_prefix = labels[d_idx]
            layer_logs = data.get('history', {}).get('layer_logs', [])
            if not layer_logs: continue
            
            steps = [log['step'] for log in layer_logs]
            avg_norms = []
            for log in layer_logs:
                norms = [layer['projections'][proj]['grad_norm'] for layer in log['layers']]
                avg_norms.append(np.mean(norms))
            
            style = '-' if d_idx == 0 else '--'
            plt.plot(steps, avg_norms, marker='o', linestyle=style, markersize=5, label=f'{label_prefix} (Avg)')
            
        comparison_title = " vs ".join(labels)
        plt.title(f'{comparison_title}\nGrad Norm (Avg) - {proj}')
        plt.xlabel('Step')
        plt.ylabel('Norm')
        plt.yscale('log')
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "grad_norm_comparison.png")
    plt.close()

    print(f"Comparison plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="Path to one or more metrics.json files")
    parser.add_argument("--labels", nargs='+', help="Labels for the files")
    parser.add_argument("--output_dir", type=str, default="plots/comparison", help="Output directory")
    args = parser.parse_args()
    
    labels = args.labels if args.labels else [Path(f).stem for f in args.files]
    plot_detailed_spectral(args.files, labels, args.output_dir)
