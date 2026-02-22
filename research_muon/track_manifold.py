import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import subprocess
import glob
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.llm_config import BlueberryConfig
from models.llm import MinimalLLM
from optimizers.muon import Muon

from utils.spectral import compute_spectral_stats, compute_singular_values

def run_tracking_and_plot(tokens="2000000"):
    print(f"Running training with --track_manifold true for {tokens} tokens...")
    cmd = [
        "python", "train_llm.py",
        "--train_tokens", tokens,
        "--track_manifold", "true"
    ]
    
    subprocess.run(cmd, check=True)
    
    print("Training complete. Finding the latest metrics file...")
    metric_files = glob.glob(f"plots/metrics_*{tokens}_*.json")
    if not metric_files:
        metric_files = glob.glob("plots/metrics_*.json")
    if not metric_files:
        print("ERROR: Could not find metrics file!")
        return
        
    metric_files.sort(key=os.path.getmtime)
    latest_file = metric_files[-1]
    
    with open(latest_file, "r") as f:
        data = json.load(f)
        
    manifold_history = data["history"].get("manifold_history", None)
    if manifold_history is None:
        print("ERROR: manifold_history not found in metrics!")
        return
        
    plot_results(manifold_history, "results/research_plots")

def plot_results(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle nested manifold_history if present
    if 'manifold_history' in history:
        # Merge manifold_history keys into a local view of history for plotting
        manifold_data = history['manifold_history']
        # We want to keep 'steps' from the manifold_history if it exists there
        plotting_history = {**manifold_data}
        if 'steps' not in plotting_history and 'steps' in history:
            plotting_history['steps'] = history['steps']
        history = plotting_history

    sns.set_theme(style="whitegrid")
    
    # 1. Training Loss
    plt.figure(figsize=(10, 6))
    if 'loss' in history and history['loss']:
        plt.plot(history['loss'], color='#2c3e50', linewidth=2)
        plt.title('Hierarchical Learning: Training Loss Convergence')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300)
    plt.close()

    # 2. Q vs V Spectral Max (First vs Last)
    plt.figure(figsize=(12, 7))
    colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
    # Dynamically find the last layer index
    k_layers = sorted([int(k.split('_')[-1]) for k in history.keys() if k.startswith('spec_norm_Q_') and k.split('_')[-1].isdigit()])
    last_idx = k_layers[-1] if k_layers else 0
    
    keys = ['spec_norm_Q_0', 'spec_norm_V_0', f'spec_norm_Q_{last_idx}', f'spec_norm_V_{last_idx}']
    labels = ['Q (First)', 'V (First)', f'Q (Layer {last_idx})', f'V (Layer {last_idx})']
    
    steps = history.get('steps', None)
    
    for k, l, c in zip(keys, labels, colors):
        if k in history and history[k]:
            data = history[k]
            # Only downsample if we have lots of data
            step_size = max(1, len(data) // 100)
            
            if steps is not None and len(steps) == len(data):
                plt.plot(steps[::step_size], data[::step_size], label=l, color=c, linewidth=2, marker='o' if len(data) < 20 else None)
            else:
                plt.plot(data[::step_size], label=l, color=c, linewidth=2, marker='o' if len(data) < 20 else None)
            
    plt.title('Hierarchical Stretching: Max Spectral Norm Evolution')
    plt.xlabel('Training Steps')
    plt.ylabel('Spectral / Operator Norm')
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'spectral_stretching_evolution.png'), dpi=300)
    plt.close()

    # 3. Spectral Gap (Q projections)
    plt.figure(figsize=(10, 6))
    steps = history.get('steps', None)
    
    if 'spec_gap_Q_0' in history and history['spec_gap_Q_0']:
        data0 = history['spec_gap_Q_0']
        step0 = max(1, len(data0) // 100)
        if steps is not None and len(steps) == len(data0):
            plt.plot(steps[::step0], data0[::step0], label='Layer 0 Gap', color='#16a085', marker='o' if len(data0) < 20 else None)
        else:
            plt.plot(data0[::step0], label='Layer 0 Gap', color='#16a085', marker='o' if len(data0) < 20 else None)

    if 'spec_gap_Q_last' in history and history['spec_gap_Q_last']:
        datalast = history['spec_gap_Q_last']
        steplast = max(1, len(datalast) // 100)
        if steps is not None and len(steps) == len(datalast):
            plt.plot(steps[::steplast], datalast[::steplast], label='Layer Last Gap', color='#d35400', marker='o' if len(datalast) < 20 else None)
        else:
            plt.plot(datalast[::steplast], label='Layer Last Gap', color='#d35400', marker='o' if len(datalast) < 20 else None)

    plt.title('Spectral Gap ($\sigma_1 / \sigma_2$): Signature of Singular Feature Focus')
    plt.xlabel('Training Steps')
    plt.ylabel('Gap Ratio')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'spectral_gap.png'), dpi=300)
    plt.close()

    # 4. Final Spectral Norm Distribution (Heatmap)
    plt.figure(figsize=(18, 6))
    q_norms, k_norms, v_norms, o_norms, up_norms, down_norms = [], [], [], [], [], []
    layer_names = []
    
    k_layers = sorted([int(k.split('_')[-1]) for k in history.keys() if k.startswith('spec_norm_Q_') and k.split('_')[-1].isdigit()])
    
    for i in k_layers:
        q_norms.append(history[f'spec_norm_Q_{i}'][-1])
        k_norms.append(history.get(f'spec_norm_K_{i}', [0])[-1])
        v_norms.append(history[f'spec_norm_V_{i}'][-1])
        o_norms.append(history.get(f'spec_norm_O_{i}', [0])[-1])
        up_norms.append(history.get(f'spec_norm_Up_{i}', [0])[-1])
        down_norms.append(history.get(f'spec_norm_Down_{i}', [0])[-1])
        layer_names.append(f"L{i}")
        
    if q_norms:
        heatmap_data = np.array([q_norms, v_norms])
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=layer_names, yticklabels=['Query', 'Value'],
                    annot_kws={"size": 8})
        plt.title('Final Q & V Spectral Norm Distribution Across All Layers')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'norm_heatmap.png'), dpi=300)
    plt.close()

    # 4b. Spectral Norm vs Depth (Scatter Plot)
    plt.figure(figsize=(12, 6))
    colors = {
        'Q': '#e74c3c',   # Red
        'V': '#2ecc71',   # Green
    }
    
    for proj_type, color in colors.items():
        norms = []
        depths = []
        for i in k_layers:
            key = f'spec_norm_{proj_type}_{i}'
            if key in history:
                norms.append(history[key][-1])
                depths.append(i)
        
        if norms:
            plt.scatter(depths, norms, color=color, label=f'{proj_type} Proj', s=60, alpha=0.7, edgecolors='white')
            # Add a trend line
            if len(depths) > 1:
                z = np.polyfit(depths, norms, 1)
                p = np.poly1d(z)
                plt.plot(depths, p(depths), color=color, linestyle='--', alpha=0.4)

    plt.xlabel('Layer Index (Depth)')
    plt.ylabel('Final Spectral Norm')
    plt.title('Spectral Norm vs Depth by Projection Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_norm_vs_depth.png'), dpi=300)
    plt.close()

    # 5. Top Singular Values (Final Step)
    plt.figure(figsize=(10, 6))
    if 'spec_vals_Q_0' in history and history['spec_vals_Q_0']:
        v0 = history['spec_vals_Q_0'][-1]
        vlast = history['spec_vals_Q_last'][-1]
        plt.bar(np.arange(len(v0))-0.2, v0, width=0.4, label='Layer 0 (Query)', alpha=0.8)
        plt.bar(np.arange(len(vlast))+0.2, vlast, width=0.4, label='Layer Last (Query)', alpha=0.8)
        plt.title('Top 10 Singular Values: Spectral Signatures')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'singular_spectrum.png'), dpi=300)
    plt.close()

    # 6. Update-Weight Alignment (Geometric Lock-in)
    plt.figure(figsize=(10, 6))
    k_layers = sorted([int(lk.split('_')[-1]) for lk in history.keys() if lk.startswith('alignment_Q_') and lk.split('_')[-1].isdigit()])
    last_idx = k_layers[-1] if k_layers else 0
    
    if f'alignment_Q_0' in history and history[f'alignment_Q_0']:
        data0 = history[f'alignment_Q_0']
        datalast = history[f'alignment_Q_{last_idx}']
        steps = history.get('steps', list(range(len(data0))))
        
        # Downsample
        step_sz = max(1, len(data0) // 100)
        plt.plot(steps[::step_sz], data0[::step_sz], label='Layer 0 Alignment', color='#8e44ad', linewidth=2)
        plt.plot(steps[::step_sz], datalast[::step_sz], label=f'Layer {last_idx} Alignment', color='#f39c12', linewidth=2)
        
        plt.title('Update-Weight Subspace Alignment (k=5)')
        plt.xlabel('Training Steps')
        plt.ylabel('Cosine Similarity (1.0 = Max Lock-in)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'update_alignment.png'), dpi=300)
    plt.close()

    # 7. Spectral Entropy Heatmap (Capacity Utilization)
    plt.figure(figsize=(16, 5))
    e_q = []
    e_v = []
    l_names = []
    for i in k_layers:
        if f'entropy_Q_{i}' in history:
            e_q.append(history[f'entropy_Q_{i}'][-1])
            e_v.append(history[f'entropy_V_{i}'][-1])
            l_names.append(f"L{i}")
            
    if e_q and e_v:
        heatmap_data = np.array([e_q, e_v])
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="magma", 
                    xticklabels=l_names, yticklabels=['Query Entropy', 'Value Entropy'])
        plt.title('Final Spectral Entropy: Capacity Utilization (0.0 = Low Rank, 1.0 = Full Rank)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'entropy_heatmap.png'), dpi=300)
    plt.close()

    print(f"Research plots successfully saved to {save_dir}/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=str, default="2000000")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--metrics_file", type=str, default=None)
    args = parser.parse_args()
    
    if args.plot_only:
        metrics_files = glob.glob("plots/metrics_*.json")
        if args.metrics_file:
            latest_file = args.metrics_file
        elif metrics_files:
            latest_file = max(metrics_files, key=os.path.getmtime)
        else:
            print("No metrics files found.")
            exit(1)
            
        print(f"Reading manifold history from {latest_file}...")
        with open(latest_file, "r") as f:
            data = json.load(f)
        manifold_history = data.get("history", {}).get("manifold_history", None)
        if manifold_history:
            plot_results(manifold_history, "results/research_plots")
        else:
            print("No manifold history found.")
    else:
        run_tracking_and_plot(args.tokens)
