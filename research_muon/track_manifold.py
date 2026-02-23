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

def run_tracking_and_plot(tokens="2000000", resume=False):
    print(f"Running training with --track_manifold true for {tokens} tokens (Resume: {resume})...")
    cmd = [
        "python", "train_llm.py",
        "--train_tokens", tokens,
        "--track_manifold", "true",
        "--save_every", "1000000" # Save every 1M tokens
    ]
    if resume:
        cmd.append("--resume")
    
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
        manifold_data = history['manifold_history']
        plotting_history = {**manifold_data}
        if 'steps' not in plotting_history and 'steps' in history:
            plotting_history['steps'] = history['steps']
        history = plotting_history

    sns.set_theme(style="whitegrid")
    steps = history.get('steps', None)
    
    # 1. Training Loss
    plt.figure(figsize=(10, 6))
    if 'loss' in history and history['loss']:
        plt.plot(history['loss'], color='#2c3e50', linewidth=2)
        plt.title('Training Loss Convergence', fontsize=15)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300)
    plt.close()

    # 2. Max Spectral Norm Evolution (Stretching)
    plt.figure(figsize=(12, 7))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    k_layers = sorted([int(k.split('_')[-1]) for k in history.keys() if k.startswith('spec_norm_Q_') and k.split('_')[-1].isdigit()])
    last_idx = k_layers[-1] if k_layers else 0
    
    keys = ['spec_norm_Q_0', f'spec_norm_Q_{last_idx}', 'spec_norm_V_0', f'spec_norm_V_{last_idx}']
    labels = ['Q (L0)', f'Q (L{last_idx})', 'V (L0)', f'V (L{last_idx})']
    
    for k, l, c in zip(keys, labels, colors):
        if k in history and history[k]:
            data = history[k]
            step_size = max(1, len(data) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(data) else range(0, len(data)*step_size, step_size)
            plt.plot(x_axis, data[::step_size], label=l, color=c, linewidth=2.5)
            
    plt.title('Hierarchical Stretching: Max Spectral Norm Evolution', fontsize=16)
    plt.xlabel('Steps')
    plt.ylabel('$\sigma_max$ (Operator Norm)')
    plt.legend(frameon=True, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'spectral_stretching_evolution.png'), dpi=300)
    plt.close()

    # 3. Update-Weight Alignment (Geometric Lock-in)
    plt.figure(figsize=(12, 7))
    # Plot top and bottom query/value alignment
    alignment_keys = [f'alignment_Q_0', f'alignment_Q_{last_idx}', f'alignment_V_0', f'alignment_V_{last_idx}']
    alignment_labels = ['Q-Align (L0)', f'Q-Align (L{last_idx})', 'V-Align (L0)', f'V-Align (L{last_idx})']
    alignment_colors = ['#8e44ad', '#9b59b6', '#d35400', '#e67e22']

    for k, l, c in zip(alignment_keys, alignment_labels, alignment_colors):
        if k in history and history[k]:
            data = history[k]
            step_size = max(1, len(data) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(data) else range(0, len(data)*step_size, step_size)
            plt.plot(x_axis, data[::step_size], label=l, color=c, linewidth=2)

    plt.title('Geometric Lock-in: Update-Weight Subspace Alignment (k=5)', fontsize=16)
    plt.xlabel('Steps')
    plt.ylabel('Average Cosine Similarity')
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'update_alignment.png'), dpi=300)
    plt.close()

    # 4. Orthogonality Error (Manifold Departure)
    plt.figure(figsize=(12, 7))
    ortho_keys = [f'ortho_err_Q_0', f'ortho_err_Q_{last_idx}', f'ortho_err_V_0', f'ortho_err_V_{last_idx}']
    ortho_labels = ['Q-Ortho (L0)', f'Q-Ortho (L{last_idx})', 'V-Ortho (L0)', f'V-Ortho (L{last_idx})']
    ortho_colors = ['#16a085', '#1abc9c', '#c0392b', '#e74c3c']

    for k, l, c in zip(ortho_keys, ortho_labels, ortho_colors):
        if k in history and history[k]:
            data = history[k]
            step_size = max(1, len(data) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(data) else range(0, len(data)*step_size, step_size)
            plt.plot(x_axis, data[::step_size], label=l, color=c, linewidth=2)

    plt.title('Manifold Departure: Orthogonality Error $||(W/|W|)^T(W/|W|) - I||_F/\sqrt{d}$', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Departure from Orthogonality')
    plt.legend(frameon=True)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'orthogonality_error.png'), dpi=300)
    plt.close()

    # 5. Spectral Entropy MRI (Capacity Utilization)
    plt.figure(figsize=(14, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
    
    # Data extraction
    e_q, e_v, depths = [], [], []
    for i in k_layers:
        if f'entropy_Q_{i}' in history:
            e_q.append(history[f'entropy_Q_{i}'][-1])
            e_v.append(history[f'entropy_V_{i}'][-1])
            depths.append(i)
            
    if e_q and e_v:
        # Top Panel: Line Plot (The "Line on Graph" request)
        ax0 = plt.subplot(gs[0])
        ax0.plot(depths, e_q, marker='o', color='#e74c3c', label='Query Entropy', linewidth=2.5, markersize=8)
        ax0.plot(depths, e_v, marker='s', color='#2ecc71', label='Value Entropy', linewidth=2.5, markersize=8)
        ax0.set_ylim(0.4, 1.05)
        ax0.set_title('Spectral Entropy vs Depth: The Capacity MRI', fontsize=16, fontweight='bold')
        ax0.set_ylabel('Entropy Score')
        ax0.legend(loc='lower left', frameon=True)
        ax0.grid(True, alpha=0.3)
        
        # Bottom Panel: Heatmap (The "MRI Scan")
        ax1 = plt.subplot(gs[1])
        heatmap_data = np.array([e_q, e_v])
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="magma", 
                    xticklabels=[f"L{i}" for i in depths], yticklabels=['Query', 'Value'],
                    ax=ax1, cbar_kws={'label': 'Utilization (1.0 = Max)'},
                    annot_kws={"size": 10, "weight": "bold"})
        ax1.set_xlabel('Layer Index (Depth)')
        ax1.set_title('Final Layer Utilization Heatmap', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'entropy_mri.png'), dpi=300)
    plt.close()

    # 6. Spectral Norm vs Depth (Scatter Plot)
    plt.figure(figsize=(12, 6))
    proj_types = {'Q': '#e74c3c', 'V': '#2ecc71', 'Up': '#3498db', 'Down': '#f1c40f'}
    
    for proj, color in proj_types.items():
        norms, depths = [], []
        for i in k_layers:
            key = f'spec_norm_{proj}_{i}'
            if key in history:
                norms.append(history[key][-1])
                depths.append(i)
        
        if norms:
            plt.scatter(depths, norms, color=color, label=f'{proj} Proj', s=70, alpha=0.7, edgecolors='white')
            if len(depths) > 1:
                z = np.polyfit(depths, norms, 1)
                p = np.poly1d(z)
                plt.plot(depths, p(depths), color=color, linestyle='--', alpha=0.4)

    plt.xlabel('Layer Index (Depth)')
    plt.ylabel('Final Spectral Norm')
    plt.title('Spectral Norm Scaling Across Model Depth', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_norm_vs_depth.png'), dpi=300)
    plt.close()

    # 7. Singular Spectrum Decay
    plt.figure(figsize=(12, 7))
    if f'spec_vals_Q_0' in history and history[f'spec_vals_Q_0']:
        v0 = history[f'spec_vals_Q_0'][-1]
        vlast = history[f'spec_vals_Q_last'][-1]
        # Normalize to see decay shape
        v0_norm = np.array(v0) / v0[0]
        vlast_norm = np.array(vlast) / vlast[0]
        
        indices = np.arange(len(v0))
        plt.bar(indices-0.2, v0_norm, width=0.4, label='Layer 0 (Query)', color='#3498db', alpha=0.8)
        plt.bar(indices+0.2, vlast_norm, width=0.4, label=f'Layer {last_idx} (Query)', color='#e67e22', alpha=0.8)
        
        plt.title('Normalized Singular Value Decay: Spectral Signature', fontsize=15)
        plt.xlabel('Singular Value Index')
        plt.ylabel('Value (Normalized to $\sigma_1=1$)')
        plt.xticks(indices)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'singular_spectrum.png'), dpi=300)
    plt.close()

    # 8. Spectral Gap Evolution
    plt.figure(figsize=(10, 6))
    gap_keys = ['spec_gap_Q_0', 'spec_gap_Q_last']
    gap_labels = ['L0 Gap', f'L{last_idx} Gap']
    
    for k, l in zip(gap_keys, gap_labels):
        if k in history and history[k]:
            data = history[k]
            step_size = max(1, len(data) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(data) else range(0, len(data)*step_size, step_size)
            plt.plot(x_axis, data[::step_size], label=l, linewidth=2)

    plt.title('Singular Feature Focus: Spectral Gap ($\sigma_1/\sigma_2$)', fontsize=15)
    plt.xlabel('Steps')
    plt.ylabel('Gap Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'spectral_gap.png'), dpi=300)
    plt.close()
    # 10. Subspace Diagnostic: Efficiency Ratio (Alignment / Rank)
    plt.figure(figsize=(12, 7))
    ratio_layers = [0, last_idx]
    colors = ['#8e44ad', '#2980b9']
    
    for i, c in zip(ratio_layers, colors):
        r_key = f'update_rank_Q_{i}'
        a_key = f'alignment_Q_{i}'
        if r_key in history and a_key in history:
            ranks = np.array(history[r_key])
            aligns = np.array(history[a_key])
            ratios = aligns / (ranks + 1e-6)
            step_size = max(1, len(ratios) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(ratios) else range(0, len(ratios)*step_size, step_size)
            plt.plot(x_axis, ratios[::step_size], label=f'L{i} Q-Efficiency', color=c, linewidth=2.5)

    plt.title('Subspace Diagnostic: Optimizer "Efficiency" ($Alignment / Rank$)', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Efficiency (High = Sharp Needle)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'subspace_efficiency.png'), dpi=300)
    plt.close()

    # 11. Update Rank Evolution (Needle vs Wave)
    plt.figure(figsize=(12, 7))
    v_colors = ['#27ae60', '#16a085']
    for i, (qc, vc) in enumerate(zip(colors, v_colors)):
        layer_idx = 0 if i == 0 else last_idx
        qr_key = f'update_rank_Q_{layer_idx}'
        vr_key = f'update_rank_V_{layer_idx}'
        
        if qr_key in history:
            data = np.array(history[qr_key])
            step_size = max(1, len(data) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(data) else range(0, len(data)*step_size, step_size)
            plt.plot(x_axis, data[::step_size], label=f'L{layer_idx} Q-Update Rank', color=qc, linewidth=2)
            
        if vr_key in history:
            data = np.array(history[vr_key])
            step_size = max(1, len(data) // 100)
            x_axis = steps[::step_size] if steps is not None and len(steps) == len(data) else range(0, len(data)*step_size, step_size)
            plt.plot(x_axis, data[::step_size], label=f'L{layer_idx} V-Update Rank', color=vc, linestyle='--', linewidth=2)

    plt.title('Update Rank Evolution: Q-Needle vs V-Wave', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Effective Rank of Momentum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'update_rank_evolution.png'), dpi=300)
    plt.close()

    print(f"Research plots successfully saved to {save_dir}/")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=str, default="2000000")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--metrics_file", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
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
        run_tracking_and_plot(args.tokens, args.resume)
