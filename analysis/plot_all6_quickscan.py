import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("experiments/01_muon_vs_adamw_baseline")
OUT_DIR = BASE_DIR / "plots_combined"
EXPERIMENTS = [
    ("muon", 42),
    ("muon", 137),
    ("muon", 256),
    ("adamw", 42),
    ("adamw", 137),
    ("adamw", 256),
]


def compute_effective_rank(singular_values):
    s = np.array(singular_values, dtype=float)
    if s.size == 0:
        return 0.0
    p = s / (np.sum(s) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def load_run_matrix(opt, seed):
    stats_path = BASE_DIR / f"{opt}_seed{seed}" / "metrics" / "manifold_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing manifold stats: {stats_path}")

    by_key = {}
    steps_set = set()
    layers_set = set()

    with stats_path.open("r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" not in row or "layer" not in row or "singular_values" not in row:
                continue
            key = (int(row["step"]), int(row["layer"]))
            by_key.setdefault(key, []).append(compute_effective_rank(row["singular_values"]))
            steps_set.add(key[0])
            layers_set.add(key[1])

    steps = sorted(steps_set)
    layers = sorted(layers_set)
    if not steps or not layers:
        raise ValueError(f"No usable records in {stats_path}")

    matrix = np.full((len(layers), len(steps)), np.nan, dtype=float)
    step_idx = {s: i for i, s in enumerate(steps)}
    layer_idx = {l: i for i, l in enumerate(layers)}

    for (step, layer), values in by_key.items():
        matrix[layer_idx[layer], step_idx[step]] = float(np.mean(values))

    return steps, layers, matrix


def save_single_image(opt, seed, steps, layers, matrix, vmin, vmax):
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=[steps[0], steps[-1], layers[0], layers[-1]],
    )
    ax.set_title(f"{opt.upper()} seed {seed}", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = OUT_DIR / f"scan_effective_rank_{opt}_seed{seed}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_grid(all_runs, vmin, vmax):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.flatten()
    last_im = None

    for i, run in enumerate(all_runs):
        ax = axes[i]
        steps = run["steps"]
        layers = run["layers"]
        matrix = run["matrix"]
        last_im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=[steps[0], steps[-1], layers[0], layers[-1]],
        )
        ax.set_title(f"{run['opt'].upper()} seed {run['seed']}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), fraction=0.015, pad=0.02, label="Effective Rank")

    out_path = OUT_DIR / "scan_effective_rank_grid_all6.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = []

    for opt, seed in EXPERIMENTS:
        steps, layers, matrix = load_run_matrix(opt, seed)
        all_runs.append(
            {
                "opt": opt,
                "seed": seed,
                "steps": steps,
                "layers": layers,
                "matrix": matrix,
            }
        )

    all_values = np.concatenate([r["matrix"][~np.isnan(r["matrix"])] for r in all_runs])
    vmin = float(np.percentile(all_values, 1))
    vmax = float(np.percentile(all_values, 99))

    for run in all_runs:
        save_single_image(
            run["opt"],
            run["seed"],
            run["steps"],
            run["layers"],
            run["matrix"],
            vmin,
            vmax,
        )

    grid_path = save_grid(all_runs, vmin, vmax)
    print(f"Saved all six quick-scan images to: {OUT_DIR}")
    print(f"Saved combined grid to: {grid_path}")


if __name__ == "__main__":
    main()
