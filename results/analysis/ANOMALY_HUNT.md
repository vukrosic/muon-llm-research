# Experiment 01: Baseline Anomaly Hunt Results (Debug 2M Tokens)

This document displays the comparative spectral dynamics between Muon (Blue) and AdamW (Red) across layers and projections.

## 1. Spectral Entropy
Rows: Layers (1, 3, 6, 9, 12) | Columns: Projections (Q, K, V, O)
![Spectral Entropy](baseline_comparison_entropy.png)

---

## 2. Update-Weight Alignment
Measures the geometric lock-in between the optimizer's updates and the weight manifold.
![Update Alignment](baseline_comparison_update_alignment.png)

---

## 3. Effective Rank
Tracking the "Needle vs Wave" hypothesis: how many dimensions the optimizer is utilizing.
![Effective Rank](baseline_comparison_effective_rank.png)

---

## 4. Gradient Norms
Spectral magnitude of the gradients.
![Grad Norm](baseline_comparison_grad_norm.png)

---

## 5. Weight Norms
Evolution of the parameter norms.
![Weight Norm](baseline_comparison_weight_norm.png)

---

# Experiment Critique for Anomaly Hunting

The primary goal of this experiment framework is to identify *anomalies*—sudden shifts, irregular alignments, and dimension collapse. The current setup, while robust for general metric tracking, has a few crucial blind spots for pure anomaly hunting.

### 1. The Averaging Trap (Resolved)
Initially, `analysis/compare_baseline.py` was interpolating and averaging metrics across all seeds, plotting the mean and standard deviation. 
**Resolution:** This has been resolved. The runs are now properly tracked, and heatmaps visualize variance without dilution.

### 2. Missing Layers in Visualization (Resolved)
**Resolution:** `compare_baseline.py` has been completely rewritten to generate **Heatmaps**, plotting Layer Depth (Y-axis) vs Step (X-axis) across all layers, colored by metric intensity.

### 3. Subspace Alignment Blind Spot (Resolved)
**Resolution:** Modified `utils/spectral.py` to independently compute and return `(left_alignment, right_alignment)`. `trainer.py` now logs both, tracking both Output and Input subspace anomalies independently!

### 4. Downsampling & Missing the Peak (Resolved)
**Resolution:** To avoid exponentially slowing down training with SVD operations on every step, `trainer.py` now accumulates a continuous `running_max` of `grad_norm` and `weight_norm` (using fast `torch.norm()`) inline inside the rapid optimizer steps. This peak is serialized at the `log_every` interval, meaning we now catch all explosive momentary divergences without lagging GPU utilization.

### 5. Muon vs AdamW Volatility (Pending Future Run)
Even on a brief 2M token run, exact point-to-point variance shows that Muon has roughly **10x higher variance** between seeds for `update_alignment` than AdamW (0.26 variance vs 0.03 variance). Future experiments should lengthen the sequence to see if Muon converging on divergent local minima actually yields different spectral structures (e.g., does high `update_alignment` variance predict early over-fitting?).
