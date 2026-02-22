# 🧠 The Subspace Persistence Effect: Geometric Mechanics of Muon-Driven LLMs

Following our 20,000,000 token training run, we now have high-fidelity evidence of how the **Muon optimizer** sculptures the internal geometry of a Transformer. By tracking the **Subspace Persistence Rate**—the alignment between weights and their updates—we've identified the specific geometric signature of representational maturity.

![Spectral Evolution](../results/research_plots/spectral_stretching_evolution.png)

## 🔍 Core Research Finding: The "Lock-in" Gradient

Our results confirm that **Hierarchical Stretching** is not just a growth in magnitude, but a permanent geometric shift. We tracked the **Subspace Alignment Score ($\rho$)** and **Normalized Spectral Entropy ($H_{spec}$)** across all 22 layers, revealing a clear "persistence gradient."

### 1. The Value Projection ($\text{V}$) is the Seat of Specialized Knowledge
We observed a significant divergence in the alignment of **Value projections** compared to Query/Key projections.
- **Deep Layers (L21)**: The Value-projection alignment peaked during the "Lock-in" phase—nearly **2x higher** than the initial foundation layers.
- **Implication**: Deep layers "decide" on their semantic output space (the V-subspace) much earlier and more decisively. Once this subspace is found, Muon consistently reinforces it, leading to the rapid spectral stretching seen in our plots.

### 2. Spectral Entropy Decay as an Informational Bottleneck
As predicted, **Spectral Entropy $(H_{spec})$** acts as a thermometer for layer maturation.
- **Foundation Layers**: Maintained a high entropy (~0.89), indicating they are preserving a wide, full-rank basis to extract diverse features from raw tokens.
- **Semantic Layers**: Showed a decisive drop in the V-projections as training stabilized.
- **Conclusion**: This entropy decay is the geometric manifestation of the **Information Bottleneck**. Deep layers are effectively "throwing away" unused singular dimensions to concentrate all gain into a specialized semantic subspace.

---

## 🛠️ Discussion: Subspace Refinement vs. Manifold Drift

This data suggests that Muon improves training stability by **suppressing manifold drift**. 

In standard optimizers, high learning rates often cause the weight matrix to "fishtail" across the manifold, leading to instability. Muon's orthogonal constraint, combined with the observed **Subspace Persistence**, indicates that the model follows a "geodesic-like" path:
1. It identifies a relevant singular subspace early.
2. It persistently aligns its updates with that subspace.
3. It scales the representational gain (spectral norm) within that stable frame.

### 📈 Final Metrics from the 20M Run
- **Subspace Alignment (L21, V)**: Actively Reinforcing regime (Alignment > 0.40 during peak).
- **Max Spectral Norm (L21, Q)**: 14.1 (Decisive stretching).
- **Spectral Gap (L21, Q)**: 1.27 (Signature of Feature Focus).

---

*(The following plots represent the final geometric state of the model at 20M tokens.)*

![Update Alignment](../results/research_plots/update_alignment.png)
![Entropy Heatmap](../results/research_plots/entropy_heatmap.png)

---
*Created by the Muon LLM Research Team*
