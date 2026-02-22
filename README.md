# Muon Research Lab (Blueberry 88M)

This repository is dedicated to the research and analysis of the **Muon optimizer** and its impact on Large Language Model (LLM) training dynamics. We shift our focus from competitive speedrunning to deep structural investigation of how manifold-aware optimization shapes transformer representations.

## 🔬 Research Focus: Hierarchical Stretching

Our primary research direction is **Hierarchical Stretching**—the observation that Muon-trained models dynamically modulate spectral signatures across layers. We provide tools to track these manifolds in real-time.

### Key Research Features:
- **Manifold Tracking**: Track spectral norms, spectral gaps, and singular value distributions per-layer.
- **Spectral Analysis Utilities**: High-performance SVD-based statistics calculation (CPU-offloaded).
- **Visualization Suite**: Generate hierarchical stretching reports and layer-wise heatmaps.

---

## 🚀 Getting Started

### 0. Setup

```bash
pip install -r requirements.txt
```

Download dataset:
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

### 1. Training with Manifold Tracking
To train a model while tracking its manifold evolution, use the `--track_manifold true` flag:

```bash
python train_llm.py --train_tokens 8000000 --track_manifold true
```

### 2. Generating Research Plots
After training, you can generate the research visualizations using the tracking script:

```bash
python research_muon/track_manifold.py --plot_only
```

Plots will be saved to `results/research_plots/`.

---

## 📊 Experimental Setup
- **Model**: Blueberry (88M Params, 22 layers, 512 d_model)
- **Optimizer**: Muon (Orthogonalized 2D Updates)
- **Dataset**: speedrun_40M (Curated subset)

## 📈 Recent Findings
Check out our latest research reports in the `research_muon/` folder:
- [Hierarchical Stretching Report](research_muon/hierarchical_stretching.md)
- [Modular Manifolds Concept](research_muon/modular-manifolds.md)

---

## 🤝 Partners & Support

If you are interested in collaborating on Muon research or manifold optimization, please reach out. We aim to keep all findings fully open source.

**Partners include:** Hugging Face, NVIDIA, Microsoft, Google, and more.


