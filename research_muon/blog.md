You might notice that the spectral norm of the LLM weights is growing during training with Muon optimizer.

![Spectral Evolution](../results/research_plots/spectral_stretching_evolution.png)

At first it seem weird -> isn't Muon making spectral norms = 1?

Muon is making weight update matrices $(\Delta W)$ orthogonal (spectral norm = 1) but adding matrices makes their spectral norms also add.