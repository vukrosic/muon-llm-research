Today I did LLM + Muon research

I got a bit confused - wasn’t Muon optimizer supposed to make spectral norms (max singular values) of weight update matrices = 1, so why are my weights growing to 50+?

Figure 1: Spectral norm of Query and Value projection matrices in first and last layers. 88M params, 100M tokens training

Well, adding matrices will grow the spectral norm (check triangle inequality).

But why do Value spec. norms grow a lot more than Query spec. norms?

![Spectral Norm vs Depth](results/research_plots/spectral_norm_vs_depth.png)