
import json
import os
from research_muon.track_manifold import plot_results

files = {
    'adamw': 'plots/metrics_adamw_2000000_20260223_162503.json',
    'muon': 'plots/metrics_muon_2000000_20260223_160932.json'
}

for opt, path in files.items():
    print(f"Generating plots for {opt}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    save_dir = f"results/{opt}_2M"
    os.makedirs(save_dir, exist_ok=True)
    
    if 'manifold_history' in data['history']:
        # The function expects the manifold_history dict or the whole history if it contains 'manifold_history'
        plot_results(data['history'], save_dir)
        print(f"Plots saved to {save_dir}")
    else:
        print(f"No manifold history found for {opt}")
