
import subprocess
import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def run_adamw_search():
    # Learning rates to test around the current 6e-3
    learning_rates = [1e-3, 3e-3, 6e-3, 1e-2, 2e-2]
    tokens = 1000000  # 1M tokens for quick search
    results = {}

    output_base = "results/adamw_lr_search"
    os.makedirs(output_base, exist_ok=True)

    for lr in learning_rates:
        print(f"\n🚀 Testing AdamW with LR: {lr}")
        run_name = f"adamw_lr_{lr}"
        output_dir = os.path.join(output_base, run_name)
        
        cmd = [
            "python", "train_llm.py",
            "--optimizer", "adamw",
            "--adamw_lr", str(lr),
            "--train_tokens", str(tokens),
            "--output_dir", output_dir,
            "--track_manifold", "false"  # Disable manifold tracking for speed
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Find the metrics file
            metrics_files = glob.glob(f"plots/metrics_adamw_{tokens}_*.json")
            if metrics_files:
                metrics_files.sort(key=os.path.getmtime)
                latest_metrics = metrics_files[-1]
                with open(latest_metrics, 'r') as f:
                    data = json.load(f)
                    val_loss = data['final_metrics']['val_loss']
                    results[lr] = val_loss
                    print(f"✅ LR {lr} finished with Val Loss: {val_loss:.4f}")
        except Exception as e:
            print(f"❌ LR {lr} failed: {e}")

    # Plot results
    if results:
        lrs = sorted(results.keys())
        losses = [results[lr] for lr in lrs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, marker='o', linestyle='-', color='red')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Validation Loss after 1M tokens')
        plt.title('AdamW Learning Rate Search (1M tokens)')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        save_path = os.path.join(output_base, "lr_search_results.png")
        plt.savefig(save_path)
        print(f"\n📊 Summary plot saved to {save_path}")
        
        best_lr = min(results, key=results.get)
        print(f"🏆 Best Learning Rate found: {best_lr} (Loss: {results[best_lr]:.4f})")

if __name__ == "__main__":
    run_adamw_search()
