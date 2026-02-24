import os
import sys
import json
import argparse
from pathlib import Path

def subsample_singular_values(sigma_full):
    """Subsamples singular values identically to the updated trainer.py"""
    if len(sigma_full) > 50:
        top_10 = sigma_full[:10]
        bottom_10 = sigma_full[-10:]
        middle = sigma_full[10:-10]
        step_size = max(1, len(middle) // 30)
        middle_sampled = middle[::step_size][:30]
        sigma = top_10 + middle_sampled + bottom_10
    else:
        sigma = sigma_full
    return sigma

def compress_jsonl_file(file_path):
    print(f"Compressing: {file_path}")
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return
        
    original_size = file_path.stat().st_size
    temp_path = file_path.with_suffix('.jsonl.tmp')
    
    # Track how much we compress (both singular values and log freq)
    # The new target log frequency is every 2500 steps.
    # We will keep steps where step % 2500 == 0 OR if it's the very first/last few logs we see just in case (e.g. step 500 is kept to mimic log_milestones)
    # Actually, to make sure plots don't break, keeping every 5th log (500 * 5 = 2500)
    
    lines_kept = 0
    lines_total = 0
    
    with open(file_path, 'r') as f_in, open(temp_path, 'w') as f_out:
        for line in f_in:
            lines_total += 1
            record = json.loads(line)
            step = record.get("step", 0)
            
            # Keep log if it matches new interval (250) OR specific early milestones
            if step % 250 == 0 or step in [100, 500, 1000]:
                original_sv = record.get("singular_values", [])
                
                # We can't subsample if they were already subsampled or missing
                if isinstance(original_sv, list) and len(original_sv) > 50:
                    record["singular_values"] = subsample_singular_values(original_sv)
                
                f_out.write(json.dumps(record) + '\n')
                lines_kept += 1

    new_size = temp_path.stat().st_size
    
    # Replace original with compressed
    os.replace(temp_path, file_path)
    
    print(f"  Done. Lines: {lines_total} -> {lines_kept}")
    print(f"  Size: {original_size / 1024 / 1024:.2f} MB -> {new_size / 1024 / 1024:.2f} MB ({(new_size/original_size)*100:.1f}%)")

if __name__ == "__main__":
    # Target files to compress
    base_dir = Path("experiments/01_muon_vs_adamw_baseline")
    
    for metrics_dir in base_dir.rglob("metrics"):
        target_file = metrics_dir / "manifold_stats.jsonl"
        if target_file.exists():
            compress_jsonl_file(target_file)
