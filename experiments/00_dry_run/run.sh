#!/bin/bash

# Experiment: 00_dry_run
# This script launches a single dry run to verify the new logging and directory structure.

RUN_NAME="muon_dry_run"
EXP_DIR="experiments/00_dry_run/${RUN_NAME}"

echo "🚀 Starting Dry Run: ${RUN_NAME}"

# Ensure directories exist
mkdir -p "${EXP_DIR}/checkpoints"
mkdir -p "${EXP_DIR}/metrics/raw"

python train_llm.py \
    --config_yaml experiments/00_dry_run/config.yaml \
    --output_dir "${EXP_DIR}" \
    --checkpoint_dir "${EXP_DIR}/checkpoints" \
    --raw_metrics_dir "${EXP_DIR}/metrics/raw" \
    --log_every 100

echo "✅ Dry Run Completed"
