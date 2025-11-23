#!/bin/bash

# Base directory for logs
LOGS_BASE_DIR="logs"

# Generate timestamped logs directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGS_DIR="$LOGS_BASE_DIR/run_$TIMESTAMP"
mkdir -p "$LOGS_DIR"

# Redirect all output (stdout + stderr) to logs.txt
{
    echo "Starting run..."

    python inference.py \
        --ensemble_test "manual-stouffer" \
        --batch_size 16 \
        --output_dir "$LOGS_DIR" \
        --num_data_workers 2 \
        --max_workers 2 \
        --experiment_id "ecdf_robustness" \
        --gpu "0" \
        --independent_keys \
            PatchProcessing_statistic=RIGID.DINO.05_patch_size=512_seed=38 \
            PatchProcessing_statistic=RIGID.CLIPOPENAI.05_patch_size=512_seed=38 \
        --patch_divisors 0 \
        --cdf_bins 400 \
        --dataset_type CELEBA_REAL_ONLY \
        --seed 38 \
        --sample_size 512 \
        --threshold 0.9 \
        --save_histograms 1 \
        --draw_pvalues_trend_figure 1 \
        --pkls_dir pkls/AIStats/new_stats \
        --run_id inference_run_$TIMESTAMP \
        --inference_aug none

    echo "Run complete."
} &> "$LOGS_DIR/logs.txt"
