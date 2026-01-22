#!/bin/bash

# Check for config file argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_config_json>"
    exit 1
fi

CONFIG_FILE=$1

# Base directory for logs
LOGS_BASE_DIR="logs"

# Load configurations from the provided JSON file
CONFIGS=$(cat "$CONFIG_FILE")
CONFIGS_LENGTH=$(echo "$CONFIGS" | jq '. | length')

# Loop through configurations
for i in $(seq 0 $((CONFIGS_LENGTH - 1))); do
    CONFIG=$(echo "$CONFIGS" | jq -r ".[$i]")

    # Generate timestamped logs directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOGS_DIR="$LOGS_BASE_DIR/run_$TIMESTAMP"
    mkdir -p "$LOGS_DIR"
   
    # Redirect all output to logs.txt
    {
        echo "Starting run $((i + 1))..."
        echo "Command: python pipeline.py $CONFIG"

        python pipeline.py \
            --batch_size 32 \
            --sample_size 512 \
            --threshold 0.05 \
            --save_histograms 1 \
            --save_independence_heatmaps 1 \
            --output_dir "$LOGS_DIR" \
            --num_data_workers 6 \
            --max_workers 3 \
            --run_id "run_$TIMESTAMP" \
            --gpu "0" \
            $CONFIG

        echo "Run $((i + 1)) complete."
    } &> "$LOGS_DIR/logs.txt"
done
