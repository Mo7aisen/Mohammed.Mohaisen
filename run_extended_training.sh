#!/bin/bash
# Extended training script for overfitting demonstrations
# Based on supervisor's meeting notes

set -e

echo "========================================================"
echo "    STARTING EXTENDED TRAINING FOR OVERFITTING         "
echo "========================================================"
echo "Note: This will continue training existing models to"
echo "demonstrate overfitting behavior as requested."
echo ""

# Check if Phoenix server is available
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: GPU not available. Training will use CPU (slower)."
fi

# Create extended training log directory
mkdir -p outputs/extended_training_logs

# Extended training jobs based on meeting notes
declare -A training_jobs=(
    ["montgomery_full"]="250"
    ["montgomery_half"]="300"
    ["jsrt_half"]="250"
)

echo "Planned extended training jobs:"
for dataset in "${!training_jobs[@]}"; do
    echo "  - $dataset to ${training_jobs[$dataset]} epochs"
done
echo ""

# Run each extended training job
for dataset in "${!training_jobs[@]}"; do
    target_epochs="${training_jobs[$dataset]}"
    log_file="outputs/extended_training_logs/${dataset}_to_${target_epochs}_epochs.log"

    echo "--------------------------------------------------------"
    echo "Starting: $dataset to $target_epochs epochs"
    echo "Log file: $log_file"

    # Check if already completed
    final_model="outputs/unet_${dataset}_${target_epochs}/final_model.pth"
    if [ -f "$final_model" ]; then
        echo "Already completed. Skipping."
        continue
    fi

    # Run extended training
    python3 train_extended.py \
        --dataset "$dataset" \
        --target_epochs "$target_epochs" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Completed successfully"
    else
        echo "✗ Failed! Check log file for details"
        exit 1
    fi
done

echo ""
echo "========================================================"
echo "    EXTENDED TRAINING COMPLETED                         "
echo "========================================================"
echo ""
echo "Next step: Run evaluation on the new overfitting models"
echo "Use: ./run_extended_evaluation.sh"