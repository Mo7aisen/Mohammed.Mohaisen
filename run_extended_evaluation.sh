#!/bin/bash
# Evaluation script for extended training models

set -e

echo "========================================================"
echo "    EVALUATING EXTENDED TRAINING MODELS                 "
echo "========================================================"

# Extended models to evaluate
declare -A extended_models=(
    ["unet_montgomery_full_250"]="montgomery_full_250"
    ["unet_montgomery_half_300"]="montgomery_half_300"
    ["unet_jsrt_half_250"]="jsrt_half_250"
)

splits="test validation training"

for run_name in "${!extended_models[@]}"; do
    echo "========================================================"
    echo "Evaluating: $run_name"

    for split in $splits; do
        echo "  - Split: $split"

        eval_dir="outputs/${run_name}/evaluation/${split}"
        summary_file="${eval_dir}/_evaluation_summary.json"

        if [ -f "$summary_file" ]; then
            echo "    Already evaluated. Skipping."
        else
            echo "    Running evaluation..."
            python3 evaluate.py \
                --run_name "$run_name" \
                --split "$split" \
                > "outputs/${run_name}/eval_${split}_extended.log" 2>&1

            if [ $? -eq 0 ]; then
                echo "    ✓ Completed"
            else
                echo "    ✗ Failed"
            fi
        fi
    done
done

echo ""
echo "========================================================"
echo "    EXTENDED EVALUATION COMPLETED                       "
echo "========================================================"
echo ""
echo "All models are now ready for the Streamlit application!"
echo "Run: streamlit run app.py"