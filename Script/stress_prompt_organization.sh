#!/bin/bash
# Organize prompts for StressQA benchmark
# Uses StressTest/prompt_organization.py with enhanced JSON contract

echo "=========================================="
echo "StressQA Prompt Organization"
echo "=========================================="

cd "$(dirname "$0")/.." || exit

# Default values
TRICK_NAME="zero-shot"
DATASET_NAME="mini-StressQA"

# Parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        --trick)
            TRICK_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--trick TRICK_NAME] [--dataset DATASET_NAME]"
            echo "  TRICK_NAME: zero-shot, one-shot, two-shot, zero-shot-CoT, one-shot-CoT, stats-prompt"
            echo "  DATASET_NAME: StressQA, mini-StressQA, etc."
            exit 1
            ;;
    esac
done

echo "Organizing prompts for:"
echo "  Dataset: $DATASET_NAME"
echo "  Trick: $TRICK_NAME"
echo ""

# Check if StressQA dataset exists
STRESSQA_PATH="Data/Integrated Dataset/Balanced Benchmark/${DATASET_NAME}.csv"

if [ ! -f "$STRESSQA_PATH" ]; then
    echo "Error: Dataset not found at $STRESSQA_PATH"
    echo "Please run stress_benchmark_construction.sh first"
    exit 1
fi

# Run StressQA prompt organization
python StressTest/prompt_organization.py \
    --trick_name "$TRICK_NAME" \
    --integ_dataset_name "$DATASET_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "StressQA prompt organization complete!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Error: Prompt organization failed"
    echo "=========================================="
    exit 1
fi

