#!/bin/bash
# Organize prompts for StressQA benchmark
# Similar to prompt_organization.sh but for StressQA with enhanced JSON contract

echo "=========================================="
echo "StressQA Prompt Organization"
echo "=========================================="

cd "$(dirname "$0")/.." || exit

# Default values
TRICK_NAME="zero-shot"
DATASET_NAME="mini-StressQA"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
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

# Note: This would need a StressQA-specific prompt organizer
# For now, we can use the existing one with extended prompt wording
# A proper implementation would create StressTest/prompt_organization.py

# Check if StressQA dataset exists
STRESSQA_PATH="Data/Integrated Dataset/Balanced Benchmark/${DATASET_NAME}.csv"

if [ ! -f "$STRESSQA_PATH" ]; then
    echo "Error: Dataset not found at $STRESSQA_PATH"
    echo "Please run stress_benchmark_construction.sh first"
    exit 1
fi

echo "Note: StressQA uses enhanced JSON contract with audit trails"
echo "The prompt wording has been extended in prompt_wording.py"
echo ""
echo "To organize prompts, you can adapt Construction/prompt_organization.py"
echo "to use STRESSQA_INSTRUCTION and STRESSQA_RESPONSE from prompt_wording.py"
echo ""
echo "For now, StressQA benchmark is ready for evaluation with:"
echo "  - Extended method list (Group Comparison, Regression, Multiple Testing)"
echo "  - Enhanced JSON contract fields"
echo "  - Oracle computations with tolerances"
echo ""

echo "=========================================="
echo "StressQA prompt organization info displayed"
echo "=========================================="

