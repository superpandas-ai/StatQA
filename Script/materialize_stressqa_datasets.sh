#!/bin/bash
# Materialize Modified Datasets for StressQA
# This script materializes (saves) all modified datasets referenced in the StressQA benchmark

echo "=========================================="
echo "StressQA Dataset Materialization"
echo "=========================================="

cd "$(dirname "$0")/.." || exit

# Default paths
BENCHMARK_FILE="${1:-Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.json}"
OUTPUT_DIR="${2:-Data/External Dataset/Processed/}"

echo ""
echo "Benchmark file: $BENCHMARK_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run materialization script
python StressTest/materialize_datasets.py \
    --benchmark-file "$BENCHMARK_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --random-state 42 \
    --update-benchmark

echo ""
echo "=========================================="
echo "Materialization complete!"
echo "=========================================="
echo ""
echo "Materialized datasets are available in: $OUTPUT_DIR"
echo "You can now use these datasets with the columns referenced in relevant_column"
echo ""

