#!/bin/bash
# StressQA Benchmark Construction Pipeline
# This script runs the full pipeline to generate the StressQA benchmark

echo "=========================================="
echo "StressQA Benchmark Construction Pipeline"
echo "=========================================="

cd "$(dirname "$0")/.." || exit

# Step 1: Materialize external datasets (if not already done)
echo ""
echo "[Step 1/3] Materializing external datasets..."
if [ -f "Data/External Dataset/Origin/_dataset_inventory.csv" ]; then
    echo "  ✓ External datasets already materialized, skipping..."
else
    python External/materialize_datasets.py
fi

# Step 2: Generate StressQA benchmark
echo ""
echo "[Step 2/3] Generating StressQA benchmark..."
python StressTest/benchmark_generator.py

# Step 3: Verify outputs
echo ""
echo "[Step 3/3] Verifying outputs..."

STRESSQA_CSV="Data/Integrated Dataset/Balanced Benchmark/StressQA.csv"
STRESSQA_JSON="Data/Integrated Dataset/Balanced Benchmark/StressQA.json"
MINI_CSV="Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.csv"

if [ -f "$STRESSQA_CSV" ] && [ -f "$STRESSQA_JSON" ]; then
    echo "  ✓ StressQA benchmark files created successfully"
    
    # Count rows
    ROWS=$(wc -l < "$STRESSQA_CSV")
    echo "  ✓ StressQA contains $ROWS rows (including header)"
    
    if [ -f "$MINI_CSV" ]; then
        MINI_ROWS=$(wc -l < "$MINI_CSV")
        echo "  ✓ mini-StressQA contains $MINI_ROWS rows (including header)"
    fi
else
    echo "  ✗ Error: StressQA benchmark files not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "StressQA benchmark construction complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - $STRESSQA_CSV"
echo "  - $STRESSQA_JSON"
echo "  - $MINI_CSV"
echo ""

