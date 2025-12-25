#!/bin/bash
# StressQA Benchmark Construction Pipeline
# This script runs the full pipeline to generate the StressQA benchmark

echo "=========================================="
echo "StressQA Benchmark Construction Pipeline"
echo "=========================================="
echo ""
echo "[i] Note: By default, this pipeline does NOT use API tokens."
echo "    All question generation is done programmatically using templates."
echo ""
echo "[i] Optional: Set ENABLE_REFINEMENT=true to enable GPT question refinement."
echo "    This will improve question quality but consumes API tokens."
echo ""

cd "$(dirname "$0")/.." || exit

# Step 1: Materialize external datasets (if not already done)
echo ""
echo "[Step 1/5] Materializing external datasets..."
if [ -f "Data/External Dataset/Origin/_dataset_inventory.csv" ]; then
    echo "  ✓ External datasets already materialized, skipping..."
else
    python External/materialize_datasets.py
fi

# Step 2: Generate StressQA benchmark
echo ""
echo "[Step 2/5] Generating StressQA benchmark..."
python StressTest/benchmark_generator.py

# Define output file paths
STRESSQA_CSV="Data/Integrated Dataset/Balanced Benchmark/StressQA.csv"
STRESSQA_JSON="Data/Integrated Dataset/Balanced Benchmark/StressQA.json"
MINI_CSV="Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.csv"
MINI_JSON="Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.json"

# Step 3: Optional GPT question refinement
echo ""
echo "[Step 3/5] Optional GPT question refinement..."
if [ "$ENABLE_REFINEMENT" = "true" ] || [ "$ENABLE_REFINEMENT" = "1" ]; then
    echo "  [*] Refinement enabled. This will use API tokens."
    echo "  [*] Refining questions for mini-StressQA..."
    if [ -f "$MINI_JSON" ]; then
        python StressTest/gpt_refine_question.py \
            --benchmark-file "$MINI_JSON" \
            --batch-size 20
        # Also update CSV from JSON
        python -c "import pandas as pd; import json; df = pd.read_json('$MINI_JSON'); df.to_csv('$MINI_CSV', index=False)"
        echo "  ✓ Questions refined for mini-StressQA"
    fi
    
    if [ -f "$STRESSQA_JSON" ]; then
        echo "  [*] Refining questions for StressQA (this may take a while)..."
        python StressTest/gpt_refine_question.py \
            --benchmark-file "$STRESSQA_JSON" \
            --batch-size 20
        # Also update CSV from JSON
        python -c "import pandas as pd; import json; df = pd.read_json('$STRESSQA_JSON'); df.to_csv('$STRESSQA_CSV', index=False)"
        echo "  ✓ Questions refined for StressQA"
    fi
else
    echo "  [i] Refinement skipped (default: API-token-free mode)"
    echo "  [i] To enable refinement, set ENABLE_REFINEMENT=true"
    echo "  [i] Example: ENABLE_REFINEMENT=true sh Script/stress_benchmark_construction.sh"
fi

# Step 4: Verify outputs
echo ""
echo "[Step 4/5] Verifying outputs..."

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

# Step 5: Materialize modified datasets
echo ""
echo "[Step 5/5] Materializing modified datasets..."
if [ -f "$MINI_JSON" ]; then
    python StressTest/materialize_datasets.py \
        --benchmark-file "$MINI_JSON" \
        --output-dir "Data/External Dataset/Processed/" \
        --random-state 42 \
        --update-benchmark
    echo "  ✓ Modified datasets materialized for mini-StressQA"
fi

if [ -f "$STRESSQA_JSON" ]; then
    python StressTest/materialize_datasets.py \
        --benchmark-file "$STRESSQA_JSON" \
        --output-dir "Data/External Dataset/Processed/" \
        --random-state 42 \
        --update-benchmark
    echo "  ✓ Modified datasets materialized for StressQA"
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
echo "  - $MINI_JSON"
echo ""
echo "Materialized datasets:"
echo "  - Data/External Dataset/Processed/*.csv"
echo ""

