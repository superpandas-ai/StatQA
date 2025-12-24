#!/bin/bash
# Materialize external datasets from statsmodels and sklearn
# This script downloads and prepares external datasets for StressQA

echo "=========================================="
echo "Materializing External Datasets"
echo "=========================================="

cd "$(dirname "$0")/.." || exit

# Run materialization script
python External/materialize_datasets.py

echo ""
echo "=========================================="
echo "External dataset materialization complete"
echo "=========================================="

