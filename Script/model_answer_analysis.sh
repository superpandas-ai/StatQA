#!/bin/bash

# StatQA Model Answer Analysis - New Unified Framework
# This script demonstrates how to use the new statqa_analysis package

cd "$(dirname "$0")/.." || exit 1

echo "========================================="
echo "StatQA Model Answer Analysis"
echo "========================================="

# Example 1: Analyze specific model outputs (single runs)
echo ""
echo "[1] Analyzing individual model outputs..."

# Analyze GPT-3.5 models
if [ -f "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" ]; then
    python -m statqa_analysis run \
        --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
        --out "AnalysisOutput" \
        --run-id "gpt-3.5-turbo_zero-shot"
fi

if [ -f "Model Answer/Origin Answer/gpt-3.5-turbo_one-shot.csv" ]; then
    python -m statqa_analysis run \
        --input "Model Answer/Origin Answer/gpt-3.5-turbo_one-shot.csv" \
        --out "AnalysisOutput" \
        --run-id "gpt-3.5-turbo_one-shot"
fi

if [ -f "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot-CoT.csv" ]; then
    python -m statqa_analysis run \
        --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot-CoT.csv" \
        --out "AnalysisOutput" \
        --run-id "gpt-3.5-turbo_zero-shot-CoT"
fi

if [ -f "Model Answer/Origin Answer/gpt-3.5-turbo_one-shot-CoT.csv" ]; then
    python -m statqa_analysis run \
        --input "Model Answer/Origin Answer/gpt-3.5-turbo_one-shot-CoT.csv" \
        --out "AnalysisOutput" \
        --run-id "gpt-3.5-turbo_one-shot-CoT"
fi

if [ -f "Model Answer/Origin Answer/gpt-3.5-turbo_stats-prompt.csv" ]; then
    python -m statqa_analysis run \
        --input "Model Answer/Origin Answer/gpt-3.5-turbo_stats-prompt.csv" \
        --out "AnalysisOutput" \
        --run-id "gpt-3.5-turbo_stats-prompt"
fi

# Example 2: Cohort analysis across all processed runs
echo ""
echo "[2] Generating cohort analysis..."
if [ -d "AnalysisOutput/runs" ]; then
    python -m statqa_analysis cohort \
        --input-dir "AnalysisOutput/runs" \
        --out "AnalysisOutput" \
        --cohort-name "gpt-3.5-experiments"
fi

echo ""
echo "========================================="
echo "Analysis Complete!"
echo "========================================="
echo "Results are in: AnalysisOutput/"
echo ""
echo "Single runs:    AnalysisOutput/runs/<run_id>/"
echo "Cohort summary: AnalysisOutput/cohorts/<cohort_name>/"
echo ""

