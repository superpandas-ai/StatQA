# StatQA Analysis Framework - Quick Start

## 🚀 Get Started in 2 Minutes

### Option 1: Unified Workflow (Recommended)

The new unified framework supports the complete pipeline from dataset import to analysis:

```bash
# 1. Import dataset
python -m statqa_analysis dataset import \
  --dataset-id mini-statqa \
  --from-json "Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json"

# 2. Build prompts
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot

# 3. Run model inference
python -m statqa_analysis model run \
  --model gpt-4 \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot

# 4. Analyze results
python -m statqa_analysis run \
  --input AnalysisOutput/runs/gpt-4_mini-statqa_v1_zero-shot/raw/model_output.csv \
  --dataset-id mini-statqa
```

### Option 2: Analyze Existing Outputs

If you already have model outputs:

```bash
# Analyze a single model output (with dataset metadata)
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
    --dataset-id mini-statqa \
    --out "AnalysisOutput"

# Or without dataset-id (if CSV has all metadata)
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
    --out "AnalysisOutput"
```

### Option 3: Python API

```python
from statqa_analysis import ModelOutputAnalyzer

analyzer = ModelOutputAnalyzer(
    input_csv="Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv",
    run_id="gpt-3.5-turbo_zero-shot"
)
context = analyzer.run_all()

# Access results
print(f"Overall accuracy: {context.get_result('overall_accuracy'):.4f}")
```

## 📊 What You Get

After running analysis, you'll find:

```
AnalysisOutput/runs/<run_id>/
├── artifacts/
│   └── processed.csv              # Full data with all scores
├── tables/
│   ├── task_performance_methods.csv
│   ├── task_performance_columns.csv
│   ├── task_performance_overall.csv
│   └── error_analysis_summary.csv
└── plots/
    └── confusion_matrix.pdf
```

## 🔄 Cohort Analysis (Compare Multiple Models)

After analyzing individual runs:

```bash
python -m statqa_analysis cohort \
    --input-dir "AnalysisOutput/runs" \
    --out "AnalysisOutput" \
    --cohort-name "my-experiments"
```

This generates:
- Summary tables across all models
- Radar charts comparing performance
- Aggregated error analysis

## 🎨 Customization

```bash
# Use Jaccard index instead of accuracy
python -m statqa_analysis run \
    --input model.csv \
    --method-metric jaccard

# Generate PNG plots instead of PDF
python -m statqa_analysis run \
    --input model.csv \
    --plot-format png

# Disable plots for faster processing
python -m statqa_analysis run \
    --input model.csv \
    --no-plots
```

## 📚 Full Documentation

- **Unified Workflow Guide**: `UNIFIED_WORKFLOW.md` ⭐ (Complete end-to-end guide)
- **CLI Reference**: `CLI_REFERENCE.md` (All commands and options)
- **Framework Architecture**: `ANALYSIS_FRAMEWORK.md` (Technical details)
- **Template Guide**: `prompts/TEMPLATE_GUIDE.md` (Custom prompt templates)

## ✅ Verify Installation

```bash
# Run test suite
python test_analysis_framework.py

# Should output: ✓ All tests passed!
```

## 🆘 Common Issues

**"Module not found: statqa_analysis"**
→ Run from StatQA root directory

**"Dataset 'X' not found"**
→ Import it first: `python -m statqa_analysis dataset import --dataset-id X --from-json <path>`

**"No 'model_answer' column found"**
→ Use files from `Model Answer/Origin Answer/` or `AnalysisOutput/runs/<run-id>/raw/`

**"No 'serial_number' column found"**
→ Use `--dataset-id` flag to enable metadata merge from StatDatasets

**Plots not generating**
→ Install: `pip install matplotlib seaborn`

**Azure OpenAI credentials not found**
→ Set environment variables: `GPT4_ENDPOINT`, `GPT4_API_KEY`, `API_VERSION`

## 💡 Next Steps

1. ✅ Run analysis on your model outputs
2. 📊 Review results in `AnalysisOutput/`
3. 🔍 Compare multiple models with cohort analysis
4. 🎯 Add custom analyses (see `ANALYSIS_FRAMEWORK.md`)

---

**Need Help?** Check `ANALYSIS_FRAMEWORK.md` for detailed documentation.

