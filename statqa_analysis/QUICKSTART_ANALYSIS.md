# StatQA Analysis Framework - Quick Start

## 🚀 Get Started in 2 Minutes

### Option 1: Command Line (Easiest)

```bash
# Analyze a single model output
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
    --out "AnalysisOutput"

# Results will be in: AnalysisOutput/runs/gpt-3.5-turbo_zero-shot/
```

### Option 2: Python API

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

### Option 3: Batch Script

```bash
# Analyze multiple models at once
sh Script/model_answer_analysis.sh
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

- **Framework Guide**: `ANALYSIS_FRAMEWORK.md`
- **Implementation Notes**: `IMPLEMENTATION_NOTES.md`
- **Main README**: `README.md` (Analysis section)

## ✅ Verify Installation

```bash
# Run test suite
python test_analysis_framework.py

# Should output: ✓ All tests passed!
```

## 🆘 Common Issues

**"Module not found: statqa_analysis"**
→ Run from StatQA root directory

**"No 'model_answer' column found"**
→ Use files from `Model Answer/Origin Answer/`, not `Processed Answer/`

**Plots not generating**
→ Install: `pip install matplotlib seaborn`

## 🔧 Legacy Scripts (Deprecated)

Old approach still works but is deprecated:
```bash
sh Script/answer_analysis.sh  # ⚠️  Legacy
```

Use the new framework instead for better organization and extensibility.

## 💡 Next Steps

1. ✅ Run analysis on your model outputs
2. 📊 Review results in `AnalysisOutput/`
3. 🔍 Compare multiple models with cohort analysis
4. 🎯 Add custom analyses (see `ANALYSIS_FRAMEWORK.md`)

---

**Need Help?** Check `ANALYSIS_FRAMEWORK.md` for detailed documentation.

