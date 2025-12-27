# StatQA Analysis Framework Documentation

## Overview

The StatQA Analysis Framework is a unified, extensible system for analyzing LLM outputs on the StatQA benchmark. It replaces the fragmented legacy scripts with a clean, plugin-based architecture.

## Quick Start

### Single Model Analysis

```python
from statqa_analysis import ModelOutputAnalyzer

# Analyze one model output
analyzer = ModelOutputAnalyzer(
    input_csv="Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv",
    output_dir="AnalysisOutput",
    run_id="gpt-3.5-turbo_zero-shot"
)
context = analyzer.run_all()

# Access results
print(f"Overall accuracy: {context.get_result('overall_accuracy'):.4f}")
print(f"Processed CSV: {context.get_artifact('processed_csv')}")
```

### Cohort Analysis

```python
from statqa_analysis import CohortAnalyzer

# Analyze multiple runs together
cohort = CohortAnalyzer(
    input_dir="AnalysisOutput/runs",
    output_dir="AnalysisOutput",
    cohort_name="gpt-experiments"
)
context = cohort.run_all()

# Summary tables and radar charts are automatically generated
```

### Command-Line Usage

```bash
# Single run
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
    --out "AnalysisOutput" \
    --run-id "gpt-3.5-turbo_zero-shot"

# Cohort
python -m statqa_analysis cohort \
    --input-dir "AnalysisOutput/runs" \
    --out "AnalysisOutput" \
    --cohort-name "my-cohort"

# With options
python -m statqa_analysis run \
    --input model.csv \
    --out results \
    --no-plots \
    --method-metric jaccard \
    --plot-format png
```

## Architecture

### Core Components

```
statqa_analysis/
├── __init__.py          # Package exports
├── config.py            # AnalysisConfig, AnalysisContext
├── io.py                # Safe I/O utilities
├── pipeline.py          # BaseAnalysis, AnalysisPipeline
├── analyzer.py          # ModelOutputAnalyzer, CohortAnalyzer
├── cli.py               # Command-line interface
└── analyses/            # Pluggable analysis modules
    ├── ground_truth.py
    ├── answer_extraction.py
    ├── compare_count.py
    ├── scoring.py
    ├── task_performance.py
    ├── confusion_matrix.py
    ├── error_types.py
    ├── summary_tables.py
    └── radar_charts.py
```

### Analysis Pipeline

The framework uses a dependency-based pipeline:

```
1. GroundTruthDerivation    → derives ground_truth from results/relevant_column
2. AnswerExtraction         → extracts JSON from model_answer
3. CompareAndCount          → compares with ground truth
4. Scoring                  → calculates accuracy scores
5. TaskPerformance          → per-task performance tables
6. ConfusionMatrixAnalysis  → task confusion matrix
7. ErrorTypeAnalysis        → error type breakdown
```

Each analysis declares:
- **name**: unique identifier
- **requires**: list of dependencies (artifacts/results)
- **produces**: list of outputs
- **run(context)**: execution logic

## Output Structure

```
AnalysisOutput/
├── runs/
│   └── <run_id>/
│       ├── artifacts/
│       │   └── processed.csv              # Enriched data with all scores
│       ├── tables/
│       │   ├── task_performance_methods.csv
│       │   ├── task_performance_columns.csv
│       │   ├── task_performance_overall.csv
│       │   └── error_analysis_summary.csv
│       └── plots/
│           └── confusion_matrix.pdf
└── cohorts/
    └── <cohort_name>/
        ├── tables/
        │   ├── methods_selection_summary_performance.csv
        │   ├── columns_selection_summary_performance.csv
        │   └── overall_selection_summary_performance.csv
        └── plots/
            ├── radar_methods.pdf
            ├── radar_columns.pdf
            └── radar_overall.pdf
```

## Configuration

### AnalysisConfig

```python
from statqa_analysis import AnalysisConfig
from pathlib import Path

config = AnalysisConfig(
    input_csv=Path("model_output.csv"),
    output_dir=Path("AnalysisOutput"),
    run_id="my_run",
    
    # Control flags
    enable_plots=True,
    enable_confusion_matrix=True,
    enable_error_analysis=True,
    enable_radar_charts=True,
    
    # Scoring
    method_metric='acc',  # or 'jaccard'
    
    # Plot settings
    plot_dpi=300,
    plot_format='pdf',  # or 'png', 'svg'
    
    # Custom parameters
    custom_params={'my_param': 'value'}
)
```

### AnalysisContext

The context object carries state through the pipeline:

```python
# Access DataFrames
df = context.df              # Current DataFrame
df_processed = context.df_processed

# Access artifacts (file paths)
processed_csv = context.get_artifact("processed_csv")
confusion_matrix = context.get_artifact("confusion_matrix_plot")

# Access results (computed values)
overall_acc = context.get_result("overall_accuracy")
methods_acc = context.get_result("methods_accuracy")

# All artifacts
for name, path in context.artifacts.items():
    print(f"{name}: {path}")
```

## Extending the Framework

### Adding a New Analysis

1. Create a new module in `statqa_analysis/analyses/`:

```python
# statqa_analysis/analyses/my_analysis.py
from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import save_csv
import pandas as pd

class MyCustomAnalysis(BaseAnalysis):
    @property
    def name(self) -> str:
        return "my_custom_analysis"
    
    @property
    def requires(self) -> list:
        # Depends on scoring being completed
        return ["df_with_scores"]
    
    @property
    def produces(self) -> list:
        return ["custom_metric", "custom_table"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        df = context.df
        
        # Your analysis logic
        custom_metric = df['selection_overall'].mean()
        
        # Create output
        output_dir = context.config.get_run_output_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = pd.DataFrame([{
            'metric': 'custom',
            'value': custom_metric
        }])
        
        output_path = output_dir / "custom_analysis.csv"
        save_csv(summary, output_path)
        
        # Register outputs
        context.add_result("custom_metric", custom_metric)
        context.add_artifact("custom_table", output_path)
        
        print(f"[+] Custom analysis: metric = {custom_metric:.4f}")
        
        return context
```

2. Register it in the pipeline:

```python
# In statqa_analysis/analyzer.py, add to ModelOutputAnalyzer.run_all():
from .analyses import MyCustomAnalysis

analyses = [
    GroundTruthDerivation(),
    AnswerExtraction(),
    CompareAndCount(),
    Scoring(),
    TaskPerformance(),
    ConfusionMatrixAnalysis(),
    ErrorTypeAnalysis(),
    MyCustomAnalysis(),  # Add here
]
```

3. Use it:

```python
analyzer = ModelOutputAnalyzer(...)
context = analyzer.run_all()
print(context.get_result("custom_metric"))
```

### Adding Configuration Parameters

```python
# In config.py, add to AnalysisConfig:
@dataclass
class AnalysisConfig:
    # ... existing fields ...
    my_new_param: str = "default_value"

# Access in your analysis:
class MyAnalysis(BaseAnalysis):
    def run(self, context: AnalysisContext) -> AnalysisContext:
        param_value = context.config.my_new_param
        # ... use param_value ...
```

## Comparison with Legacy Approach

| Aspect | Legacy | New Framework |
|--------|--------|---------------|
| **Entry point** | Multiple scripts | Single API/CLI |
| **Configuration** | Hardcoded paths | Configurable |
| **Output structure** | Fragmented (`Model Answer/`, `Chart/`) | Unified (`AnalysisOutput/`) |
| **Extensibility** | Edit multiple files | Add one module |
| **Dependencies** | Implicit | Explicit (requires/produces) |
| **Error handling** | Inconsistent | Centralized |
| **Path handling** | `path.py` globals | Config-based |
| **Safety** | Uses `eval()` | Safe parsing only |

## Migration Guide

### From Legacy Scripts

**Old approach:**
```bash
sh Script/answer_analysis.sh
```

**New approach:**
```bash
# Single run
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
    --out "AnalysisOutput"

# Cohort
python -m statqa_analysis cohort \
    --input-dir "AnalysisOutput/runs" \
    --out "AnalysisOutput"
```

### From Legacy Python Code

**Old approach:**
```python
import analyze_model_answer
analyze_model_answer.model_answer_integrate_analysis('gpt-3.5-turbo_zero-shot')
```

**New approach:**
```python
from statqa_analysis import ModelOutputAnalyzer

analyzer = ModelOutputAnalyzer(
    input_csv="Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv",
    run_id="gpt-3.5-turbo_zero-shot"
)
analyzer.run_all()
```

## Troubleshooting

### Issue: "Module not found: statqa_analysis"

Ensure you're running from the StatQA root directory:
```bash
cd /path/to/StatQA
python -m statqa_analysis run --input ...
```

### Issue: "No 'model_answer' column found"

The input CSV must be from `Model Answer/Origin Answer/`, not `Processed Answer/`.

### Issue: "Cannot make progress in pipeline"

An analysis has unsatisfied dependencies. Check the error message for which inputs are missing.

### Issue: Plots not generating

Check that matplotlib is installed:
```bash
pip install matplotlib seaborn
```

Or disable plots:
```bash
python -m statqa_analysis run --input ... --no-plots
```

## Best Practices

1. **Use descriptive run IDs**: `gpt-4_zero-shot` not `run1`
2. **Keep cohorts focused**: Group related experiments together
3. **Check artifacts**: Verify outputs in `AnalysisOutput/` after each run
4. **Extend via plugins**: Don't modify core framework files
5. **Use config objects**: Pass `AnalysisConfig` for complex setups
6. **Version control outputs**: Add `AnalysisOutput/` to `.gitignore` for large runs

## Examples

### Batch Processing Multiple Models

```python
from statqa_analysis import ModelOutputAnalyzer
from pathlib import Path

input_dir = Path("Model Answer/Origin Answer")
models = [
    "gpt-3.5-turbo_zero-shot",
    "gpt-3.5-turbo_one-shot",
    "gpt-4_zero-shot",
]

for model in models:
    input_csv = input_dir / f"{model}.csv"
    if input_csv.exists():
        analyzer = ModelOutputAnalyzer(
            input_csv=str(input_csv),
            run_id=model
        )
        analyzer.run_all()
```

### Custom Scoring Metric

```python
from statqa_analysis import ModelOutputAnalyzer, AnalysisConfig

config = AnalysisConfig(
    method_metric='jaccard',  # Use Jaccard index instead of accuracy
    plot_format='png'
)

analyzer = ModelOutputAnalyzer(
    input_csv="model.csv",
    config=config
)
analyzer.run_all()
```

### Programmatic Access to Results

```python
from statqa_analysis import ModelOutputAnalyzer
import pandas as pd

analyzer = ModelOutputAnalyzer(input_csv="model.csv")
context = analyzer.run_all()

# Get processed data
df = context.df
print(df[['task', 'methods_score', 'columns_score', 'selection_overall']].head())

# Get task performance
task_perf_path = context.get_artifact("task_performance_overall")
task_perf = pd.read_csv(task_perf_path)
print(task_perf)

# Get error analysis
error_path = context.get_artifact("error_analysis_summary")
errors = pd.read_csv(error_path)
print(errors)
```

## Support

For issues or questions:
1. Check this documentation
2. Review `README.md` analysis section
3. Examine example outputs in `AnalysisOutput/`
4. Open an issue on the StatQA repository

## License

Same as StatQA project.

