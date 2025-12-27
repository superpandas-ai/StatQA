# Analysis Framework - Radar Charts & Error Bar Charts

## Summary

Added radar charts and error analysis bar charts to the analysis framework. The framework now generates:

### Single-Run Outputs

1. **Confusion Matrix** ✓ (already existed)
2. **Error Analysis Bar Chart** ✓ (NEW)
3. **Radar Chart** ✓ (NEW - shows performance across tasks)

### Cohort Outputs

1. **Radar Charts** ✓ (for comparing multiple models)
2. **Error Analysis Bar Charts** ✓ (NEW - for comparing error types across models)

## New Features

### 1. Single-Run Radar Chart

**Module**: `statqa_analysis/analyses/single_run_radar.py`

**What it does**:
- Creates a radar chart showing performance across all 5 task types
- Uses task abbreviations (CA, DCT, CTT, DS, VT)
- Shows score_rate for each task
- Saves as `plots/radar_chart.pdf`

**Example**:
```
AnalysisOutput/runs/<run_id>/plots/radar_chart.pdf
```

### 2. Error Analysis Bar Chart (Single Run)

**Module**: `statqa_analysis/analyses/error_types.py` (extended)

**What it does**:
- Creates a stacked bar chart showing error type breakdown
- Shows 8 error categories:
  - Invalid Answer
  - Column Selection Error (CSE)
  - Statistical Task Confusion (STC)
  - Applicability Error (AE)
  - Mixed Errors (CSE+STC)
  - Mixed Errors (CSE+AE)
  - Mixed Errors (STC+AE)
  - Mixed Errors (CSE+STC+AE)
- Saves as `plots/error_analysis_bar_chart.pdf`

**Example**:
```
AnalysisOutput/runs/<run_id>/plots/error_analysis_bar_chart.pdf
```

### 3. Cohort Error Analysis Bar Chart

**Module**: `statqa_analysis/analyzer.py` (CohortAnalyzer)

**What it does**:
- Collects error analysis summaries from all runs in a cohort
- Creates a combined stacked bar chart comparing error types across models
- Saves combined summary CSV and bar chart PDF

**Example**:
```
AnalysisOutput/cohorts/<cohort_name>/
├── tables/error_analysis_summary.csv
└── plots/error_analysis_bar_chart.pdf
```

## Usage

### Single Run (Automatic)

The radar chart and error bar chart are generated automatically for each run:

```bash
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-3.5-turbo_one-shot.csv" \
    --out "AnalysisOutput" \
    --run-id "gpt-3.5-turbo_one-shot"
```

**Outputs**:
- `plots/confusion_matrix.pdf` ✓
- `plots/error_analysis_bar_chart.pdf` ✓ (NEW)
- `plots/radar_chart.pdf` ✓ (NEW)

### Cohort Analysis

Error bar charts are generated automatically for cohorts:

```bash
python -m statqa_analysis cohort \
    --input-dir "AnalysisOutput/runs" \
    --out "AnalysisOutput" \
    --cohort-name "my-experiments"
```

**Outputs**:
- `plots/radar_methods.pdf` ✓
- `plots/radar_columns.pdf` ✓
- `plots/radar_overall.pdf` ✓
- `plots/error_analysis_bar_chart.pdf` ✓ (NEW)
- `tables/error_analysis_summary.csv` ✓ (NEW)

## Configuration

All plots respect the configuration flags:

```python
config = AnalysisConfig(
    enable_plots=True,              # Enable/disable all plots
    enable_confusion_matrix=True,    # Confusion matrix
    enable_error_analysis=True,     # Error analysis (includes bar chart)
    plot_dpi=300,                    # Plot resolution
    plot_format='pdf'                # Format: pdf, png, svg
)
```

## File Structure

### Single Run

```
AnalysisOutput/runs/<run_id>/
├── artifacts/
│   └── processed.csv
├── tables/
│   ├── task_performance_methods.csv
│   ├── task_performance_columns.csv
│   ├── task_performance_overall.csv
│   └── error_analysis_summary.csv
└── plots/
    ├── confusion_matrix.pdf          ✓
    ├── error_analysis_bar_chart.pdf   ✓ NEW
    └── radar_chart.pdf                ✓ NEW
```

### Cohort

```
AnalysisOutput/cohorts/<cohort_name>/
├── tables/
│   ├── methods_selection_summary_performance.csv
│   ├── columns_selection_summary_performance.csv
│   ├── overall_selection_summary_performance.csv
│   └── error_analysis_summary.csv     ✓ NEW
└── plots/
    ├── radar_methods.pdf
    ├── radar_columns.pdf
    ├── radar_overall.pdf
    └── error_analysis_bar_chart.pdf   ✓ NEW
```

## Implementation Details

### Single-Run Radar Chart

- Uses task performance data from `task_performance_overall.csv`
- Maps full task names to abbreviations via `mappings.task_abbreviations`
- Creates polar plot with 5 axes (one per task)
- Shows score_rate (0-1) for each task
- Includes value labels on each axis

### Error Analysis Bar Chart

- Uses error analysis summary data
- Creates stacked bar chart with 8 error categories
- Color-coded for easy identification
- Shows value labels for significant errors (>0.02)
- Automatically formats model names for display

### Cohort Error Analysis

- Scans all runs in the cohort directory
- Collects `error_analysis_summary.csv` from each run
- Combines into single DataFrame
- Creates multi-model comparison bar chart
- Saves combined summary CSV

## Testing

All features tested and working:

```bash
$ python test_analysis_framework.py
✓ All tests passed! (5/5)
```

Tested with:
- ✅ Single-run analysis generates all 3 plots
- ✅ Cohort analysis generates error bar charts
- ✅ All plots saved in correct format
- ✅ No breaking changes to existing functionality

## Backward Compatibility

- All existing functionality preserved
- New plots are optional (controlled by `enable_plots` flag)
- No changes to existing output files
- Can disable individual plot types via config

## Examples

### Example 1: Single Run

```bash
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/Extracted Answer/General Analysis/gpt-3.5-turbo_one-shot.csv" \
    --out "AnalysisOutput"
```

**Generated**:
- `confusion_matrix.pdf`
- `error_analysis_bar_chart.pdf` ← NEW
- `radar_chart.pdf` ← NEW

### Example 2: Cohort

```bash
python -m statqa_analysis cohort \
    --input-dir "AnalysisOutput/runs" \
    --out "AnalysisOutput" \
    --cohort-name "gpt-experiments"
```

**Generated**:
- `radar_methods.pdf`
- `radar_columns.pdf`
- `radar_overall.pdf`
- `error_analysis_bar_chart.pdf` ← NEW
- `error_analysis_summary.csv` ← NEW

## Status

✅ **COMPLETE**: All requested features implemented and tested
✅ **WORKING**: Radar charts and error bar charts generated successfully
✅ **READY**: Ready for production use

