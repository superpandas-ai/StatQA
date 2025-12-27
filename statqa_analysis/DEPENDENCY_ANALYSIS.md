# Dependency Analysis: statqa_analysis, AnalysisOutput, StatDatasets

## Summary

The `statqa_analysis`, `AnalysisOutput`, and `StatDatasets` directories are **mostly independent** but have **some dependencies** on the rest of the codebase.

## Dependencies Found

### 1. statqa_analysis Dependencies

#### External Dependencies (from repo root):
- **`utils.py`**: Used by multiple analysis modules
  - `utils.extract_json_answer()` - in `analyses/answer_extraction.py`
  - `utils.get_metadata()` - in `prompts/metadata.py`
  - `utils.derive_ground_truth()` - in `analyses/ground_truth.py`
  
- **`mappings.py`**: Used for task abbreviations and method mappings
  - `mappings.task_abbreviations` - in `analyses/radar_charts.py`, `analyses/single_run_radar.py`
  - `mappings.tasks_to_methods` - in `analyses/confusion_matrix.py`, `analyses/error_types.py`

- **`prompt_wording.py`**: Used for prompt templates
  - Imported in `prompts/templates.py` (direct import, not via sys.path)

#### Path Manipulation:
Multiple files use `sys.path.insert()` to access repo root modules:
- `statqa_analysis/analyses/answer_extraction.py`
- `statqa_analysis/analyses/ground_truth.py`
- `statqa_analysis/analyses/confusion_matrix.py`
- `statqa_analysis/analyses/error_types.py`
- `statqa_analysis/analyses/radar_charts.py`
- `statqa_analysis/analyses/single_run_radar.py`
- `statqa_analysis/analyses/summary_tables.py`
- `statqa_analysis/prompts/metadata.py`

### 2. AnalysisOutput Dependencies

#### References from outside statqa_analysis:
- **`Script/model_answer_analysis.sh`**: Uses `AnalysisOutput` as output directory
- **`Script/answer_analysis.sh`**: Mentions `AnalysisOutput` in comments
- **`README.md`**: Documents `AnalysisOutput` structure

**No hardcoded paths** in other Python modules (Construction/, Evaluation/, etc.)

### 3. StatDatasets Dependencies

#### References from outside statqa_analysis:
- **Only referenced within `statqa_analysis/`** itself
- No external Python code references it
- Created and managed entirely by `statqa_analysis` modules

## Independence Assessment

### ✅ Mostly Independent:
- **StatDatasets**: Fully independent - only used by statqa_analysis
- **AnalysisOutput**: Mostly independent - only referenced in scripts/docs, not in core Python code

### ⚠️ Partially Dependent:
- **statqa_analysis**: Has dependencies on:
  - `utils.py` (extract_json_answer, get_metadata, derive_ground_truth)
  - `mappings.py` (task_abbreviations, tasks_to_methods)
  - `prompt_wording.py` (prompt constants)

## Recommendations

### To Make Fully Independent:

1. **Copy required functions** from `utils.py` into `statqa_analysis`:
   - `extract_json_answer()` → `statqa_analysis/utils_extract.py`
   - `get_metadata()` → Already adapted in `prompts/metadata.py` (but still uses utils)
   - `derive_ground_truth()` → Already in `analyses/ground_truth.py` (but uses utils)

2. **Copy required mappings** from `mappings.py`:
   - `task_abbreviations` → `statqa_analysis/mappings.py`
   - `tasks_to_methods` → `statqa_analysis/mappings.py`

3. **Remove sys.path manipulation**:
   - Replace direct imports with local copies
   - Or make `prompt_wording.py` a proper package dependency

### Current State:
- **Can be used standalone** if `utils.py`, `mappings.py`, and `prompt_wording.py` are available
- **Not fully portable** without these dependencies
- **AnalysisOutput and StatDatasets** are fully independent directories

## External References Summary

| Component | External References | Dependency Level |
|-----------|-------------------|------------------|
| `statqa_analysis/` | `utils.py`, `mappings.py`, `prompt_wording.py` | Medium |
| `AnalysisOutput/` | Scripts, docs only | Low |
| `StatDatasets/` | None | None |

## Files Using External Dependencies

### utils.py usage:
- `statqa_analysis/analyses/answer_extraction.py` - `utils.extract_json_answer()`
- `statqa_analysis/analyses/ground_truth.py` - `utils.derive_ground_truth()`
- `statqa_analysis/prompts/metadata.py` - `utils.get_metadata()`

### mappings.py usage:
- `statqa_analysis/analyses/radar_charts.py` - `mappings.task_abbreviations`
- `statqa_analysis/analyses/single_run_radar.py` - `mappings.task_abbreviations`
- `statqa_analysis/analyses/confusion_matrix.py` - `mappings.tasks_to_methods`, `mappings.task_abbreviations`
- `statqa_analysis/analyses/error_types.py` - `mappings.tasks_to_methods`

### prompt_wording.py usage:
- `statqa_analysis/prompts/templates.py` - Direct import (assumes in Python path)

---

**Conclusion**: `statqa_analysis` is **functionally independent** but has **code dependencies** on repo root utilities. `AnalysisOutput` and `StatDatasets` are **fully independent** directories.

