# StatQA Analysis Framework - Updates for Serial Number & Metadata Merge

## Summary

Updated the analysis framework to handle two types of input files:
1. **Files with `extracted_answer` already present** (e.g., from "General Analysis" folder)
2. **Files with `model_answer` that need extraction** (standard model outputs)

Additionally, the framework now automatically merges metadata from `mini-StatQA.json` based on `serial_number` to populate fields like `task`, `results`, `relevant_column`, `ground_truth`, etc.

## Changes Made

### 1. New Analysis Module: `metadata_merge.py`

**Purpose**: Merges metadata from `mini-StatQA.json` into the DataFrame based on `serial_number`.

**Features**:
- Loads `Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json`
- Creates lookup dictionary by `serial_number`
- Merges columns: `task`, `results`, `relevant_column`, `ground_truth`, `dataset`, `refined_question`, `difficulty`
- Skips merge if metadata columns already exist
- Handles missing `serial_number` column gracefully

**Location**: `statqa_analysis/analyses/metadata_merge.py`

### 2. Updated `ground_truth.py`

**Changes**:
- Now depends on `df_with_metadata` instead of `raw_data`
- Checks if `ground_truth` is already populated (not just present)
- Only derives ground truth if needed
- Better error messages for missing required columns

### 3. Updated `answer_extraction.py`

**Changes**:
- Checks if `extracted_answer` already exists and is populated
- Skips extraction if `extracted_answer` is already present
- Falls back to extracting from `model_answer` if needed
- Clear error message if neither column is present

### 4. Updated Pipeline

**Changes**:
- Added `MetadataMerge` as the first analysis step
- Pipeline now handles both file types automatically
- Dependency chain: `raw_data` → `metadata_merge` → `ground_truth` → `answer_extraction` → ...

## Usage

### Files with `extracted_answer` (General Analysis folder)

```bash
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/Extracted Answer/General Analysis/gpt-3.5-turbo_one-shot.csv" \
    --out "AnalysisOutput" \
    --run-id "gpt-3.5-turbo_one-shot"
```

**What happens**:
1. Loads CSV with `extracted_answer` and `serial_number`
2. Merges metadata from `mini-StatQA.json` based on `serial_number`
3. Skips extraction (already done)
4. Skips ground truth derivation (from JSON)
5. Proceeds with comparison, scoring, etc.

### Files with `model_answer` (standard outputs)

```bash
python -m statqa_analysis run \
    --input "Model Answer/Origin Answer/gpt-5.2_one-shot.csv" \
    --out "AnalysisOutput" \
    --run-id "gpt-5.2_one-shot"
```

**What happens**:
1. Loads CSV with `model_answer` and `serial_number`
2. Merges metadata from `mini-StatQA.json` based on `serial_number`
3. Extracts JSON from `model_answer` → `extracted_answer`
4. Derives ground truth from `results`/`relevant_column` if needed
5. Proceeds with comparison, scoring, etc.

## Input File Requirements

### Minimum Required Columns

**For files with `extracted_answer`**:
- `extracted_answer` (JSON string)
- `serial_number` (integer, links to mini-StatQA.json)

**For files with `model_answer`**:
- `model_answer` (raw model response)
- `serial_number` (integer, links to mini-StatQA.json)

### Optional Columns (will be merged from JSON if missing)

- `task`
- `results`
- `relevant_column`
- `ground_truth`
- `dataset`
- `refined_question`
- `difficulty`

## Example Output

After running analysis on a file from "General Analysis":

```
[*] Merging metadata from mini-StatQA.json...
[+] Loaded metadata for 1163 entries from JSON
[+] Merged metadata for 8141 cells
[+] All required metadata columns present
[i] ground_truth column already exists and populated, skipping derivation
[i] extracted_answer column already exists and populated, skipping extraction
[*] Comparing answers with ground truth...
[*] Calculating scores...
Methods Accuracy:  0.4454
Columns Accuracy:  0.9235
Overall Accuracy:  0.4076
```

## Verification

The framework automatically detects:
- ✅ Presence of `serial_number` → merges metadata from JSON
- ✅ Presence of `extracted_answer` → skips extraction
- ✅ Presence of `ground_truth` → skips derivation
- ✅ Missing columns → attempts to merge from JSON

## Backward Compatibility

- Files without `serial_number` will still work (metadata merge skipped)
- Files with all metadata already present will work (merge skipped)
- Files with `model_answer` still work as before (extraction happens)

## Testing

All tests pass:
```bash
$ python test_analysis_framework.py
✓ All tests passed! (5/5)
```

Tested with:
- ✅ Files from "General Analysis" folder (extracted_answer + serial_number)
- ✅ Standard model output files (model_answer + serial_number)
- ✅ All analysis steps complete successfully

## Files Modified

1. `statqa_analysis/analyses/metadata_merge.py` (NEW)
2. `statqa_analysis/analyses/ground_truth.py` (UPDATED)
3. `statqa_analysis/analyses/answer_extraction.py` (UPDATED)
4. `statqa_analysis/analyses/__init__.py` (UPDATED)
5. `statqa_analysis/analyzer.py` (UPDATED - added MetadataMerge to pipeline)
6. `statqa_analysis/pipeline.py` (UPDATED - supports initial_resources)

## Notes

- The `mini-StatQA.json` file must be present at:
  `Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json`
- If JSON file is missing, metadata merge is skipped with a warning
- Missing `serial_number` column results in metadata merge being skipped
- All existing functionality remains intact

