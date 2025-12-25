# StressQA Dataset Materialization

## Overview

The StressQA benchmark references modified datasets that are created on-the-fly during benchmark generation. These modified datasets include additional columns (like `condition_A`, `condition_B`, `subject_id`, etc.) that are created by modifiers but are not saved in the original dataset files.

This script materializes (saves) all modified datasets so they can be accessed when working with the benchmark.

## Usage

### Basic Usage

```bash
# Materialize datasets for mini-StressQA
python StressTest/materialize_datasets.py

# Or use the shell script
sh Script/materialize_stressqa_datasets.sh
```

### Advanced Usage

```bash
# Specify custom benchmark file and output directory
python StressTest/materialize_datasets.py \
    --benchmark-file "Data/Integrated Dataset/Balanced Benchmark/StressQA.json" \
    --output-dir "Data/External Dataset/Processed/" \
    --random-state 42 \
    --update-benchmark

# Update benchmark with paths (without overwriting original)
python StressTest/materialize_datasets.py \
    --benchmark-file "Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.json" \
    --output-dir "Data/External Dataset/Processed/" \
    --update-benchmark \
    --output-benchmark "Data/Integrated Dataset/Balanced Benchmark/mini-StressQA_with_paths.json"
```

## What It Does

1. **Reads the benchmark JSON** to identify all unique dataset+modifier combinations
2. **Groups entries** by dataset and modifier combination to avoid duplicates
3. **Recreates modified datasets** by:
   - Loading the original dataset
   - Applying the same modifiers with parameters extracted from `design_metadata`
   - Saving the modified dataset with a unique signature
4. **Optionally updates the benchmark** to include `materialized_dataset_path` field

## Output Structure

```
Data/External Dataset/Processed/
├── grunfeld_investment_paired_unpaired_trap_89b40ec4.csv    # Modified dataset with paired_unpaired_trap
├── grunfeld_investment_heteroscedasticity_a1d4b681.csv      # Modified dataset with heteroscedasticity
├── duncan_prestige_heteroscedasticity_0b740797.csv          # Modified dataset with heteroscedasticity
├── grunfeld_investment_d0e5efd4.csv                         # Original dataset (no modifiers)
└── _materialization_mapping.json                             # Mapping from signatures to file paths
```

## Dataset Signatures

Each materialized dataset is named using a signature that includes:
- Original dataset name
- Applied modifier names (if any)
- Short hash for uniqueness (based on dataset, modifiers, and key parameters)

**Naming Format:**
- With modifiers: `{dataset}_{modifier1}_{modifier2}_{hash}.csv`
- Without modifiers: `{dataset}_{hash}.csv`

**Examples:**
- `grunfeld_investment_paired_unpaired_trap_89b40ec4.csv` - paired_unpaired_trap modifier applied
- `grunfeld_investment_heteroscedasticity_a1d4b681.csv` - heteroscedasticity modifier applied
- `duncan_prestige_heteroscedasticity_paired_unpaired_trap_51b99e0e.csv` - multiple modifiers applied
- `grunfeld_investment_d0e5efd4.csv` - no modifiers (original dataset)

## Modified Columns

Depending on the modifiers applied, materialized datasets may include:

### Paired/Unpaired Trap Modifier
- `subject_id`: Subject identifier for pairing
- `condition`: Condition labels (`condition_A`, `condition_B`)

### Heteroscedasticity Modifier
- `group`: Binary grouping variable (`group_A`, `group_B`)
- Modified target variable with unequal variances

### Multiple Endpoints Modifier
- `endpoint_1`, `endpoint_2`, ..., `endpoint_N`: Multiple correlated outcome variables
- `group`: Binary grouping variable

### Sparse Contingency Modifier
- `category_A`, `category_B`: Categorical variables with sparse counts

### Confounding/Simpson's Modifier
- `hospital_size`: Strata variable
- `treatment`: Binary exposure variable
- `outcome`: Binary outcome variable

## Reproducibility

The script uses `random_state=42` by default to ensure reproducibility. However, note that:

1. **Exact match**: If the original benchmark was generated with a different random state, the materialized datasets may have slightly different values, but the structure (columns, relationships) will be the same.

2. **Parameter extraction**: Parameters are extracted from `design_metadata` in the benchmark. If some parameters are missing, defaults are used, which may result in slightly different datasets.

3. **Deterministic structure**: The key columns (like `condition_A`, `condition_B`, `subject_id`) will be present and correctly structured, even if exact values differ.

## Integration with Benchmark

After materialization, you can:

1. **Use the mapping file** to find the correct dataset for each benchmark entry:
```python
import json

with open('Data/External Dataset/Processed/_materialization_mapping.json') as f:
    mapping = json.load(f)

# Find dataset for a specific signature
signature = create_dataset_signature(benchmark_entry)
dataset_path = mapping[signature]
```

2. **Use the updated benchmark** (if `--update-benchmark` was used):
```python
import json

with open('benchmark_with_paths.json') as f:
    benchmark = json.load(f)

for entry in benchmark:
    if 'materialized_dataset_path' in entry:
        df = pd.read_csv(entry['materialized_dataset_path'])
        # Now df has all columns referenced in relevant_column
```

## Troubleshooting

### Error: "Original dataset not found"
- Ensure the original dataset exists in `Data/External Dataset/Origin/` or `Data/Origin Dataset/`
- Check that the dataset name in the benchmark matches the filename (without `.csv`)

### Error: "Error applying modifiers"
- Check that all required parameters are present in `design_metadata`
- Some modifiers may fail if the dataset doesn't meet requirements (e.g., not enough rows, wrong column types)
- Check the error message for specific details

### Materialized datasets don't match expected structure
- Verify that `random_state=42` matches the original benchmark generation
- Check that parameters in `design_metadata` are complete
- Some modifiers use random sampling, so exact values may differ slightly

## Example

```python
import pandas as pd
import json

# Load benchmark
with open('Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.json') as f:
    benchmark = json.load(f)

# Load mapping
with open('Data/External Dataset/Processed/_materialization_mapping.json') as f:
    mapping = json.load(f)

# Process an entry
entry = benchmark[0]
signature = create_dataset_signature(entry)

if signature in mapping:
    # Load the materialized dataset
    df = pd.read_csv(mapping[signature])
    
    # Now df has all columns referenced in entry['relevant_column']
    relevant_cols = json.loads(entry['relevant_column'])
    for col_info in relevant_cols:
        col_name = col_info['column_header']
        if col_name in df.columns:
            print(f"✓ Column {col_name} found")
        else:
            print(f"✗ Column {col_name} missing")
```

## See Also

- `StressTest/benchmark_generator.py`: How the benchmark is generated
- `StressTest/modifiers.py`: How modifiers transform datasets
- `STRESSQA_README.md`: Overall StressQA documentation

