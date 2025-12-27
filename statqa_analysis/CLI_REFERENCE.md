# StatQA Model Output Framework - CLI Reference

## Quick Reference

### Dataset Management

```bash
# Import a dataset
python -m statqa_analysis dataset import \
  --dataset-id <id> \
  --from-json <path>

# Options:
#   --force                     Overwrite existing dataset
#   --no-ensure-serial-number   Don't synthesize serial_number
```

### Prompt Generation

```bash
# Build prompts
python -m statqa_analysis prompts build \
  --dataset-id <id> \
  --prompt-version <version> \
  --trick <strategy>

# Options:
#   --seed <int>               Random seed for reproducibility
#   --template <path>          Custom prompt template JSON file
#   --manual-template <path>   F-string template file for manual trick
# Tricks: zero-shot, one-shot, two-shot, zero-shot-CoT, one-shot-CoT, stats-prompt, manual
```

### Model Inference

```bash
# Run model inference
python -m statqa_analysis model run \
  --model <deployment> \
  --dataset-id <id> \
  --prompt-version <version> \
  --trick <strategy>

# Options:
#   --run-id <id>              Custom run identifier
#   --max-retries <n>          Max retries per prompt (default: 2)
#   --batch-size <n>           Save frequency (default: 5)
```

### Analysis

```bash
# Analyze model output
python -m statqa_analysis run \
  --input <csv> \
  --dataset-id <id>

# Options:
#   --run-id <id>              Custom run identifier
#   --out <dir>                Output directory (default: AnalysisOutput)
#   --method-metric <metric>   Use 'acc' or 'jaccard' (default: acc)
#   --plot-format <fmt>        'pdf', 'png', or 'svg' (default: pdf)
#   --no-plots                 Disable plots
#   --no-confusion-matrix      Disable confusion matrices
#   --no-error-analysis        Disable error analysis
```

### Cohort Analysis

```bash
# Compare multiple runs
python -m statqa_analysis cohort \
  --input-dir <dir> \
  --cohort-name <name>

# Options:
#   --out <dir>                Output directory
#   --plot-format <fmt>        Plot format
#   --no-radar-charts          Disable radar charts
#   --filter <pattern>          Filter run directories by name pattern
#                               (e.g., "stats-prompt" to only include runs
#                                with "stats-prompt" in the name)
```

## Complete Workflow Example

```bash
# 1. Import dataset
python -m statqa_analysis dataset import \
  --dataset-id mini-statqa \
  --from-json "Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json"

# 2. Build prompts for all strategies
for trick in zero-shot one-shot two-shot zero-shot-CoT one-shot-CoT stats-prompt; do
  python -m statqa_analysis prompts build \
    --dataset-id mini-statqa \
    --prompt-version v1 \
    --trick $trick \
    --seed 42
done

# Or use custom template
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v2 \
  --trick zero-shot \
  --template my_template.json

# Or use manual f-string template
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v2 \
  --trick manual \
  --manual-template my_template.txt

# 3. Run inference for multiple models
for model in gpt-3.5-turbo gpt-4 gpt-4o; do
  python -m statqa_analysis model run \
    --model $model \
    --dataset-id mini-statqa \
    --prompt-version v1 \
    --trick zero-shot
done

# 4. Analyze each run
# Option A: Using glob (requires shopt -s nullglob to handle no matches)
shopt -s nullglob
for run in AnalysisOutput/runs/*/raw/model_output.csv; do
  python -m statqa_analysis run \
    --input "$run" \
    --dataset-id mini-statqa
done
shopt -u nullglob

# Option B: Using find (more robust, handles missing files gracefully)
find AnalysisOutput/runs -name "model_output.csv" -path "*/raw/*" | while read -r run; do
  python -m statqa_analysis run \
    --input "$run" \
    --dataset-id mini-statqa
done

# 5. Compare all models
python -m statqa_analysis cohort \
  --input-dir AnalysisOutput/runs \
  --cohort-name gpt_comparison

# Or filter by pattern (e.g., only stats-prompt runs)
python -m statqa_analysis cohort \
  --input-dir AnalysisOutput/runs \
  --cohort-name stats_prompt_only \
  --filter stats-prompt
```

## Environment Setup

Required environment variables for Azure OpenAI:

```bash
export GPT4_ENDPOINT="https://your-resource.openai.azure.com/"
export GPT4_API_KEY="your-api-key"
export GPT5_ENDPOINT="https://your-gpt5-resource.openai.azure.com/"
export GPT5_API_KEY="your-gpt5-api-key"
export API_VERSION="2024-02-15-preview"
```

Or create a `.env` file in the project root.

## Directory Structure

```
StatDatasets/                              # Dataset assets
├── registry.json                          # Global registry
├── raw/<dataset-id>/
│   ├── data.json                          # Canonical data with serial_number
│   └── dataset_manifest.json              # Import metadata
└── prompts/<dataset-id>/<version>/
    ├── <trick>.csv                        # Minimal: serial_number, prompt
    └── prompt_manifest_<trick>.json       # Generation metadata

AnalysisOutput/                            # Run outputs
└── runs/<run-id>/
    ├── inputs/
    │   ├── prompt_source.csv              # Prompt provenance
    │   └── run_manifest.json              # Run metadata
    ├── raw/
    │   └── model_output.csv               # Minimal: serial_number, model_answer, tokens
    ├── artifacts/
    │   └── processed.csv                  # Full analysis data
    ├── tables/                            # Performance tables
    └── plots/                             # Visualizations
```

## Schema Contracts

### Prompt CSV
- `serial_number`: Integer
- `prompt`: String

### Model Output CSV
- `serial_number`: Integer
- `model_answer`: String
- Optional: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_cache_hit_tokens`

### Dataset JSON (StatDatasets/raw/<id>/data.json)
Required fields per entry:
- `serial_number`: Integer (unique)
- `dataset`: String (dataset name)
- `refined_question`: String
- `task`: String
- `difficulty`: String
- `ground_truth`: JSON string
- `results`: JSON string
- `relevant_column`: JSON string

## See Also

- **Unified Workflow Guide**: `UNIFIED_WORKFLOW.md`
- **Framework Architecture**: `ANALYSIS_FRAMEWORK.md`
- **Quick Start**: `QUICKSTART_ANALYSIS.md`

