# StatQA Analysis Framework - Unified Workflow Guide

This guide covers the complete unified workflow from dataset management to model inference and analysis.

## 🚀 Complete Workflow (New Unified Approach)

The new unified framework supports the full pipeline:
1. **Dataset Management**: Import and version datasets
2. **Prompt Generation**: Build minimal prompt CSVs
3. **Model Inference**: Run Azure OpenAI on prompts
4. **Analysis**: Compute metrics and visualizations

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Azure OpenAI credentials (create .env file or export)
export GPT4_ENDPOINT="your-azure-endpoint"
export GPT4_API_KEY="your-api-key"
export GPT5_ENDPOINT="your-gpt5-endpoint"  # if using GPT-5
export GPT5_API_KEY="your-gpt5-api-key"
export API_VERSION="2024-02-15-preview"
```

## Step-by-Step Workflow

### Step 1: Import a Dataset

Import a dataset from JSON into `StatDatasets/`:

```bash
# Import mini-StatQA dataset
python -m statqa_analysis dataset import \
  --dataset-id mini-statqa \
  --from-json "Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json"

# Import full StatQA (will synthesize serial_number if missing)
python -m statqa_analysis dataset import \
  --dataset-id statqa-full \
  --from-json "StatQA/StatQA.json"

# Force reimport if dataset already exists
python -m statqa_analysis dataset import \
  --dataset-id mini-statqa \
  --from-json "path/to/data.json" \
  --force
```

**What this creates:**
- `StatDatasets/raw/<dataset-id>/data.json` (canonical dataset with `serial_number`)
- `StatDatasets/raw/<dataset-id>/dataset_manifest.json` (import metadata)
- `StatDatasets/registry.json` (global dataset registry)

### Step 2: Build Prompts

Generate minimal prompt CSVs for different prompting strategies:

```bash
# Build zero-shot prompts
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot

# Build one-shot with seed for reproducibility
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick one-shot \
  --seed 42

# Build all strategies (bash loop)
for trick in zero-shot one-shot two-shot zero-shot-CoT one-shot-CoT stats-prompt; do
  python -m statqa_analysis prompts build \
    --dataset-id mini-statqa \
    --prompt-version v1 \
    --trick $trick \
    --seed 42
done
```

**Available tricks:**
- `zero-shot`: Direct question answering
- `one-shot`: Single example demonstration
- `two-shot`: Two example demonstrations
- `zero-shot-CoT`: Chain-of-thought without examples
- `one-shot-CoT`: Chain-of-thought with example
- `stats-prompt`: Enhanced statistical domain knowledge

**What this creates:**
- `StatDatasets/prompts/<dataset-id>/<version>/<trick>.csv` (minimal CSV: `serial_number`, `prompt`)
- `StatDatasets/prompts/<dataset-id>/<version>/prompt_manifest_<trick>.json` (generation metadata)

### Step 3: Run Model Inference

Run Azure OpenAI inference on the prompts:

```bash
# Run GPT-4 with zero-shot
python -m statqa_analysis model run \
  --model gpt-4 \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot

# Run GPT-3.5-turbo with custom settings
python -m statqa_analysis model run \
  --model gpt-3.5-turbo \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick one-shot \
  --run-id gpt35_oneshot_exp1 \
  --batch-size 10 \
  --max-retries 3

# Run GPT-4o
python -m statqa_analysis model run \
  --model gpt-4o \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot-CoT
```

**What this creates:**
- `AnalysisOutput/runs/<run-id>/raw/model_output.csv` (minimal output: `serial_number`, `model_answer`, token usage)
- `AnalysisOutput/runs/<run-id>/inputs/prompt_source.csv` (copy of prompts for provenance)
- `AnalysisOutput/runs/<run-id>/inputs/run_manifest.json` (run metadata)

**Key features:**
- **Resumable**: If interrupted, re-run the same command to continue from last checkpoint
- **Auto run-id**: Defaults to `<model>_<dataset-id>_<version>_<trick>` if not specified
- **Token tracking**: Records prompt/completion/cached tokens

### Step 4: Analyze Model Outputs

Analyze the model outputs to compute accuracy, confusion matrices, error analysis:

```bash
# Analyze a run (must specify dataset-id for metadata merge)
python -m statqa_analysis run \
  --input AnalysisOutput/runs/gpt-4_mini-statqa_v1_zero-shot/raw/model_output.csv \
  --dataset-id mini-statqa \
  --out AnalysisOutput

# Analyze with custom options
python -m statqa_analysis run \
  --input AnalysisOutput/runs/gpt-3.5-turbo_mini-statqa_v1_one-shot/raw/model_output.csv \
  --dataset-id mini-statqa \
  --run-id gpt35_oneshot_custom \
  --method-metric jaccard \
  --plot-format png
```

**Important:** The `--dataset-id` flag tells the analyzer where to find metadata in `StatDatasets/raw/<dataset-id>/data.json` for merging with the minimal model output CSV.

**What this creates:**
- `AnalysisOutput/runs/<run-id>/tables/` (performance tables by task, methods, columns)
- `AnalysisOutput/runs/<run-id>/plots/` (confusion matrices, error analysis, radar charts)
- `AnalysisOutput/runs/<run-id>/artifacts/processed.csv` (fully processed data with all columns)

### Step 5: Cohort Analysis

Compare multiple runs side-by-side:

```bash
# Analyze a cohort of runs
python -m statqa_analysis cohort \
  --input-dir AnalysisOutput/runs \
  --cohort-name gpt_models_comparison \
  --out AnalysisOutput

# Custom cohort with PNG outputs
python -m statqa_analysis cohort \
  --input-dir AnalysisOutput/runs \
  --cohort-name my_experiments \
  --plot-format png \
  --out AnalysisOutput
```

**What this creates:**
- `AnalysisOutput/cohorts/<cohort-name>/summary_methods.csv` (comparative table)
- `AnalysisOutput/cohorts/<cohort-name>/summary_columns.csv`
- `AnalysisOutput/cohorts/<cohort-name>/summary_overall.csv`
- `AnalysisOutput/cohorts/<cohort-name>/radar_chart.pdf` (visual comparison)

## 📂 Directory Structure

After running the complete pipeline:

```
StatDatasets/
├── registry.json                          # Global dataset registry
├── raw/
│   └── mini-statqa/
│       ├── data.json                      # Canonical dataset (with serial_number)
│       └── dataset_manifest.json          # Import metadata
└── prompts/
    └── mini-statqa/
        └── v1/                            # Prompt version
            ├── zero-shot.csv              # Minimal: serial_number, prompt
            ├── one-shot.csv
            ├── two-shot.csv
            ├── zero-shot-CoT.csv
            ├── one-shot-CoT.csv
            ├── stats-prompt.csv
            ├── prompt_manifest_zero-shot.json
            └── ...

AnalysisOutput/
└── runs/
    └── gpt-4_mini-statqa_v1_zero-shot/    # Auto-generated run-id
        ├── inputs/
        │   ├── prompt_source.csv          # Prompt provenance
        │   └── run_manifest.json          # Run metadata
        ├── raw/
        │   └── model_output.csv           # Minimal: serial_number, model_answer, tokens
        ├── artifacts/
        │   └── processed.csv              # Full analysis data
        ├── tables/
        │   ├── task_performance_methods.csv
        │   ├── task_performance_columns.csv
        │   ├── task_performance_overall.csv
        │   └── error_analysis_summary.csv
        └── plots/
            ├── confusion_matrix_methods.pdf
            ├── error_analysis_bar.pdf
            └── radar_chart.pdf
```

## 🔑 Key Schema Contracts

### Prompt CSV (minimal)
Columns:
- `serial_number`: Integer linking to raw dataset
- `prompt`: Full prompt text

### Model Output CSV (input to analysis)
Columns:
- `serial_number`: Integer linking to raw dataset
- `model_answer`: Raw model response
- Optional: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_cache_hit_tokens`

### Analysis Flow
1. Load model output CSV (`serial_number` + `model_answer`)
2. **Merge metadata** from `StatDatasets/raw/<dataset-id>/data.json` using `serial_number`
3. Extract JSON answers from `model_answer`
4. Compare with `ground_truth`
5. Compute scores and generate visualizations

## 🔄 Complete Example

```bash
# Full pipeline: import → prompts → inference → analysis

# 1. Import dataset
python -m statqa_analysis dataset import \
  --dataset-id mini-statqa \
  --from-json "Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json"

# 2. Build prompts
python -m statqa_analysis prompts build \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot \
  --seed 42

# 3. Run inference
python -m statqa_analysis model run \
  --model gpt-4 \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot

# 4. Analyze results
python -m statqa_analysis run \
  --input AnalysisOutput/runs/gpt-4_mini-statqa_v1_zero-shot/raw/model_output.csv \
  --dataset-id mini-statqa \
  --out AnalysisOutput

# 5. (Optional) Cohort analysis after running multiple models
python -m statqa_analysis cohort \
  --input-dir AnalysisOutput/runs \
  --out AnalysisOutput
```

## ⚙️ Advanced Options

### Custom Run IDs
```bash
python -m statqa_analysis model run \
  --model gpt-4 \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot \
  --run-id my_custom_experiment_name
```

### Resumable Inference
If inference is interrupted, simply re-run the same command:
```bash
# Will resume from last checkpoint in model_output.csv
python -m statqa_analysis model run \
  --model gpt-4 \
  --dataset-id mini-statqa \
  --prompt-version v1 \
  --trick zero-shot
```

### Analysis with Custom Metrics
```bash
# Use Jaccard index for methods scoring
python -m statqa_analysis run \
  --input path/to/model_output.csv \
  --dataset-id mini-statqa \
  --method-metric jaccard

# Disable specific analyses
python -m statqa_analysis run \
  --input path/to/model_output.csv \
  --dataset-id mini-statqa \
  --no-plots \
  --no-confusion-matrix
```

## 🔗 Backward Compatibility

### Legacy CSV Files (with full metadata)
If you have existing CSVs with all metadata columns already present:
```bash
# Analysis still works without --dataset-id
python -m statqa_analysis run \
  --input "Model Answer/Origin Answer/gpt-3.5-turbo_zero-shot.csv" \
  --out AnalysisOutput
```

### Legacy Scripts
Old scripts still work but are deprecated:
```bash
sh Script/prompt_organization.sh      # ⚠️ Legacy
sh Script/gpt_exp.sh                  # ⚠️ Legacy
sh Script/answer_analysis.sh          # ⚠️ Legacy
```

Use the new unified framework for better reproducibility and organization.

## 🆘 Common Issues

**"Dataset 'X' not found"**
→ Import it first with `dataset import`

**"Prompt CSV not found"**
→ Build prompts first with `prompts build`

**"Azure OpenAI credentials not found"**
→ Set environment variables (GPT4_ENDPOINT, GPT4_API_KEY, etc.)

**"No 'serial_number' column found"**
→ Ensure you're using the new workflow or pass `--dataset-id` for metadata merge

**Metadata merge fails**
→ Check that `--dataset-id` matches an imported dataset in `StatDatasets/raw/`

## 📚 Further Reading

- **Framework Architecture**: `ANALYSIS_FRAMEWORK.md`
- **Main README**: `../README.md`

## 📚 Related Documentation

- **Documentation Index**: `README.md` (navigation guide)
- **CLI Reference**: `CLI_REFERENCE.md` (complete command reference)
- **Framework Architecture**: `ANALYSIS_FRAMEWORK.md` (technical details)
- **Template Guide**: `prompts/TEMPLATE_GUIDE.md` (custom templates)
- **Quick Start**: `QUICKSTART_ANALYSIS.md` (2-minute guide)

---

**Need Help?** Check the documentation index in `README.md` or open an issue on GitHub.

