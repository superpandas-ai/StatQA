# StressQA: Enhanced Statistical Analysis Benchmark

## Overview

StressQA is an enhanced version of the StatQA benchmark designed specifically to stress-test **superstat** and other AI-powered statistical analysis systems. It extends StatQA's foundation with:

1. **Harder scenario variants** via composable modifiers
2. **New test families** (group comparisons, regression/ANCOVA, multiple testing)
3. **Structured JSON output contract** with audit trails
4. **Layered evaluation** (selection, applicability, numeric tolerance, decision quality, audit completeness)

## Architecture

```
┌─────────────────┐
│ External Data   │ ← statsmodels/sklearn datasets
│ (materialized)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Base Cases      │ ← TestSpec registry
│ Generation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Modifier        │ ← Paired trap, heteroscedasticity,
│ Pipeline        │   sparse contingency, multiple endpoints,
└────────┬────────┘   confounding/Simpson's
         │
         ▼
┌─────────────────┐
│ Oracle          │ ← Deterministic ground truth
│ Computation     │   with numeric outputs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StressQA.csv/   │ ← Backward compatible + new fields
│ StressQA.json   │
└─────────────────┘
```

## Key Components

### 1. Test Specification Registry (`test_spec_registry.py`)

Defines test specs with:
- Variable roles (DV/IV/covariates/strata/cluster/id)
- Design types (independent/paired/repeated/clustered/stratified)
- Prerequisites (normality, equal variance, min sample size, etc.)
- **Acceptable method sets** (multiple correct answers allowed)
- Expected oracle outputs (statistic, df, p-value, effect size, CI, post-hoc)

**Families supported:**
- Existing: Correlation, Contingency Tables, Distribution Compliance, Variance Tests, Descriptive Stats
- **New:** Group Comparison, Regression/ANCOVA, Multiple Testing

### 2. Scenario Modifiers (`StressTest/modifiers.py`)

Composable transforms that add difficulty axes:

| Modifier | Effect | Tests |
|----------|--------|-------|
| `paired_unpaired_trap` | Creates paired structure; question suggests independent test | Paired vs independent t-test |
| `heteroscedasticity` | Violates equal variance | Student's vs Welch's t-test |
| `sparse_contingency` | Small expected counts | Chi-square vs Fisher's exact |
| `multiple_endpoints` | K correlated outcomes | Requires FDR/Bonferroni correction |
| `confounding_simpsons` | Simpson's paradox | Stratified vs marginal analysis |

### 3. Oracle Computer (`StressTest/oracle_computer.py`)

Computes deterministic ground truth for:
- **Group comparisons:** t-tests (independent, Welch, paired), Mann-Whitney, Wilcoxon, ANOVA, Kruskal-Wallis
- **Regression:** Simple/multiple linear regression, ANCOVA
- **Multiple testing:** Multiple endpoints with BH/FDR or Bonferroni correction

Outputs include:
- Test statistic, df, p-value
- Effect sizes (Cohen's d, eta², rank-biserial, r², etc.)
- Confidence intervals
- Post-hoc recommendations
- Assumption check results

### 4. Enhanced JSON Contract (`prompt_wording.py`)

Systems must return:

```json
{
  "columns": ["col1", "col2"],
  "methods": ["Welch t-test"],
  "applicability": true,
  "checks": {"normality": true, "equal_variance": false},
  "warnings": ["Equal variance violated"],
  "test_result": {"statistic": 2.45, "df": 98, "p_value": 0.016},
  "effect_size": {"value": 0.49, "type": "cohen_d"},
  "ci": {"lower": 0.12, "upper": 1.84, "level": 0.95},
  "post_hoc": null,
  "corrections": null,
  "audit_trail": {
    "prerequisite_checks": "Checked normality...",
    "method_choice_reason": "Selected Welch due to heteroscedasticity",
    "alternatives_rejected": "Student's t-test rejected"
  }
}
```

### 5. Layered Scorer (`StressTest/enhanced_scorer.py`)

Evaluates:

| Layer | Weight | Metrics |
|-------|--------|---------|
| **Column selection** | 20% | Exact match, precision, recall, F1 |
| **Method selection** | 25% | Acceptable (in valid set), exact match |
| **Applicability** | 10% | Correct reject/accept |
| **Numeric results** | 20% | p-value (±0.01), statistic (±5% rel), effect size (±10% rel), CI overlap |
| **Decision quality** | 15% | Post-hoc recommended, correction applied, warnings provided |
| **Audit trail** | 10% | Completeness of prerequisite checks, method rationale, alternatives |

**Breakdown by difficulty_axes** for detailed analysis.

## Dataset Schema

StressQA rows include:

**Existing StatQA columns (backward compatible):**
- `dataset`, `refined_question`, `relevant_column`, `results`, `task`, `difficulty`

**New StressQA columns:**
- `analysis_spec_id` – TestSpec identifier
- `difficulty_axes` – JSON list of applied modifiers
- `acceptable_methods` – JSON list of acceptable method sets
- `oracle` – JSON object with numeric oracle outputs
- `is_applicable` – Boolean applicability flag
- `design_metadata` – JSON with paired_id, strata_col, covariates, etc.
- `oracle_checks` – JSON with assumption check results

## Usage

### Build StressQA Benchmark

```bash
# Step 1: Materialize external datasets (statsmodels/sklearn)
sh Script/external_dataset_materialize.sh

# Step 2: Generate StressQA benchmark
sh Script/stress_benchmark_construction.sh

# Outputs:
#   Data/Integrated Dataset/Balanced Benchmark/StressQA.csv
#   Data/Integrated Dataset/Balanced Benchmark/StressQA.json
#   Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.csv (50 rows for testing)
```

### Evaluate a System

```python
from StressTest.enhanced_scorer import StressQAScorer
import pandas as pd

# Load benchmark
df = pd.read_csv("Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.csv")

# Load model answers (your system's outputs)
model_answers = [...]  # List of JSON strings

# Score
scorer = StressQAScorer()
results = []

for idx, row in df.iterrows():
    gt = {
        'columns': json.loads(row['relevant_column']),
        'acceptable_methods': json.loads(row['acceptable_methods']),
        'oracle': json.loads(row['oracle']),
        'is_applicable': row['is_applicable'],
    }
    
    score = scorer.score_single_answer(model_answers[idx], gt)
    results.append(score)

# Aggregate
results_df = pd.DataFrame(results)
print(f"Overall score: {results_df['overall_score'].mean():.3f}")

# Breakdown by difficulty axis
df['overall_score'] = results_df['overall_score']
axis_breakdown = scorer.analyze_by_difficulty_axes(df)
print(axis_breakdown)
```

## File Structure

```
StatQA/
├── test_spec_registry.py          # Test specification registry
├── StressTest/
│   ├── modifiers.py                # Scenario modifiers
│   ├── oracle_computer.py          # Oracle computation
│   ├── benchmark_generator.py      # StressQA generation pipeline
│   └── enhanced_scorer.py          # Layered evaluation
├── External/
│   └── materialize_datasets.py     # statsmodels/sklearn loader
├── Script/
│   ├── external_dataset_materialize.sh
│   ├── stress_benchmark_construction.sh
│   └── stress_prompt_organization.sh
├── prompt_wording.py               # Extended with STRESSQA_* prompts
├── utils.py                        # Updated extract_json_answer (balanced braces)
└── Data/
    ├── External Dataset/
    │   └── Origin/                 # Materialized external datasets
    └── Integrated Dataset/
        └── Balanced Benchmark/
            ├── StressQA.csv
            ├── StressQA.json
            ├── mini-StressQA.csv
            └── mini-StressQA.json
```

## Design Rationale

### Why composable modifiers?

Adding modifiers like `heteroscedasticity + multiple_endpoints` creates exponentially more stress scenarios without hand-crafting every combination. Each modifier is independently testable and can be applied to any base test.

### Why acceptable method sets?

Real statistical practice allows multiple valid approaches (e.g., Pearson vs Spearman for correlation when normality is borderline). StatQA's exact-match scoring penalized this legitimate flexibility. StressQA's `acceptable_methods` field encodes sets like:

```python
[
  ["Independent Samples t-test", "Welch t-test"],  # Both valid
  ["Mann-Whitney U Test"]                           # Nonparametric alternative
]
```

Scoring checks if the system's choice is in *any* acceptable set.

### Why numeric tolerances?

Different statistical packages (R, Python scipy, statsmodels) use slightly different algorithms (especially for df in Welch's test, p-value approximations). Tolerances (p±0.01, statistic±5%, effect size±10%) avoid penalizing trivial numerical differences while catching meaningful errors.

### Why audit trails?

This is **the key differentiator for superstat**: documenting *why* a test was chosen, what assumptions were checked, and what alternatives were rejected. Pure LLMs often skip or hallucinate these steps; superstat's deterministic profiling should excel here.

## Extending StressQA

### Add a new modifier

1. Subclass `BaseModifier` in `StressTest/modifiers.py`
2. Implement `apply(df, metadata) -> ModifierResult`
3. Register in `ModifierPipeline.available_modifiers`

Example: outlier injection, missing data patterns, small-n scenarios.

### Add a new test family

1. Add specs to `test_spec_registry.py` (e.g., survival analysis, time series)
2. Implement oracle computation in `oracle_computer.py`
3. Add question templates in `Construction/question_templates.py`
4. Update `PROMPT_CLASSIFICATION` in `prompt_wording.py`

### Add a new evaluation metric

Extend `StressQAScorer` in `StressTest/enhanced_scorer.py`:
- Add scoring method (e.g., `score_sensitivity_analysis`)
- Update `score_single_answer` to call it
- Adjust weights in `component_scores`

## Comparison: StatQA vs StressQA

| Feature | StatQA | StressQA |
|---------|--------|----------|
| **Test families** | 5 (Correlation, Contingency, Distribution, Variance, Descriptive) | **8** (+ Group Comparison, Regression, Multiple Testing) |
| **Difficulty sources** | Task complexity, "Not applicable" ratio | **Composable modifiers** (design traps, assumption violations, confounding) |
| **Acceptable answers** | Exact method match only | **Multiple valid method sets** |
| **Ground truth** | Method name + conclusion | **Full numeric outputs** (stat, p, ES, CI) + checks |
| **Scoring** | Binary (columns + methods correct?) | **Layered** (6 components, weighted, numeric tolerances) |
| **Audit trail** | Not evaluated | **Evaluated** (prerequisite checks, rationale, alternatives) |
| **Output format** | `{"columns": [...], "methods": [...]}` | **Structured contract** (11 fields, nested JSON) |
| **Data sources** | Rdatasets, Kaggle (manual) | **+ statsmodels/sklearn** (materialized) |
| **Evaluation focus** | Test selection | **End-to-end analysis** (selection + computation + interpretation + reasoning) |

## Limitations & Future Work

1. **Oracle coverage:** Currently implements ~15 tests; StatQA had 20+ (mostly distribution/descriptive). Need to port remaining tests.

2. **Synthetic data dominance:** Many modifiers generate synthetic data. Hybrid approach planned: keep real datasets for baseline, apply modifiers to create variants.

3. **Prompt organization:** StressQA benchmark is ready, but prompt organization script (to add full prompts for each trick: zero-shot, CoT, etc.) needs to be adapted from `Construction/prompt_organization.py` to use `STRESSQA_INSTRUCTION`.

4. **Post-hoc computation:** Oracle currently flags when post-hoc is needed but doesn't compute actual Tukey HSD / Dunn's test results. Could add for even stricter evaluation.

5. **Inapplicable cases:** Need more systematic generation of cases where the analysis *should* be rejected (e.g., repeated measures without ID, sparse data, violated assumptions with no valid alternative).

6. **Human baseline:** StatQA included human expert baselines. StressQA should recruit statisticians to attempt harder scenarios and establish human performance ceiling.

## Citation

If you use StressQA, please cite:

```
@misc{stressqa2024,
  title={StressQA: Enhanced Statistical Analysis Benchmark for AI Systems},
  author={[Your Name]},
  year={2024},
  note={Extension of StatQA (https://statqa.github.io/)}
}
```

Original StatQA:
```
@article{statqa2024,
  title={Are Large Language Models Good Statisticians?},
  author={[StatQA authors]},
  url={https://statqa.github.io/},
  year={2024}
}
```

## License

Same license as StatQA (check repository root).

---

**Ready to stress-test your statistical AI? Run `sh Script/stress_benchmark_construction.sh` to build StressQA!**

