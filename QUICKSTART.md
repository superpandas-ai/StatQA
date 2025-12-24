# StressQA Quick Start Guide

Get up and running with StressQA in 5 minutes!

## Prerequisites

```bash
# Ensure you have the required packages
pip install numpy pandas scipy statsmodels scikit-learn

# Check your Python version
python --version  # Should be 3.10+
```

## Step 1: Generate StressQA Benchmark (2 minutes)

```bash
cd /home/haris/git/StatQA

# One command to build everything
sh Script/stress_benchmark_construction.sh
```

**What this does:**
1. Downloads 11 datasets from statsmodels/sklearn
2. Generates test cases with modifiers (paired traps, heteroscedasticity, etc.)
3. Computes oracle ground truth with numeric outputs
4. Outputs `StressQA.csv` + `mini-StressQA.csv` (for testing)

**Expected output:**
```
[Step 1/3] Materializing external datasets...
  ✓ Saved: iris.csv (150 rows, 5 cols)
  ✓ Saved: wine.csv (178 rows, 14 cols)
  ...
[Step 2/3] Generating StressQA benchmark...
  [*] Generated 120 benchmark rows
[Step 3/3] Verifying outputs...
  ✓ StressQA contains 121 rows (including header)
  ✓ mini-StressQA contains 51 rows (including header)
```

## Step 2: Explore the Benchmark (1 minute)

```python
import pandas as pd
import json

# Load mini version for quick exploration
df = pd.read_csv("Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.csv")

print(f"Total cases: {len(df)}")
print(f"\nDifficulty distribution:\n{df['difficulty'].value_counts()}")
print(f"\nTask families:\n{df['task'].value_counts()}")

# Examine a single case
row = df.iloc[0]
print(f"\n--- Example Case ---")
print(f"Dataset: {row['dataset']}")
print(f"Question: {row['refined_question']}")
print(f"Difficulty: {row['difficulty']}")
print(f"Difficulty axes: {row['difficulty_axes']}")

# Oracle ground truth
oracle = json.loads(row['oracle'])
print(f"\nOracle outputs:")
print(f"  Statistic: {oracle.get('statistic')}")
print(f"  P-value: {oracle.get('p_value')}")
print(f"  Effect size: {oracle.get('effect_size')}")
```

## Step 3: Score a System's Output (2 minutes)

```python
from StressTest.enhanced_scorer import StressQAScorer

# Initialize scorer
scorer = StressQAScorer(
    p_value_tolerance=0.01,        # ±0.01 for p-values
    statistic_rel_tolerance=0.05,  # ±5% for test statistics
    effect_size_rel_tolerance=0.10 # ±10% for effect sizes
)

# Mock system output (replace with your system's actual JSON response)
system_output = json.dumps({
    "columns": ["sepal length (cm)", "target"],
    "methods": ["One-Way ANOVA"],
    "applicability": True,
    "checks": {"normality": True, "equal_variance": True},
    "warnings": [],
    "test_result": {"statistic": 119.26, "df": "2, 147", "p_value": 1.67e-31},
    "effect_size": {"value": 0.619, "type": "eta_squared"},
    "ci": None,
    "post_hoc": {"recommended": ["Tukey HSD"], "reason": "Omnibus test significant"},
    "corrections": None,
    "audit_trail": {
        "prerequisite_checks": "Checked normality (Shapiro-Wilk p>0.05 in all groups) and homogeneity of variance (Levene p=0.24)",
        "method_choice_reason": "Selected one-way ANOVA for comparing means across 3 independent groups",
        "alternatives_rejected": "Kruskal-Wallis not needed; parametric assumptions met"
    }
})

# Ground truth from benchmark
gt = {
    'columns': json.loads(row['relevant_column']),
    'acceptable_methods': json.loads(row['acceptable_methods']),
    'oracle': oracle,
    'is_applicable': row['is_applicable'],
}

# Score
result = scorer.score_single_answer(system_output, gt)

print("\n--- Scoring Results ---")
print(f"Overall score: {result['overall_score']:.3f}")
print(f"\nComponent scores:")
for component, score in result['component_scores'].items():
    print(f"  {component}: {score:.3f}")

print(f"\nColumn selection: {result['column_selection']}")
print(f"Method selection: {result['method_selection']}")
print(f"Applicability: {result['applicability']}")
```

## Step 4: Batch Evaluation (bonus)

```python
# Evaluate all cases in mini-StressQA
results = []

for idx, row in df.iterrows():
    # Parse ground truth
    gt = {
        'columns': json.loads(row['relevant_column']),
        'acceptable_methods': json.loads(row['acceptable_methods']),
        'oracle': json.loads(row['oracle']),
        'is_applicable': row['is_applicable'],
    }
    
    # Get your system's output for this case
    # system_output = your_system.analyze(row['refined_question'], row['dataset'])
    
    # For demo, use a mock output
    system_output = json.dumps({
        "columns": json.loads(row['relevant_column'])[0:2],  # Simplified mock
        "methods": json.loads(row['acceptable_methods'])[0] if row['acceptable_methods'] != '[]' else [],
        "applicability": row['is_applicable'],
        "audit_trail": {
            "prerequisite_checks": "Mock checks",
            "method_choice_reason": "Mock reason",
        }
    })
    
    # Score
    score_result = scorer.score_single_answer(system_output, gt)
    score_result['case_id'] = idx
    results.append(score_result)

# Aggregate
results_df = pd.DataFrame(results)
print(f"\n--- Batch Results ---")
print(f"Mean overall score: {results_df['overall_score'].mean():.3f}")
print(f"Std dev: {results_df['overall_score'].std():.3f}")

# Breakdown by difficulty
df['overall_score'] = results_df['overall_score'].values
print(f"\nBy difficulty:")
print(df.groupby('difficulty')['overall_score'].agg(['mean', 'count']))

# Breakdown by difficulty axes
axis_breakdown = scorer.analyze_by_difficulty_axes(df)
print(f"\nBy difficulty axis:")
print(axis_breakdown)
```

## Common Workflows

### Workflow 1: Test Superstat

```python
# Assuming superstat has an API
from superstat import analyze  # hypothetical

for idx, row in df.iterrows():
    # Run superstat
    result = analyze(
        question=row['refined_question'],
        dataset_path=f"Data/External Dataset/Origin/{row['dataset']}.csv",
        columns=None  # Let superstat select
    )
    
    # Score against ground truth
    gt = {...}  # From benchmark
    score = scorer.score_single_answer(json.dumps(result), gt)
    
    # Save scores
    # ...
```

### Workflow 2: Compare Systems

```python
systems = {
    'gpt-4o': lambda q, d: gpt4o_analyze(q, d),
    'superstat': lambda q, d: superstat.analyze(q, d),
    'baseline': lambda q, d: baseline_heuristic(q, d),
}

comparison = []
for system_name, system_fn in systems.items():
    for idx, row in df.iterrows():
        output = system_fn(row['refined_question'], row['dataset'])
        gt = {...}
        score = scorer.score_single_answer(json.dumps(output), gt)
        comparison.append({
            'system': system_name,
            'case_id': idx,
            'score': score['overall_score'],
            'difficulty': row['difficulty'],
        })

# Analyze
comp_df = pd.DataFrame(comparison)
pivot = comp_df.pivot_table(
    index='difficulty',
    columns='system',
    values='score',
    aggfunc='mean'
)
print(pivot)
```

### Workflow 3: Debug Failures

```python
# Find cases where system failed
failed = results_df[results_df['overall_score'] < 0.5]

for idx in failed['case_id']:
    row = df.iloc[idx]
    result = results[idx]
    
    print(f"\n--- Failed Case #{idx} ---")
    print(f"Question: {row['refined_question']}")
    print(f"Difficulty axes: {row['difficulty_axes']}")
    print(f"Score: {result['overall_score']:.3f}")
    
    # Drill down
    if result['column_selection']['exact'] == 0:
        print(f"  Issue: Wrong columns selected")
    if not result['method_selection']['acceptable']:
        print(f"  Issue: Unacceptable method")
    if not result['applicability']['correct']:
        print(f"  Issue: Wrong applicability judgment")
    if result['audit_trail']['completeness'] < 0.5:
        print(f"  Issue: Incomplete audit trail")
```

## Troubleshooting

### Issue: "statsmodels not found"
```bash
pip install statsmodels
```

### Issue: "scikit-learn not found"
```bash
pip install scikit-learn
```

### Issue: "Dataset not found"
Run materialization first:
```bash
sh Script/external_dataset_materialize.sh
```

### Issue: "Invalid JSON in oracle field"
The oracle field should parse as JSON:
```python
oracle = json.loads(row['oracle'])
```
If this fails, the benchmark generation had an error. Regenerate:
```bash
sh Script/stress_benchmark_construction.sh
```

### Issue: "Scorer returns NaN"
Check that your system's output matches the expected JSON contract:
```python
required_fields = ['columns', 'methods', 'applicability']
assert all(field in your_output for field in required_fields)
```

## Next Steps

1. **Read full documentation:** `STRESSQA_README.md`
2. **Understand implementation:** `IMPLEMENTATION_SUMMARY.md`
3. **Extend StressQA:** Add new modifiers, tests, or scoring layers
4. **Contribute:** Share results, suggest improvements

## Support

- **Issues:** Check existing StatQA issues or create new ones
- **Questions:** See `STRESSQA_README.md` FAQ section
- **Examples:** Look at test code in `if __name__ == "__main__"` blocks

---

**Happy stress testing! 🚀**

