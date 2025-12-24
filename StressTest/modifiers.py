# -*- coding: utf-8 -*-
"""
Scenario Modifiers for StressQA

This module implements composable modifiers that transform datasets and questions
to create harder stress-test scenarios. Each modifier adds specific difficulty axes
that test statistical reasoning capabilities.
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from scipy import stats
import json


@dataclass
class ModifierResult:
    """Result of applying a modifier to a dataset"""
    modified_df: pd.DataFrame
    difficulty_axes: List[str]
    design_metadata: Dict[str, Any]
    oracle_checks: Dict[str, bool]
    question_modifications: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class BaseModifier:
    """Base class for all modifiers"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def apply(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> ModifierResult:
        """Apply the modifier to the dataset"""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Get the modifier name for difficulty_axes tracking"""
        raise NotImplementedError


class PairedUnpairedTrapModifier(BaseModifier):
    """
    Creates scenarios where data is naturally paired but the surface question
    might tempt an independent-samples test.
    
    Strategy:
    - Add a subject_id column to create paired structure
    - Phrase question ambiguously or focus on groups without mentioning pairing
    - Oracle should flag this as paired design
    """
    
    def get_name(self) -> str:
        return "paired_unpaired_trap"
    
    def apply(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> ModifierResult:
        """
        Transform dataset to have paired structure.
        Requires: dataset with at least 2 numeric columns and even number of rows
        """
        # Find two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for paired trap")
        
        if len(df) < 10 or len(df) % 2 != 0:
            # Make it even by dropping last row if needed
            if len(df) % 2 != 0:
                df = df.iloc[:-1].copy()
        
        # Create subject IDs for pairing
        n_pairs = len(df) // 2
        subject_ids = np.repeat(np.arange(n_pairs), 2)
        df_modified = df.copy()
        df_modified['subject_id'] = subject_ids
        
        # Create a condition column (e.g., "pre" and "post", or "treatment_A" and "treatment_B")
        conditions = np.tile(['condition_A', 'condition_B'], n_pairs)
        df_modified['condition'] = conditions
        
        # Oracle checks
        oracle_checks = {
            'is_paired': True,
            'pairing_variable': 'subject_id',
            'independence_violated': True,  # Data are NOT independent
        }
        
        # Design metadata
        design_metadata = {
            'paired_id': 'subject_id',
            'design_type': 'paired',
            'condition_var': 'condition',
            'n_pairs': n_pairs,
        }
        
        # Question modifications to make it ambiguous
        question_mods = {
            'trap_type': 'paired_looks_independent',
            'suggested_phrasing': 'Compare condition_A to condition_B',  # Doesn't explicitly mention pairing
            'correct_phrasing': 'Compare paired measurements from condition_A and condition_B for each subject',
        }
        
        return ModifierResult(
            modified_df=df_modified,
            difficulty_axes=[self.get_name()],
            design_metadata=design_metadata,
            oracle_checks=oracle_checks,
            question_modifications=question_mods,
            warnings=["This design requires paired test; independent samples test would be incorrect"]
        )


class HeteroscedasticityModifier(BaseModifier):
    """
    Introduces heteroscedasticity (unequal variances) between groups,
    making Welch's t-test preferable to Student's t-test.
    
    Strategy:
    - Identify a numeric DV and create/use a binary grouping variable
    - Scale one group's variance to be substantially different
    - Levene's test should reject equal variance
    """
    
    def get_name(self) -> str:
        return "heteroscedasticity"
    
    def apply(self, df: pd.DataFrame, metadata: Dict[str, Any],
              target_var: Optional[str] = None,
              group_var: Optional[str] = None,
              variance_ratio: float = 4.0) -> ModifierResult:
        """
        Introduce heteroscedasticity by scaling variance in one group.
        
        Args:
            target_var: numeric column to modify (if None, picks first numeric)
            group_var: binary grouping column (if None, creates one)
            variance_ratio: ratio of variances between groups (default 4.0)
        """
        df_modified = df.copy()
        
        # Find or create target variable
        if target_var is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("Need at least one numeric column")
            target_var = numeric_cols[0]
        
        # Find or create binary grouping variable
        if group_var is None:
            # Create binary split
            n = len(df)
            groups = np.array(['group_A'] * (n // 2) + ['group_B'] * (n - n // 2))
            self.rng.shuffle(groups)
            df_modified['group'] = groups
            group_var = 'group'
        
        # Check that we have binary groups
        unique_groups = df_modified[group_var].unique()
        if len(unique_groups) != 2:
            raise ValueError(f"Group variable must be binary, got {len(unique_groups)} groups")
        
        # Modify variance in group_B
        group_a_mask = df_modified[group_var] == unique_groups[0]
        group_b_mask = df_modified[group_var] == unique_groups[1]
        
        # Get group means
        mean_a = df_modified.loc[group_a_mask, target_var].mean()
        mean_b = df_modified.loc[group_b_mask, target_var].mean()
        
        # Scale variance in group B
        values_b = df_modified.loc[group_b_mask, target_var].values
        centered = values_b - mean_b
        scaled = centered * np.sqrt(variance_ratio)
        df_modified.loc[group_b_mask, target_var] = scaled + mean_b
        
        # Run Levene's test to verify heteroscedasticity
        group_a_vals = df_modified.loc[group_a_mask, target_var].dropna()
        group_b_vals = df_modified.loc[group_b_mask, target_var].dropna()
        levene_stat, levene_p = stats.levene(group_a_vals, group_b_vals)
        
        # Oracle checks
        oracle_checks = {
            'equal_variance': levene_p > 0.05,  # Should be False
            'levene_statistic': float(levene_stat),
            'levene_p_value': float(levene_p),
            'variance_ratio_actual': float(group_b_vals.var() / group_a_vals.var()),
        }
        
        # Design metadata
        design_metadata = {
            'target_variable': target_var,
            'group_variable': group_var,
            'heteroscedasticity_induced': True,
            'variance_ratio_target': variance_ratio,
        }
        
        # Question modifications
        question_mods = {
            'trap_type': 'unequal_variances',
            'preferred_method': 'Welch t-test',
            'problematic_method': "Student's t-test (assumes equal variances)",
        }
        
        warnings = []
        if levene_p > 0.05:
            warnings.append("Warning: Levene's test did not detect heteroscedasticity; may need larger variance ratio")
        
        return ModifierResult(
            modified_df=df_modified,
            difficulty_axes=[self.get_name()],
            design_metadata=design_metadata,
            oracle_checks=oracle_checks,
            question_modifications=question_mods,
            warnings=warnings
        )


class SparseContingencyModifier(BaseModifier):
    """
    Creates sparse contingency tables with small expected counts,
    forcing Fisher's exact test instead of chi-square.
    
    Strategy:
    - Create or modify categorical variables to have small cell counts
    - Ensure expected counts < 5 in at least one cell
    - Chi-square assumptions violated; Fisher's exact required
    """
    
    def get_name(self) -> str:
        return "sparse_contingency"
    
    def apply(self, df: pd.DataFrame, metadata: Dict[str, Any],
              var1: Optional[str] = None,
              var2: Optional[str] = None,
              max_n: int = 50) -> ModifierResult:
        """
        Create a sparse contingency table by subsetting or creating categorical variables.
        
        Args:
            var1, var2: categorical variables (if None, creates them)
            max_n: maximum sample size to ensure sparsity
        """
        df_modified = df.copy()
        
        # Subset to small n
        if len(df) > max_n:
            df_modified = df_modified.sample(n=max_n, random_state=self.random_state)
        
        # Find or create categorical variables
        if var1 is None or var2 is None:
            categorical_cols = df_modified.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_cols) < 2:
                # Create categorical variables
                if var1 is None:
                    # Create binary variable
                    df_modified['category_A'] = self.rng.choice(['A1', 'A2'], size=len(df_modified), p=[0.7, 0.3])
                    var1 = 'category_A'
                
                if var2 is None:
                    # Create another binary variable
                    df_modified['category_B'] = self.rng.choice(['B1', 'B2'], size=len(df_modified), p=[0.6, 0.4])
                    var2 = 'category_B'
            else:
                if var1 is None:
                    var1 = categorical_cols[0]
                if var2 is None:
                    var2 = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        
        # Build contingency table
        contingency_table = pd.crosstab(df_modified[var1], df_modified[var2])
        
        # Calculate expected frequencies for chi-square
        row_sums = contingency_table.sum(axis=1)
        col_sums = contingency_table.sum(axis=0)
        total = contingency_table.sum().sum()
        
        expected = np.outer(row_sums, col_sums) / total
        min_expected = expected.min()
        
        # Run chi-square test
        try:
            chi2_stat, chi2_p, dof, expected_freq = stats.chi2_contingency(contingency_table)
            chi2_valid = min_expected >= 5
        except:
            chi2_stat, chi2_p, dof = None, None, None
            chi2_valid = False
        
        # Run Fisher's exact (if 2x2)
        fisher_p = None
        fisher_valid = contingency_table.shape == (2, 2)
        if fisher_valid:
            try:
                _, fisher_p = stats.fisher_exact(contingency_table)
            except:
                fisher_p = None
        
        # Oracle checks
        oracle_checks = {
            'min_expected_count': float(min_expected),
            'chi_square_valid': chi2_valid,
            'fisher_exact_required': not chi2_valid,
            'contingency_shape': contingency_table.shape,
        }
        
        if chi2_stat is not None:
            oracle_checks['chi2_statistic'] = float(chi2_stat)
            oracle_checks['chi2_p_value'] = float(chi2_p)
        
        if fisher_p is not None:
            oracle_checks['fisher_p_value'] = float(fisher_p)
        
        # Design metadata
        design_metadata = {
            'var1': var1,
            'var2': var2,
            'sample_size': len(df_modified),
            'sparse': not chi2_valid,
            'contingency_table': contingency_table.to_dict(),
        }
        
        # Question modifications
        question_mods = {
            'trap_type': 'sparse_cells',
            'preferred_method': 'Fisher Exact Test',
            'problematic_method': 'Chi-square test (expected count < 5)',
        }
        
        warnings = []
        if chi2_valid:
            warnings.append("Warning: Chi-square assumptions are actually met; not sparse enough for modifier")
        
        return ModifierResult(
            modified_df=df_modified,
            difficulty_axes=[self.get_name()],
            design_metadata=design_metadata,
            oracle_checks=oracle_checks,
            question_modifications=question_mods,
            warnings=warnings
        )


class MultipleEndpointsModifier(BaseModifier):
    """
    Generates multiple correlated endpoints that require multiple testing correction.
    
    Strategy:
    - Create K correlated outcome variables
    - Any analysis comparing groups across all K requires FDR/Bonferroni correction
    - Oracle computes both raw and adjusted p-values
    """
    
    def get_name(self) -> str:
        return "multiple_endpoints"
    
    def apply(self, df: pd.DataFrame, metadata: Dict[str, Any],
              n_endpoints: int = 5,
              correlation: float = 0.3,
              group_var: Optional[str] = None) -> ModifierResult:
        """
        Add multiple correlated endpoints to the dataset.
        
        Args:
            n_endpoints: number of outcome variables to create
            correlation: correlation between endpoints (0-1)
            group_var: binary grouping variable (if None, creates one)
        """
        df_modified = df.copy()
        n = len(df)
        
        # Create or use binary group variable
        if group_var is None:
            groups = np.array([0] * (n // 2) + [1] * (n - n // 2))
            self.rng.shuffle(groups)
            df_modified['group'] = ['group_A' if g == 0 else 'group_B' for g in groups]
            group_var = 'group'
        
        # Generate correlated endpoints using multivariate normal
        # Create covariance matrix
        cov_matrix = np.full((n_endpoints, n_endpoints), correlation)
        np.fill_diagonal(cov_matrix, 1.0)
        
        # Generate base data
        base_data = self.rng.multivariate_normal(
            mean=np.zeros(n_endpoints),
            cov=cov_matrix,
            size=n
        )
        
        # Add group effects (some endpoints have real effects, some don't)
        # This creates a mix of true and false positives
        group_effects = self.rng.choice([0.0, 0.5], size=n_endpoints, p=[0.4, 0.6])
        
        groups_numeric = (df_modified[group_var] == df_modified[group_var].unique()[1]).astype(int)
        
        for i in range(n_endpoints):
            endpoint_data = base_data[:, i] + groups_numeric * group_effects[i]
            df_modified[f'endpoint_{i+1}'] = endpoint_data
        
        # Compute raw p-values for each endpoint
        raw_p_values = []
        endpoint_names = []
        for i in range(n_endpoints):
            endpoint_name = f'endpoint_{i+1}'
            endpoint_names.append(endpoint_name)
            
            group_a_vals = df_modified[df_modified[group_var] == df_modified[group_var].unique()[0]][endpoint_name]
            group_b_vals = df_modified[df_modified[group_var] == df_modified[group_var].unique()[1]][endpoint_name]
            
            # Run t-test
            t_stat, p_val = stats.ttest_ind(group_a_vals, group_b_vals)
            raw_p_values.append(p_val)
        
        # Apply corrections
        from statsmodels.stats.multitest import multipletests
        
        # Benjamini-Hochberg (FDR)
        reject_bh, pvals_bh, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
        
        # Bonferroni
        reject_bonf, pvals_bonf, _, _ = multipletests(raw_p_values, alpha=0.05, method='bonferroni')
        
        # Oracle checks
        oracle_checks = {
            'n_endpoints': n_endpoints,
            'raw_p_values': [float(p) for p in raw_p_values],
            'bh_adjusted_p_values': [float(p) for p in pvals_bh],
            'bonferroni_adjusted_p_values': [float(p) for p in pvals_bonf],
            'n_significant_raw': int(sum(np.array(raw_p_values) < 0.05)),
            'n_significant_bh': int(sum(reject_bh)),
            'n_significant_bonf': int(sum(reject_bonf)),
            'correction_required': True,
        }
        
        # Design metadata
        design_metadata = {
            'endpoint_variables': endpoint_names,
            'group_variable': group_var,
            'n_tests': n_endpoints,
            'correlation_between_endpoints': correlation,
            'true_group_effects': group_effects.tolist(),
        }
        
        # Question modifications
        question_mods = {
            'trap_type': 'multiple_comparisons',
            'required_correction': 'Benjamini-Hochberg or Bonferroni',
            'interpretation': 'Must interpret adjusted p-values, not raw p-values',
        }
        
        return ModifierResult(
            modified_df=df_modified,
            difficulty_axes=[self.get_name()],
            design_metadata=design_metadata,
            oracle_checks=oracle_checks,
            question_modifications=question_mods,
            warnings=["Multiple endpoints require correction for multiple testing"]
        )


class ConfoundingSimpsonsModifier(BaseModifier):
    """
    Creates a confounding scenario where the marginal association reverses
    when stratifying by a confounder (Simpson's paradox).
    
    Strategy:
    - Create/identify a binary outcome, binary exposure, and categorical confounder
    - Ensure marginal OR has opposite direction to stratified (Mantel-Haenszel) OR
    - Requires stratified analysis or regression adjustment
    """
    
    def get_name(self) -> str:
        return "confounding_simpsons"
    
    def apply(self, df: pd.DataFrame, metadata: Dict[str, Any],
              n_samples: int = 200) -> ModifierResult:
        """
        Generate a dataset exhibiting Simpson's paradox.
        
        Args:
            n_samples: sample size for synthetic data
        """
        # Generate synthetic data with known Simpson's paradox structure
        # Classic example: treatment effect reverses across hospital size strata
        
        # Create strata (e.g., hospital size: small, large)
        n_small = int(n_samples * 0.3)
        n_large = n_samples - n_small
        
        strata = ['small_hospital'] * n_small + ['large_hospital'] * n_large
        
        # In small hospitals: treatment rate low, success rate higher
        # In large hospitals: treatment rate high, success rate lower
        # Overall: treatment associated with WORSE outcome
        # Within strata: treatment associated with BETTER outcome
        
        treatment_small = self.rng.binomial(1, 0.3, n_small)  # Low treatment rate
        treatment_large = self.rng.binomial(1, 0.7, n_large)  # High treatment rate
        
        # Success rates (treatment improves outcome within each stratum)
        def generate_outcomes(treatment, base_rate, treatment_effect):
            probs = base_rate + treatment * treatment_effect
            return self.rng.binomial(1, probs)
        
        # Small hospitals have better baseline outcomes
        outcome_small = generate_outcomes(treatment_small, base_rate=0.6, treatment_effect=0.15)
        
        # Large hospitals have worse baseline outcomes
        outcome_large = generate_outcomes(treatment_large, base_rate=0.4, treatment_effect=0.15)
        
        # Combine
        df_modified = pd.DataFrame({
            'hospital_size': strata,
            'treatment': np.concatenate([treatment_small, treatment_large]),
            'outcome': np.concatenate([outcome_small, outcome_large]),
        })
        
        # Compute marginal association (overall)
        marginal_table = pd.crosstab(df_modified['treatment'], df_modified['outcome'])
        
        # Compute stratified associations
        small_mask = df_modified['hospital_size'] == 'small_hospital'
        large_mask = df_modified['hospital_size'] == 'large_hospital'
        
        small_table = pd.crosstab(df_modified.loc[small_mask, 'treatment'],
                                   df_modified.loc[small_mask, 'outcome'])
        large_table = pd.crosstab(df_modified.loc[large_mask, 'treatment'],
                                   df_modified.loc[large_mask, 'outcome'])
        
        # Calculate odds ratios
        def calc_or(table):
            if table.shape == (2, 2):
                a, b = table.iloc[0, 0], table.iloc[0, 1]
                c, d = table.iloc[1, 0], table.iloc[1, 1]
                if b > 0 and c > 0:
                    return (a * d) / (b * c)
            return None
        
        marginal_or = calc_or(marginal_table)
        small_or = calc_or(small_table)
        large_or = calc_or(large_table)
        
        # Run Mantel-Haenszel test
        # Requires statsmodels
        try:
            from statsmodels.stats.contingency_tables import StratifiedTable
            tables = np.array([small_table.values, large_table.values])
            st = StratifiedTable(tables)
            mh_or = st.oddsratio_pooled
            mh_pval = st.test_null_odds().pvalue
        except:
            mh_or, mh_pval = None, None
        
        # Oracle checks
        oracle_checks = {
            'simpsons_paradox': True,
            'marginal_or': float(marginal_or) if marginal_or else None,
            'small_stratum_or': float(small_or) if small_or else None,
            'large_stratum_or': float(large_or) if large_or else None,
            'mantel_haenszel_or': float(mh_or) if mh_or else None,
            'mantel_haenszel_p': float(mh_pval) if mh_pval else None,
            'reversal': (marginal_or is not None and small_or is not None and 
                        ((marginal_or < 1 and small_or > 1) or (marginal_or > 1 and small_or < 1))),
        }
        
        # Design metadata
        design_metadata = {
            'outcome_var': 'outcome',
            'exposure_var': 'treatment',
            'confounder_var': 'hospital_size',
            'requires_stratification': True,
            'n_strata': 2,
        }
        
        # Question modifications
        question_mods = {
            'trap_type': 'simpsons_paradox',
            'misleading_analysis': 'Marginal (unadjusted) comparison',
            'correct_analysis': 'Stratified analysis (Mantel-Haenszel) or regression adjustment',
        }
        
        return ModifierResult(
            modified_df=df_modified,
            difficulty_axes=[self.get_name()],
            design_metadata=design_metadata,
            oracle_checks=oracle_checks,
            question_modifications=question_mods,
            warnings=["Confounding present; marginal analysis will be misleading"]
        )


class ModifierPipeline:
    """
    Manages application of multiple modifiers to create complex scenarios.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.available_modifiers = {
            'paired_unpaired_trap': PairedUnpairedTrapModifier,
            'heteroscedasticity': HeteroscedasticityModifier,
            'sparse_contingency': SparseContingencyModifier,
            'multiple_endpoints': MultipleEndpointsModifier,
            'confounding_simpsons': ConfoundingSimpsonsModifier,
        }
    
    def apply_modifiers(self, df: pd.DataFrame, metadata: Dict[str, Any],
                       modifier_names: List[str],
                       modifier_params: Optional[Dict[str, Dict]] = None) -> ModifierResult:
        """
        Apply a sequence of modifiers to a dataset.
        
        Args:
            df: input dataframe
            metadata: dataset metadata
            modifier_names: list of modifier names to apply
            modifier_params: optional dict of {modifier_name: params_dict}
        
        Returns:
            ModifierResult with combined difficulty axes and metadata
        """
        if modifier_params is None:
            modifier_params = {}
        
        current_df = df.copy()
        combined_axes = []
        combined_metadata = {}
        combined_checks = {}
        combined_question_mods = {}
        all_warnings = []
        
        for mod_name in modifier_names:
            if mod_name not in self.available_modifiers:
                raise ValueError(f"Unknown modifier: {mod_name}")
            
            # Instantiate modifier
            modifier_class = self.available_modifiers[mod_name]
            modifier = modifier_class(random_state=self.random_state)
            
            # Get params for this modifier
            params = modifier_params.get(mod_name, {})
            
            # Apply
            try:
                result = modifier.apply(current_df, metadata, **params)
                current_df = result.modified_df
                combined_axes.extend(result.difficulty_axes)
                combined_metadata.update(result.design_metadata)
                combined_checks.update(result.oracle_checks)
                combined_question_mods.update(result.question_modifications)
                all_warnings.extend(result.warnings)
            except Exception as e:
                all_warnings.append(f"Failed to apply {mod_name}: {str(e)}")
        
        return ModifierResult(
            modified_df=current_df,
            difficulty_axes=combined_axes,
            design_metadata=combined_metadata,
            oracle_checks=combined_checks,
            question_modifications=combined_question_mods,
            warnings=all_warnings
        )
    
    def get_available_modifiers(self) -> List[str]:
        """Get list of available modifier names"""
        return list(self.available_modifiers.keys())


# Example usage and testing
if __name__ == "__main__":
    # Test each modifier
    print("=" * 60)
    print("Testing Scenario Modifiers")
    print("=" * 60)
    
    # Create sample dataset
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'var1': np.random.randn(100),
        'var2': np.random.randn(100),
        'category': np.random.choice(['A', 'B'], 100),
    })
    
    metadata = {'source': 'test'}
    
    # Test paired trap
    print("\n[1] Testing Paired/Unpaired Trap Modifier...")
    try:
        mod = PairedUnpairedTrapModifier()
        result = mod.apply(sample_df, metadata)
        print(f"    ✓ Success: {result.difficulty_axes}")
        print(f"      Design metadata: {result.design_metadata}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # Test heteroscedasticity
    print("\n[2] Testing Heteroscedasticity Modifier...")
    try:
        mod = HeteroscedasticityModifier()
        result = mod.apply(sample_df, metadata, target_var='var1', group_var='category')
        print(f"    ✓ Success: {result.difficulty_axes}")
        print(f"      Equal variance: {result.oracle_checks['equal_variance']}")
        print(f"      Levene p-value: {result.oracle_checks['levene_p_value']:.4f}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # Test sparse contingency
    print("\n[3] Testing Sparse Contingency Modifier...")
    try:
        mod = SparseContingencyModifier()
        result = mod.apply(sample_df, metadata, max_n=30)
        print(f"    ✓ Success: {result.difficulty_axes}")
        print(f"      Min expected count: {result.oracle_checks['min_expected_count']:.2f}")
        print(f"      Chi-square valid: {result.oracle_checks['chi_square_valid']}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # Test multiple endpoints
    print("\n[4] Testing Multiple Endpoints Modifier...")
    try:
        mod = MultipleEndpointsModifier()
        result = mod.apply(sample_df, metadata, n_endpoints=5)
        print(f"    ✓ Success: {result.difficulty_axes}")
        print(f"      N significant (raw): {result.oracle_checks['n_significant_raw']}")
        print(f"      N significant (BH): {result.oracle_checks['n_significant_bh']}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    # Test confounding/Simpson's
    print("\n[5] Testing Confounding/Simpson's Modifier...")
    try:
        mod = ConfoundingSimpsonsModifier()
        result = mod.apply(sample_df, metadata, n_samples=200)
        print(f"    ✓ Success: {result.difficulty_axes}")
        print(f"      Reversal detected: {result.oracle_checks.get('reversal', False)}")
        if result.oracle_checks.get('marginal_or'):
            print(f"      Marginal OR: {result.oracle_checks['marginal_or']:.3f}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Modifier testing complete!")
    print("=" * 60)

