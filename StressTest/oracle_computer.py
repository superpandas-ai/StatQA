# -*- coding: utf-8 -*-
"""
Oracle Computation for StressQA

This module computes deterministic ground-truth statistical outputs for various
test families, including new group comparison, regression, and multiple testing scenarios.
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import json


@dataclass
class OracleResult:
    """Container for oracle computation results"""
    method: str
    is_applicable: bool
    statistic: Optional[float] = None
    statistic_name: Optional[str] = None
    df: Optional[float] = None  # Can be fractional for Welch
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    post_hoc_results: Optional[Dict] = None
    correction_applied: Optional[str] = None
    assumptions: Optional[Dict[str, bool]] = None
    warnings: List[str] = None
    additional_info: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Handle None values
        return {k: v for k, v in result.items() if v is not None}


class OracleComputer:
    """Base class for oracle computation"""
    
    def __init__(self, alpha: float = 0.05, ci_level: float = 0.95):
        self.alpha = alpha
        self.ci_level = ci_level
    
    def compute(self, df: pd.DataFrame, spec: Dict[str, Any]) -> OracleResult:
        """Compute oracle result"""
        raise NotImplementedError


class GroupComparisonOracle(OracleComputer):
    """Oracle for group comparison tests"""
    
    def compute_independent_ttest(self, df: pd.DataFrame, dv: str, iv: str) -> OracleResult:
        """Compute independent samples t-test"""
        groups = df[iv].unique()
        if len(groups) != 2:
            return OracleResult(
                method="Independent Samples t-test",
                is_applicable=False,
                warnings=[f"Requires exactly 2 groups, got {len(groups)}"]
            )
        
        group1 = df[df[iv] == groups[0]][dv].dropna()
        group2 = df[df[iv] == groups[1]][dv].dropna()
        
        if len(group1) < 2 or len(group2) < 2:
            return OracleResult(
                method="Independent Samples t-test",
                is_applicable=False,
                warnings=["Need at least 2 observations per group"]
            )
        
        # Run t-test
        t_stat, p_val = stats.ttest_ind(group1, group2)
        
        # Check assumptions
        _, p_norm1 = stats.shapiro(group1) if len(group1) <= 5000 else (None, 0.05)
        _, p_norm2 = stats.shapiro(group2) if len(group2) <= 5000 else (None, 0.05)
        _, p_levene = stats.levene(group1, group2)
        
        assumptions = {
            'normality_group1': p_norm1 > self.alpha if p_norm1 else None,
            'normality_group2': p_norm2 > self.alpha if p_norm2 else None,
            'equal_variance': p_levene > self.alpha,
        }
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(group1) - 1) * group1.std() ** 2 + 
                              (len(group2) - 1) * group2.std() ** 2) / 
                             (len(group1) + len(group2) - 2))
        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for mean difference
        se = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
        df_val = len(group1) + len(group2) - 2
        t_crit = stats.t.ppf(1 - self.alpha/2, df_val)
        mean_diff = group1.mean() - group2.mean()
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        
        warnings = []
        if not assumptions['equal_variance']:
            warnings.append("Equal variance assumption violated; consider Welch's t-test")
        if assumptions['normality_group1'] is not None and not assumptions['normality_group1']:
            warnings.append("Group 1 may not be normally distributed")
        if assumptions['normality_group2'] is not None and not assumptions['normality_group2']:
            warnings.append("Group 2 may not be normally distributed")
        
        return OracleResult(
            method="Independent Samples t-test",
            is_applicable=True,
            statistic=float(t_stat),
            statistic_name="t",
            df=float(df_val),
            p_value=float(p_val),
            effect_size=float(cohens_d),
            effect_size_type="cohen_d",
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            ci_level=self.ci_level,
            assumptions=assumptions,
            warnings=warnings if warnings else None,
            additional_info={
                'mean_group1': float(group1.mean()),
                'mean_group2': float(group2.mean()),
                'mean_difference': float(mean_diff),
            }
        )
    
    def compute_welch_ttest(self, df: pd.DataFrame, dv: str, iv: str) -> OracleResult:
        """Compute Welch's t-test (unequal variances)"""
        groups = df[iv].unique()
        if len(groups) != 2:
            return OracleResult(
                method="Welch t-test",
                is_applicable=False,
                warnings=[f"Requires exactly 2 groups, got {len(groups)}"]
            )
        
        group1 = df[df[iv] == groups[0]][dv].dropna()
        group2 = df[df[iv] == groups[1]][dv].dropna()
        
        if len(group1) < 2 or len(group2) < 2:
            return OracleResult(
                method="Welch t-test",
                is_applicable=False,
                warnings=["Need at least 2 observations per group"]
            )
        
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Welch-Satterthwaite degrees of freedom
        s1, s2 = group1.var(), group2.var()
        n1, n2 = len(group1), len(group2)
        df_welch = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        
        # Cohen's d (with unequal variances adjustment)
        cohens_d = (group1.mean() - group2.mean()) / np.sqrt((s1 + s2) / 2)
        
        # CI for mean difference
        se = np.sqrt(s1/n1 + s2/n2)
        t_crit = stats.t.ppf(1 - self.alpha/2, df_welch)
        mean_diff = group1.mean() - group2.mean()
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        
        return OracleResult(
            method="Welch t-test",
            is_applicable=True,
            statistic=float(t_stat),
            statistic_name="t",
            df=float(df_welch),
            p_value=float(p_val),
            effect_size=float(cohens_d),
            effect_size_type="cohen_d",
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            ci_level=self.ci_level,
            additional_info={
                'mean_group1': float(group1.mean()),
                'mean_group2': float(group2.mean()),
                'mean_difference': float(mean_diff),
            }
        )
    
    def compute_mann_whitney(self, df: pd.DataFrame, dv: str, iv: str) -> OracleResult:
        """Compute Mann-Whitney U test"""
        groups = df[iv].unique()
        if len(groups) != 2:
            return OracleResult(
                method="Mann-Whitney U Test",
                is_applicable=False,
                warnings=[f"Requires exactly 2 groups, got {len(groups)}"]
            )
        
        group1 = df[df[iv] == groups[0]][dv].dropna()
        group2 = df[df[iv] == groups[1]][dv].dropna()
        
        if len(group1) < 1 or len(group2) < 1:
            return OracleResult(
                method="Mann-Whitney U Test",
                is_applicable=False,
                warnings=["Need at least 1 observation per group"]
            )
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Rank-biserial correlation as effect size
        n1, n2 = len(group1), len(group2)
        rank_biserial = 1 - (2*u_stat) / (n1 * n2)
        
        return OracleResult(
            method="Mann-Whitney U Test",
            is_applicable=True,
            statistic=float(u_stat),
            statistic_name="U",
            p_value=float(p_val),
            effect_size=float(rank_biserial),
            effect_size_type="rank_biserial",
            additional_info={
                'median_group1': float(group1.median()),
                'median_group2': float(group2.median()),
            }
        )
    
    def compute_paired_ttest(self, df: pd.DataFrame, dv: str, iv: str, id_col: str) -> OracleResult:
        """Compute paired samples t-test"""
        # Check that we have exactly 2 conditions per ID
        counts = df.groupby(id_col)[iv].nunique()
        
        if not (counts == 2).all():
            return OracleResult(
                method="Paired Samples t-test",
                is_applicable=False,
                warnings=["Not all subjects have exactly 2 measurements"]
            )
        
        # Pivot to wide format
        try:
            pivot = df.pivot(index=id_col, columns=iv, values=dv)
            conditions = pivot.columns.tolist()
            
            if len(conditions) != 2:
                return OracleResult(
                    method="Paired Samples t-test",
                    is_applicable=False,
                    warnings=[f"Expected 2 conditions, got {len(conditions)}"]
                )
            
            sample1 = pivot[conditions[0]].dropna()
            sample2 = pivot[conditions[1]].dropna()
            
            # Find common IDs
            common_ids = sample1.index.intersection(sample2.index)
            sample1 = sample1.loc[common_ids]
            sample2 = sample2.loc[common_ids]
            
            if len(sample1) < 2:
                return OracleResult(
                    method="Paired Samples t-test",
                    is_applicable=False,
                    warnings=["Need at least 2 complete pairs"]
                )
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(sample1, sample2)
            
            # Cohen's d for paired data
            differences = sample1 - sample2
            cohens_d = differences.mean() / differences.std()
            
            # CI for mean difference
            n = len(differences)
            se = differences.std() / np.sqrt(n)
            t_crit = stats.t.ppf(1 - self.alpha/2, n - 1)
            mean_diff = differences.mean()
            ci_lower = mean_diff - t_crit * se
            ci_upper = mean_diff + t_crit * se
            
            return OracleResult(
                method="Paired Samples t-test",
                is_applicable=True,
                statistic=float(t_stat),
                statistic_name="t",
                df=float(n - 1),
                p_value=float(p_val),
                effect_size=float(cohens_d),
                effect_size_type="cohen_d",
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                ci_level=self.ci_level,
                additional_info={
                    'n_pairs': int(n),
                    'mean_difference': float(mean_diff),
                }
            )
        except Exception as e:
            return OracleResult(
                method="Paired Samples t-test",
                is_applicable=False,
                warnings=[f"Error computing paired test: {str(e)}"]
            )
    
    def compute_one_way_anova(self, df: pd.DataFrame, dv: str, iv: str) -> OracleResult:
        """Compute one-way ANOVA"""
        groups = df[iv].unique()
        
        if len(groups) < 2:
            return OracleResult(
                method="One-Way ANOVA",
                is_applicable=False,
                warnings=["Need at least 2 groups"]
            )
        
        # Collect data for each group
        group_data = [df[df[iv] == g][dv].dropna() for g in groups]
        
        # Check minimum sample size
        if any(len(g) < 2 for g in group_data):
            return OracleResult(
                method="One-Way ANOVA",
                is_applicable=False,
                warnings=["Need at least 2 observations per group"]
            )
        
        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(*group_data)
        
        # Degrees of freedom
        k = len(groups)  # number of groups
        n = sum(len(g) for g in group_data)  # total n
        df_between = k - 1
        df_within = n - k
        
        # Eta squared (effect size)
        grand_mean = df[dv].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
        ss_total = sum((df[dv] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Check assumptions
        _, p_levene = stats.levene(*group_data)
        
        warnings = []
        if p_levene < self.alpha:
            warnings.append("Equal variance assumption may be violated (Levene's test significant)")
        
        # Post-hoc: Tukey HSD
        post_hoc = None
        if p_val < self.alpha:
            # Indicate that post-hoc is needed
            warnings.append("Significant omnibus test; post-hoc comparisons recommended")
            post_hoc = {
                'recommended': ['Tukey HSD', 'Bonferroni'],
                'reason': 'Omnibus test significant',
            }
        
        return OracleResult(
            method="One-Way ANOVA",
            is_applicable=True,
            statistic=float(f_stat),
            statistic_name="F",
            df=f"{df_between}, {df_within}",
            p_value=float(p_val),
            effect_size=float(eta_squared),
            effect_size_type="eta_squared",
            post_hoc_results=post_hoc,
            assumptions={'equal_variance': p_levene > self.alpha},
            warnings=warnings if warnings else None,
            additional_info={
                'n_groups': int(k),
                'total_n': int(n),
            }
        )
    
    def compute_kruskal_wallis(self, df: pd.DataFrame, dv: str, iv: str) -> OracleResult:
        """Compute Kruskal-Wallis H test"""
        groups = df[iv].unique()
        
        if len(groups) < 2:
            return OracleResult(
                method="Kruskal-Wallis H Test",
                is_applicable=False,
                warnings=["Need at least 2 groups"]
            )
        
        # Collect data for each group
        group_data = [df[df[iv] == g][dv].dropna() for g in groups]
        
        # Check minimum sample size
        if any(len(g) < 1 for g in group_data):
            return OracleResult(
                method="Kruskal-Wallis H Test",
                is_applicable=False,
                warnings=["Need at least 1 observation per group"]
            )
        
        # Kruskal-Wallis test
        h_stat, p_val = stats.kruskal(*group_data)
        
        # Degrees of freedom
        k = len(groups)
        df_val = k - 1
        
        # Epsilon squared (effect size for Kruskal-Wallis)
        n = sum(len(g) for g in group_data)
        epsilon_squared = (h_stat - k + 1) / (n - k) if n > k else 0
        epsilon_squared = max(0, min(1, epsilon_squared))  # Clamp to [0, 1]
        
        # Post-hoc
        post_hoc = None
        warnings = []
        if p_val < self.alpha:
            warnings.append("Significant test; Dunn's post-hoc with correction recommended")
            post_hoc = {
                'recommended': ["Dunn's test with Bonferroni correction"],
                'reason': 'Omnibus test significant',
            }
        
        return OracleResult(
            method="Kruskal-Wallis H Test",
            is_applicable=True,
            statistic=float(h_stat),
            statistic_name="H",
            df=float(df_val),
            p_value=float(p_val),
            effect_size=float(epsilon_squared),
            effect_size_type="epsilon_squared",
            post_hoc_results=post_hoc,
            warnings=warnings if warnings else None,
            additional_info={
                'n_groups': int(k),
                'total_n': int(n),
            }
        )


class RegressionOracle(OracleComputer):
    """Oracle for regression and ANCOVA"""
    
    def compute_linear_regression(self, df: pd.DataFrame, dv: str, 
                                  iv: List[str]) -> OracleResult:
        """Compute linear regression (simple or multiple)"""
        try:
            import statsmodels.api as sm
        except ImportError:
            return OracleResult(
                method="Linear Regression",
                is_applicable=False,
                warnings=["statsmodels not available for regression computation"]
            )
        
        # Prepare data
        y = df[dv].dropna()
        X = df[iv].dropna()
        
        # Find common indices
        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]
        
        if len(y) < len(iv) + 2:
            return OracleResult(
                method="Linear Regression",
                is_applicable=False,
                warnings=[f"Need at least {len(iv) + 2} observations for {len(iv)} predictors"]
            )
        
        # Add constant
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Extract results
        f_stat = model.fvalue
        p_val = model.f_pvalue
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        
        # Coefficients
        coefficients = model.params.to_dict()
        std_errors = model.bse.to_dict()
        t_values = model.tvalues.to_dict()
        p_values = model.pvalues.to_dict()
        
        # Check assumptions (simplified)
        residuals = model.resid
        _, p_shapiro = stats.shapiro(residuals) if len(residuals) <= 5000 else (None, None)
        
        assumptions = {}
        if p_shapiro is not None:
            assumptions['residuals_normal'] = p_shapiro > self.alpha
        
        warnings = []
        if p_shapiro is not None and p_shapiro < self.alpha:
            warnings.append("Residuals may not be normally distributed")
        
        return OracleResult(
            method="Linear Regression",
            is_applicable=True,
            statistic=float(f_stat),
            statistic_name="F",
            df=f"{len(iv)}, {len(y) - len(iv) - 1}",
            p_value=float(p_val),
            effect_size=float(r_squared),
            effect_size_type="r_squared",
            assumptions=assumptions,
            warnings=warnings if warnings else None,
            additional_info={
                'adjusted_r_squared': float(adj_r_squared),
                'coefficients': {k: float(v) for k, v in coefficients.items()},
                'std_errors': {k: float(v) for k, v in std_errors.items()},
                't_values': {k: float(v) for k, v in t_values.items()},
                'p_values': {k: float(v) for k, v in p_values.items()},
                'n_predictors': len(iv),
                'n_observations': int(len(y)),
            }
        )


class MultipleTestingOracle(OracleComputer):
    """Oracle for multiple testing scenarios"""
    
    def compute_multiple_comparisons(self, df: pd.DataFrame, 
                                     outcome_vars: List[str],
                                     group_var: str,
                                     correction: str = 'fdr_bh') -> OracleResult:
        """Compute multiple comparisons with correction"""
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            return OracleResult(
                method=f"Multiple Endpoints with {correction}",
                is_applicable=False,
                warnings=["statsmodels not available for multiple testing correction"]
            )
        
        groups = df[group_var].unique()
        if len(groups) != 2:
            return OracleResult(
                method=f"Multiple Endpoints with {correction}",
                is_applicable=False,
                warnings=[f"Currently only supports 2 groups, got {len(groups)}"]
            )
        
        # Compute raw p-values for each outcome
        raw_p_values = []
        test_results = []
        
        for outcome in outcome_vars:
            group1 = df[df[group_var] == groups[0]][outcome].dropna()
            group2 = df[df[group_var] == groups[1]][outcome].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                raw_p_values.append(np.nan)
                test_results.append({'outcome': outcome, 'error': 'insufficient data'})
                continue
            
            t_stat, p_val = stats.ttest_ind(group1, group2)
            raw_p_values.append(p_val)
            test_results.append({
                'outcome': outcome,
                't_statistic': float(t_stat),
                'raw_p_value': float(p_val),
            })
        
        # Remove NaN p-values
        valid_indices = [i for i, p in enumerate(raw_p_values) if not np.isnan(p)]
        valid_p_values = [raw_p_values[i] for i in valid_indices]
        
        if not valid_p_values:
            return OracleResult(
                method=f"Multiple Endpoints with {correction}",
                is_applicable=False,
                warnings=["No valid tests computed"]
            )
        
        # Apply correction
        reject, pvals_corrected, _, _ = multipletests(
            valid_p_values, alpha=self.alpha, method=correction
        )
        
        # Update test results with corrected p-values
        for i, idx in enumerate(valid_indices):
            test_results[idx]['adjusted_p_value'] = float(pvals_corrected[i])
            test_results[idx]['significant_raw'] = valid_p_values[i] < self.alpha
            test_results[idx]['significant_adjusted'] = bool(reject[i])
        
        n_sig_raw = sum(1 for p in valid_p_values if p < self.alpha)
        n_sig_adj = sum(reject)
        
        return OracleResult(
            method=f"Multiple Endpoints with {correction.upper()}",
            is_applicable=True,
            p_value=None,  # No single p-value
            correction_applied=correction,
            additional_info={
                'n_tests': len(valid_p_values),
                'n_significant_raw': int(n_sig_raw),
                'n_significant_adjusted': int(n_sig_adj),
                'test_results': test_results,
                'correction_method': correction,
            }
        )


# Main oracle dispatcher
class OracleDispatcher:
    """Dispatches oracle computation to appropriate computers"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.group_comp = GroupComparisonOracle(alpha=alpha)
        self.regression = RegressionOracle(alpha=alpha)
        self.multiple_testing = MultipleTestingOracle(alpha=alpha)
    
    def compute_oracle(self, df: pd.DataFrame, spec_id: str, 
                      params: Dict[str, Any]) -> OracleResult:
        """
        Dispatch oracle computation based on spec_id.
        
        Args:
            df: dataset
            spec_id: test specification ID
            params: parameters including variable roles, etc.
        
        Returns:
            OracleResult
        """
        # Group comparison tests
        if spec_id == "independent_t_test":
            return self.group_comp.compute_independent_ttest(
                df, params['dv'], params['iv']
            )
        elif spec_id == "welch_t_test":
            return self.group_comp.compute_welch_ttest(
                df, params['dv'], params['iv']
            )
        elif spec_id == "mann_whitney":
            return self.group_comp.compute_mann_whitney(
                df, params['dv'], params['iv']
            )
        elif spec_id == "paired_t_test":
            return self.group_comp.compute_paired_ttest(
                df, params['dv'], params['iv'], params['id_col']
            )
        elif spec_id == "one_way_anova":
            return self.group_comp.compute_one_way_anova(
                df, params['dv'], params['iv']
            )
        elif spec_id == "kruskal_wallis":
            return self.group_comp.compute_kruskal_wallis(
                df, params['dv'], params['iv']
            )
        # Regression
        elif spec_id in ["simple_linear_regression", "multiple_linear_regression"]:
            return self.regression.compute_linear_regression(
                df, params['dv'], params['ivs']
            )
        # Multiple testing
        elif spec_id == "multiple_endpoints":
            return self.multiple_testing.compute_multiple_comparisons(
                df, params['outcome_vars'], params['group_var'],
                correction=params.get('correction', 'fdr_bh')
            )
        else:
            return OracleResult(
                method=spec_id,
                is_applicable=False,
                warnings=[f"Oracle computation not implemented for {spec_id}"]
            )


if __name__ == "__main__":
    # Test oracle computations
    print("=" * 60)
    print("Testing Oracle Computations")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Test independent t-test
    print("\n[1] Testing Independent t-test Oracle...")
    df_ttest = pd.DataFrame({
        'value': np.concatenate([np.random.randn(50) + 0.5, np.random.randn(50)]),
        'group': ['A'] * 50 + ['B'] * 50,
    })
    
    oracle = OracleDispatcher()
    result = oracle.compute_oracle(df_ttest, "independent_t_test", {'dv': 'value', 'iv': 'group'})
    print(f"    Method: {result.method}")
    print(f"    Applicable: {result.is_applicable}")
    print(f"    t = {result.statistic:.3f}, p = {result.p_value:.4f}")
    print(f"    Cohen's d = {result.effect_size:.3f}")
    
    print("\n[✓] Oracle computation testing complete!")

