# -*- coding: utf-8 -*-
"""
StressQA Benchmark Construction

This module integrates external datasets, applies modifiers, computes oracles,
and generates the StressQA benchmark with enhanced difficulty and evaluation capabilities.
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Import our new modules
from test_spec_registry import (
    get_registry, TestSpec, VariableRole, VariableType, 
    DesignType, PrerequisiteType, VariableSpec
)
from StressTest.modifiers import ModifierPipeline, ModifierResult
from StressTest.oracle_computer import OracleDispatcher, OracleResult
from Construction import question_templates
import utils
import path as path_config

# Try to import scipy for statistical checks, but make it optional
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types"""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.generic,)):
        return obj.item()  # Convert numpy scalar to Python native type
    else:
        return obj


class StressQABenchmarkGenerator:
    """Generates StressQA benchmark with modifiers and oracle computations"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.registry = get_registry()
        self.modifier_pipeline = ModifierPipeline(random_state=random_state)
        self.oracle = OracleDispatcher(alpha=0.05)
        
    def _get_columns_by_type(self, df: pd.DataFrame, var_type: VariableType) -> List[str]:
        """
        Get columns matching a specific variable type.
        
        Args:
            df: DataFrame to search
            var_type: Target variable type
        
        Returns:
            List of column names matching the type
        """
        if var_type == VariableType.CONTINUOUS:
            # Numeric columns (excluding binary)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Filter out binary columns
            continuous = []
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) > 0:
                    unique_vals = series.nunique()
                    # Consider continuous if has many unique values or high variance
                    if unique_vals > 10 or (unique_vals > 2 and series.std() > 1e-6):
                        continuous.append(col)
            return continuous
        
        elif var_type == VariableType.BINARY:
            # Columns with exactly 2 unique values
            binary = []
            for col in df.columns:
                series = df[col].dropna()
                if series.nunique() == 2:
                    binary.append(col)
            return binary
        
        elif var_type == VariableType.CATEGORICAL:
            # Object/category columns, or numeric with few unique values
            categorical = []
            for col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                unique_vals = series.nunique()
                # Categorical if object type or numeric with few categories
                if df[col].dtype in ['object', 'category']:
                    if 2 <= unique_vals <= 50:  # Reasonable number of categories
                        categorical.append(col)
                elif df[col].dtype in [np.number]:
                    if 2 <= unique_vals <= 20:  # Numeric but categorical
                        categorical.append(col)
            return categorical
        
        elif var_type == VariableType.ORDINAL:
            # Similar to categorical but ordered
            return self._get_columns_by_type(df, VariableType.CATEGORICAL)
        
        elif var_type == VariableType.COUNT:
            # Integer columns with non-negative values
            count_cols = []
            for col in df.select_dtypes(include=[np.number]).columns:
                series = df[col].dropna()
                if len(series) > 0 and (series >= 0).all() and series.dtype in [np.int64, np.int32]:
                    count_cols.append(col)
            return count_cols
        
        else:
            return []
    
    def _is_column_suitable(self, series: pd.Series, var_spec: VariableSpec) -> bool:
        """
        Check if a column meets quality requirements for a variable spec.
        
        Args:
            series: Data series to check
            var_spec: Variable specification requirements
        
        Returns:
            True if column is suitable
        """
        # Missing values check
        missing_ratio = series.isna().sum() / len(series) if len(series) > 0 else 1.0
        if missing_ratio > 0.3:  # More than 30% missing
            return False
        
        # Type-specific checks
        if var_spec.var_type == VariableType.CONTINUOUS:
            # Need sufficient variation
            if series.std() < 1e-6:
                return False
            # Need sufficient sample size
            if series.notna().sum() < 30:
                return False
        
        elif var_spec.var_type == VariableType.CATEGORICAL:
            # Need at least 2 categories
            if series.nunique() < 2:
                return False
            # Categories should have reasonable counts (at least 5 per category)
            value_counts = series.value_counts()
            if value_counts.min() < 5:
                return False
        
        elif var_spec.var_type == VariableType.BINARY:
            if series.nunique() != 2:
                return False
            # Both categories should have reasonable counts
            value_counts = series.value_counts()
            if value_counts.min() < 5:
                return False
        
        elif var_spec.var_type == VariableType.COUNT:
            # Count data should be non-negative integers
            if (series < 0).any():
                return False
            if series.notna().sum() < 30:
                return False
        
        return True
    
    def _select_best_column(self, df: pd.DataFrame, candidates: List[str], 
                           var_spec: VariableSpec) -> str:
        """
        Select the best column from candidates based on suitability criteria.
        Prefer columns with better distributional properties.
        
        Args:
            df: DataFrame
            candidates: List of candidate column names
            var_spec: Variable specification
        
        Returns:
            Best column name
        """
        if not candidates:
            raise ValueError("No candidates provided")
        
        if len(candidates) == 1:
            return candidates[0]
        
        scores = {}
        
        for col in candidates:
            score = 0.0
            series = df[col].dropna()
            
            if var_spec.var_type == VariableType.CONTINUOUS:
                # Prefer columns closer to normal distribution
                if HAS_SCIPY and len(series) >= 3:
                    try:
                        # Sample if too large for Shapiro-Wilk
                        sample_size = min(5000, len(series))
                        sample = series.sample(n=sample_size, random_state=self.random_state) if len(series) > sample_size else series
                        _, p_shapiro = stats.shapiro(sample)
                        if not np.isnan(p_shapiro):
                            score += p_shapiro  # Higher p = more normal
                    except:
                        pass
                
                # Prefer columns without extreme outliers
                if HAS_SCIPY and len(series) > 3:
                    try:
                        z_scores = np.abs(stats.zscore(series))
                        outlier_ratio = (z_scores > 3).sum() / len(series)
                        score += (1 - outlier_ratio) * 0.5  # Less weight than normality
                    except:
                        pass
                
                # Prefer columns with good variance
                cv = series.std() / (series.mean() + 1e-10)  # Coefficient of variation
                score += min(cv, 1.0) * 0.3
            
            elif var_spec.var_type in [VariableType.CATEGORICAL, VariableType.BINARY]:
                # Prefer balanced categories
                counts = series.value_counts()
                if len(counts) > 1:
                    balance_score = counts.min() / counts.max()
                    score += balance_score
                else:
                    score += 0.1
            
            elif var_spec.var_type == VariableType.COUNT:
                # Prefer columns with reasonable range
                if series.max() > 0:
                    range_score = min(np.log10(series.max() + 1) / 5.0, 1.0)
                    score += range_score
            
            # Bonus for lower missing data
            missing_ratio = series.isna().sum() / len(df)
            score += (1 - missing_ratio) * 0.2
            
            scores[col] = score
        
        # Return column with highest suitability score
        return max(scores, key=scores.get)
    
    def _identify_id_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify potential ID/subject identifier columns.
        
        Args:
            df: DataFrame to search
        
        Returns:
            List of potential ID column names
        """
        id_candidates = []
        
        for col in df.columns:
            series = df[col].dropna()
            # ID columns typically have:
            # - High uniqueness (most values unique)
            # - String or integer type
            # - No missing values (or very few)
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            missing_ratio = series.isna().sum() / len(df)
            
            if unique_ratio > 0.8 and missing_ratio < 0.1:
                id_candidates.append(col)
        
        return id_candidates
    
    def _validate_pairing(self, df: pd.DataFrame, id_col: str, group_col: str) -> bool:
        """
        Validate that pairing structure is correct for paired design.
        Each ID should have exactly 2 observations (one per group).
        
        Args:
            df: DataFrame
            id_col: ID column name
            group_col: Group column name
        
        Returns:
            True if pairing is valid
        """
        if id_col not in df.columns or group_col not in df.columns:
            return False
        
        # Check that each ID has exactly 2 observations
        id_counts = df.groupby(id_col).size()
        if not (id_counts == 2).all():
            return False
        
        # Check that each ID has both group values
        id_groups = df.groupby(id_col)[group_col].nunique()
        if not (id_groups == 2).all():
            return False
        
        return True
    
    def _check_prerequisites(self, df: pd.DataFrame, case: Dict[str, Any], 
                            spec: TestSpec) -> bool:
        """
        Check if the case satisfies critical prerequisites.
        Return False if critical assumptions are severely violated.
        
        Args:
            df: DataFrame
            case: Case dictionary with column assignments
            spec: Test specification
        
        Returns:
            True if prerequisites are satisfied
        """
        for prereq in spec.prerequisites:
            if not prereq.critical:
                continue  # Only check critical ones at selection time
            
            if prereq.type == PrerequisiteType.MIN_SAMPLE_SIZE:
                if len(df) < 30:
                    return False
            
            elif prereq.type == PrerequisiteType.MIN_EXPECTED_COUNT:
                # For contingency tables
                if 'dv' in case and 'iv' in case:
                    if case['dv'] in df.columns and case['iv'] in df.columns:
                        contingency = pd.crosstab(df[case['dv']], df[case['iv']])
                        expected = contingency.sum(axis=0).values[:, None] * contingency.sum(axis=1).values / contingency.sum().sum()
                        if (expected < 5).any():
                            return False
            
            elif prereq.type == PrerequisiteType.INDEPENDENCE:
                # For paired designs, need proper pairing structure
                if spec.design == DesignType.PAIRED:
                    id_col = case.get('id_col') or case.get('design_metadata', {}).get('id_col')
                    if id_col and 'iv' in case:
                        if not self._validate_pairing(df, id_col, case['iv']):
                            return False
        
        return True
    
    def select_columns_for_spec(self, df: pd.DataFrame, spec: TestSpec, 
                               verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Intelligently select columns that match the test spec's requirements.
        
        Args:
            df: DataFrame to select from
            spec: Test specification
            verbose: If True, print diagnostic information when selection fails
        
        Returns:
            Dict with column assignments or None if no valid combination exists
        """
        selected = {}
        used_columns = set()
        
        # Process required variables first
        required_specs = [vs for vs in spec.variable_specs if vs.required]
        optional_specs = [vs for vs in spec.variable_specs if not vs.required]
        
        # Handle design-specific requirements
        if spec.design == DesignType.PAIRED:
            # Need an ID column for pairing
            id_candidates = self._identify_id_columns(df)
            if not id_candidates:
                if verbose:
                    print(f"    [!] Cannot find ID column for paired design")
                return None  # Can't do paired design without ID
            id_col = self.rng.choice(id_candidates)
            selected['id_col'] = id_col
            used_columns.add(id_col)
        
        elif spec.design == DesignType.STRATIFIED:
            # Need a stratification variable (optional, but helpful)
            strata_candidates = self._get_columns_by_type(df, VariableType.CATEGORICAL)
            strata_candidates = [c for c in strata_candidates 
                               if c not in used_columns
                               and df[c].nunique() >= 2 and df[c].nunique() <= 10]
            if strata_candidates:
                selected['strata_col'] = self.rng.choice(strata_candidates)
                used_columns.add(selected['strata_col'])
        
        # Process required variable specs
        for var_spec in required_specs:
            # Get candidate columns based on required type
            candidates = self._get_columns_by_type(df, var_spec.var_type)
            
            if verbose and not candidates:
                print(f"    [!] No {var_spec.var_type.value} columns found for {var_spec.role.value}")
                # Show what columns are available
                all_cols = list(df.columns)
                print(f"        Available columns: {all_cols}")
                # Show unique value counts
                for col in all_cols:
                    n_unique = df[col].nunique()
                    print(f"          {col}: {n_unique} unique values")
            
            # Filter by data quality
            candidates = [c for c in candidates 
                         if c not in used_columns 
                         and self._is_column_suitable(df[c], var_spec)]
            
            if not candidates:
                if verbose:
                    print(f"    [!] No suitable {var_spec.var_type.value} columns for {var_spec.role.value}")
                    # Show why candidates were filtered out
                    raw_candidates = self._get_columns_by_type(df, var_spec.var_type)
                    raw_candidates = [c for c in raw_candidates if c not in used_columns]
                    if raw_candidates:
                        print(f"        Found {len(raw_candidates)} {var_spec.var_type.value} columns but none passed quality checks:")
                        for col in raw_candidates[:5]:  # Show first 5
                            series = df[col]
                            missing_ratio = series.isna().sum() / len(series)
                            print(f"          {col}: missing={missing_ratio:.1%}, unique={series.nunique()}, std={series.std():.4f}")
                return None  # Can't satisfy requirements
            
            # Select best candidate
            best_col = self._select_best_column(df, candidates, var_spec)
            
            # Map role to case key
            role_key_map = {
                VariableRole.DV: 'dv',
                VariableRole.IV: 'iv',
                VariableRole.COVARIATE: 'covariate',
                VariableRole.STRATA: 'strata_col',
                VariableRole.CLUSTER: 'cluster_col',
                VariableRole.ID: 'id_col',
                VariableRole.TIME: 'time_col',
                VariableRole.CONTROL: 'control',
            }
            
            # Handle multiple IVs for regression (use 'ivs' list)
            if var_spec.role == VariableRole.IV and spec.family == "Regression":
                if 'ivs' not in selected:
                    selected['ivs'] = []
                selected['ivs'].append(best_col)
            else:
                # Use standard key mapping
                key = role_key_map.get(var_spec.role, var_spec.role.value)
                selected[key] = best_col
            
            used_columns.add(best_col)
        
        # Process optional variables if we have room
        for var_spec in optional_specs:
            candidates = self._get_columns_by_type(df, var_spec.var_type)
            candidates = [c for c in candidates 
                         if c not in used_columns 
                         and self._is_column_suitable(df[c], var_spec)]
            
            if candidates:
                best_col = self._select_best_column(df, candidates, var_spec)
                role_key_map = {
                    VariableRole.DV: 'dv',
                    VariableRole.IV: 'iv',
                    VariableRole.COVARIATE: 'covariate',
                    VariableRole.STRATA: 'strata_col',
                    VariableRole.CLUSTER: 'cluster_col',
                    VariableRole.ID: 'id_col',
                    VariableRole.TIME: 'time_col',
                    VariableRole.CONTROL: 'control',
                }
                key = role_key_map.get(var_spec.role, var_spec.role.value)
                selected[key] = best_col
                used_columns.add(best_col)
        
        return selected
    
    def generate_base_cases(self, dataset_df: pd.DataFrame, 
                           dataset_name: str,
                           spec: TestSpec,
                           n_cases: int = 3) -> List[Dict[str, Any]]:
        """
        Generate base test cases for a given dataset and test spec.
        Uses sophisticated column selection based on spec requirements.
        
        Args:
            dataset_df: the dataset to use
            dataset_name: name of the dataset
            spec: test specification
            n_cases: number of cases to generate
        
        Returns:
            List of case dictionaries
        """
        cases = []
        attempts = 0
        max_attempts = n_cases * 20  # Try multiple times to find valid cases
        
        while len(cases) < n_cases and attempts < max_attempts:
            attempts += 1
            
            case = {
                'dataset': dataset_name,
                'spec_id': spec.id,
                'task': spec.family,
                'method': spec.get_primary_method(),
                'difficulty': 'medium',  # Will be updated based on modifiers
            }
            
            # Attempt intelligent column selection
            # Only show verbose diagnostics on first attempt
            verbose = (attempts == 1)
            selected_cols = self.select_columns_for_spec(dataset_df, spec, verbose=verbose)
            if selected_cols is None:
                if attempts == max_attempts:
                    # Log why we failed after all attempts
                    print(f"    [!] Could not generate cases for {spec.id}: no suitable columns found")
                continue  # No valid combination found
            
            # Assign columns to case
            case.update(selected_cols)
            
            # Check prerequisites
            if not self._check_prerequisites(dataset_df, case, spec):
                continue  # Prerequisites not met
            
            # Validate sample sizes per group (for group comparisons)
            if spec.family == "Group Comparison" and 'iv' in case:
                if case['iv'] in dataset_df.columns:
                    group_sizes = dataset_df.groupby(case['iv']).size()
                    if group_sizes.min() < 5:  # Too few observations per group
                        continue
            
            # Check for sufficient variation in DV
            if 'dv' in case and case['dv'] in dataset_df.columns:
                if dataset_df[case['dv']].std() < 1e-6:
                    continue  # No variation
            
            # For paired designs, validate pairing structure
            if spec.design == DesignType.PAIRED and 'id_col' in case and 'iv' in case:
                if not self._validate_pairing(dataset_df, case['id_col'], case['iv']):
                    continue
            
            # Store design metadata
            if 'id_col' in case:
                case.setdefault('design_metadata', {})['id_col'] = case['id_col']
            if 'strata_col' in case:
                case.setdefault('design_metadata', {})['strata_col'] = case['strata_col']
            
            cases.append(case)
        
        return cases
    
    def apply_stress_modifiers(self, dataset_df: pd.DataFrame,
                              base_case: Dict[str, Any],
                              modifier_names: List[str]) -> Dict[str, Any]:
        """
        Apply stress modifiers to a base case.
        
        Returns:
            Enhanced case dictionary with modified data and metadata
        """
        metadata = {'dataset': base_case['dataset']}
        
        try:
            modifier_result = self.modifier_pipeline.apply_modifiers(
                dataset_df, metadata, modifier_names
            )
            
            enhanced_case = base_case.copy()
            enhanced_case['modified_data'] = modifier_result.modified_df
            enhanced_case['difficulty_axes'] = modifier_result.difficulty_axes
            enhanced_case['design_metadata'] = modifier_result.design_metadata
            enhanced_case['oracle_checks'] = modifier_result.oracle_checks
            enhanced_case['question_modifications'] = modifier_result.question_modifications
            enhanced_case['modifier_warnings'] = modifier_result.warnings
            
            # Update difficulty based on number of modifiers
            if len(modifier_names) >= 2:
                enhanced_case['difficulty'] = 'hard'
            elif len(modifier_names) == 1:
                enhanced_case['difficulty'] = 'medium'
            
            return enhanced_case
            
        except Exception as e:
            base_case['modifier_error'] = str(e)
            base_case['modified_data'] = dataset_df
            return base_case
    
    def compute_oracle_for_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute oracle (ground truth) for a test case.
        
        Returns:
            Case with oracle results added
        """
        df = case.get('modified_data', None)
        if df is None or not isinstance(df, pd.DataFrame):
            case['oracle_error'] = "No valid dataframe for oracle computation"
            case['is_applicable'] = False
            return case
        
        # Prepare parameters for oracle
        params = {}
        if 'dv' in case:
            params['dv'] = case['dv']
        if 'iv' in case:
            params['iv'] = case['iv']
        if 'ivs' in case:
            params['ivs'] = case['ivs']
        if 'id_col' in case.get('design_metadata', {}):
            params['id_col'] = case['design_metadata']['id_col']
        
        try:
            oracle_result = self.oracle.compute_oracle(df, case['spec_id'], params)
            case['oracle'] = oracle_result.to_dict()
            case['is_applicable'] = oracle_result.is_applicable
            
            # Store acceptable methods from spec
            spec = self.registry.get(case['spec_id'])
            if spec:
                case['acceptable_methods'] = spec.acceptable_methods
            
            return case
            
        except Exception as e:
            case['oracle_error'] = str(e)
            case['is_applicable'] = False
            return case
    
    def generate_question(self, case: Dict[str, Any]) -> str:
        """
        Generate a natural language question for the case.
        
        Uses question templates and applies any modifier-specific phrasings.
        """
        spec = self.registry.get(case['spec_id'])
        if not spec:
            return "Error: Could not generate question"
        
        # Get base question template
        question = ""
        
        if spec.family == "Group Comparison":
            dv = case.get('dv', 'outcome')
            iv = case.get('iv', 'group')
            
            if 'paired' in case.get('difficulty_axes', []):
                question = f"Compare {dv} between conditions for each subject in {iv}."
            else:
                question = f"Is there a difference in {dv} between groups defined by {iv}?"
        
        elif spec.family == "Correlation Analysis":
            dv = case.get('dv', 'variable1')
            iv = case.get('iv', 'variable2')
            question = f"Is there a correlation between {dv} and {iv}?"
        
        elif spec.family == "Regression":
            dv = case.get('dv', 'outcome')
            ivs = case.get('ivs', ['predictor'])
            if len(ivs) == 1:
                question = f"How does {ivs[0]} predict {dv}?"
            else:
                iv_str = ', '.join(ivs[:-1]) + f', and {ivs[-1]}'
                question = f"How do {iv_str} predict {dv}?"
        
        elif spec.family == "Multiple Testing":
            endpoints = case.get('design_metadata', {}).get('endpoint_variables', [])
            group_var = case.get('design_metadata', {}).get('group_variable', 'group')
            if endpoints:
                question = f"Compare {len(endpoints)} outcomes ({', '.join(endpoints[:3])}...) between {group_var} groups."
        
        else:
            question = f"Analyze {case.get('dv', 'the data')} using {spec.name}."
        
        # Apply modifier-specific phrasing if available
        question_mods = case.get('question_modifications', {})
        if 'suggested_phrasing' in question_mods:
            question = question_mods['suggested_phrasing']
        
        return question
    
    def generate_relevant_columns_json(self, case: Dict[str, Any]) -> str:
        """
        Generate relevant_column JSON in StatQA format.
        """
        columns = []
        
        if 'dv' in case:
            columns.append({
                'column_header': case['dv'],
                'is_strata': False,
                'is_control': False,
            })
        
        if 'iv' in case:
            columns.append({
                'column_header': case['iv'],
                'is_strata': False,
                'is_control': False,
            })
        
        if 'ivs' in case:
            for iv in case['ivs']:
                columns.append({
                    'column_header': iv,
                    'is_strata': False,
                    'is_control': False,
                })
        
        # Add special role columns from design metadata
        design_meta = case.get('design_metadata', {})
        if 'paired_id' in design_meta:
            columns.append({
                'column_header': design_meta['paired_id'],
                'is_strata': False,
                'is_control': False,
            })
        
        if 'strata_col' in design_meta:
            columns.append({
                'column_header': design_meta['strata_col'],
                'is_strata': True,
                'is_control': False,
            })
        
        return json.dumps(columns)
    
    def generate_results_json(self, case: Dict[str, Any]) -> str:
        """
        Generate results JSON in StatQA format, but enhanced with oracle outputs.
        """
        oracle_data = case.get('oracle', {})
        
        # Build conclusion
        if not case.get('is_applicable', False):
            conclusion = "Not applicable"
        elif oracle_data.get('p_value') is not None:
            p_val = oracle_data['p_value']
            conclusion = f"Significant (p={p_val:.4f})" if p_val < 0.05 else f"Not significant (p={p_val:.4f})"
        else:
            conclusion = "Computed"
        
        result = {
            'method': case.get('method', 'Unknown'),
            'conclusion': conclusion,
        }
        
        # Add oracle fields
        if oracle_data:
            if 'statistic' in oracle_data:
                result['statistic'] = oracle_data['statistic']
            if 'p_value' in oracle_data:
                result['p_value'] = oracle_data['p_value']
            if 'effect_size' in oracle_data:
                result['effect_size'] = oracle_data['effect_size']
                result['effect_size_type'] = oracle_data.get('effect_size_type')
            if 'ci_lower' in oracle_data and 'ci_upper' in oracle_data:
                result['confidence_interval'] = {
                    'lower': oracle_data['ci_lower'],
                    'upper': oracle_data['ci_upper'],
                    'level': oracle_data.get('ci_level', 0.95),
                }
        
        return json.dumps([result])
    
    def case_to_benchmark_row(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a case to a StressQA benchmark row.
        
        Schema (backward compatible + new fields):
        - dataset, refined_question, relevant_column, results, task, difficulty (existing)
        - analysis_spec_id, difficulty_axes, acceptable_methods, oracle, is_applicable (new)
        - design_metadata, oracle_checks (new, for advanced analysis)
        """
        row = {
            # Existing StatQA columns
            'dataset': case['dataset'],
            'refined_question': self.generate_question(case),
            'relevant_column': self.generate_relevant_columns_json(case),
            'results': self.generate_results_json(case),
            'task': case['task'],
            'difficulty': case['difficulty'],
            
            # New StressQA columns
            'analysis_spec_id': case['spec_id'],
            'difficulty_axes': json.dumps(convert_to_json_serializable(case.get('difficulty_axes', []))),
            'acceptable_methods': json.dumps(convert_to_json_serializable(case.get('acceptable_methods', []))),
            'oracle': json.dumps(convert_to_json_serializable(case.get('oracle', {}))),
            'is_applicable': case.get('is_applicable', True),
            'design_metadata': json.dumps(convert_to_json_serializable(case.get('design_metadata', {}))),
            'oracle_checks': json.dumps(convert_to_json_serializable(case.get('oracle_checks', {}))),
        }
        
        return row
    
    def generate_stressqa_benchmark(self, 
                                   datasets: List[str],
                                   n_cases_per_dataset: int = 10,
                                   modifier_combinations: Optional[List[List[str]]] = None) -> pd.DataFrame:
        """
        Generate the full StressQA benchmark.
        
        Args:
            datasets: list of dataset names (from external or existing data)
            n_cases_per_dataset: number of test cases per dataset
            modifier_combinations: list of modifier combinations to apply
        
        Returns:
            DataFrame with benchmark rows
        """
        if modifier_combinations is None:
            # Default combinations
            modifier_combinations = [
                [],  # Base case (no modifiers)
                ['heteroscedasticity'],
                ['paired_unpaired_trap'],
                ['sparse_contingency'],
                ['multiple_endpoints'],
                ['confounding_simpsons'],
                ['heteroscedasticity', 'multiple_endpoints'],  # Compound
            ]
        
        all_rows = []
        
        # Get test specs for group comparison and regression (our new families)
        group_comp_specs = self.registry.get_by_family("Group Comparison")
        regression_specs = self.registry.get_by_family("Regression")
        multiple_testing_specs = self.registry.get_by_family("Multiple Testing")
        
        target_specs = group_comp_specs + regression_specs + multiple_testing_specs
        
        print(f"[*] Generating StressQA with {len(datasets)} datasets, {len(target_specs)} specs")
        
        for dataset_name in datasets:
            print(f"\n[*] Processing dataset: {dataset_name}")
            
            # Load dataset
            dataset_path = f"Data/External Dataset/Origin/{dataset_name}.csv"
            if not os.path.exists(dataset_path):
                # Try origin dataset path
                dataset_path = f"{path_config.dataset_dir}{dataset_name}.csv"
            
            if not os.path.exists(dataset_path):
                print(f"[!] Dataset not found: {dataset_name}")
                continue
            
            try:
                df = pd.read_csv(dataset_path)
            except Exception as e:
                print(f"[!] Error loading {dataset_name}: {e}")
                continue
            
            # For each spec, generate cases
            for spec in target_specs[:3]:  # Limit to avoid too many cases
                print(f"  - Spec: {spec.id}")
                
                # Generate base cases
                base_cases = self.generate_base_cases(df, dataset_name, spec, n_cases=2)
                
                if not base_cases:
                    print(f"    [!] No base cases generated for {spec.id}")
                
                for base_case in base_cases:
                    # Apply modifier combinations
                    for modifiers in modifier_combinations[:3]:  # Limit combinations
                        # Apply modifiers
                        enhanced_case = self.apply_stress_modifiers(df, base_case, modifiers)
                        
                        # Compute oracle
                        enhanced_case = self.compute_oracle_for_case(enhanced_case)
                        
                        # Convert to row
                        row = self.case_to_benchmark_row(enhanced_case)
                        all_rows.append(row)
        
        # Create DataFrame
        df_benchmark = pd.DataFrame(all_rows)
        
        print(f"\n[+] Generated {len(df_benchmark)} benchmark rows")
        
        return df_benchmark


def integrate_and_postprocess_stressqa(output_name: str = "StressQA"):
    """
    Main entrypoint: generate and postprocess StressQA benchmark.
    """
    print("=" * 60)
    print("StressQA Benchmark Generation")
    print("=" * 60)
    
    # Initialize generator
    generator = StressQABenchmarkGenerator(random_state=42)
    
    # Get list of available datasets
    # For now, use external datasets inventory
    inventory_path = "Data/External Dataset/Origin/_dataset_inventory.csv"
    
    if os.path.exists(inventory_path):
        inventory = pd.read_csv(inventory_path)
        datasets = inventory['name'].tolist()[:5]  # Limit to first 5 for MVP
    else:
        # Fallback: use a small sample
        datasets = ['iris', 'wine']
    
    print(f"[*] Using datasets: {datasets}")
    
    # Generate benchmark
    benchmark_df = generator.generate_stressqa_benchmark(
        datasets=datasets,
        n_cases_per_dataset=5,
    )
    
    # Postprocess: add ground_truth column (StatQA compatibility)
    def extract_ground_truth_row(row):
        try:
            results = json.loads(row['results'])
            relevant_cols = json.loads(row['relevant_column'])
            
            methods = [r['method'] for r in results if r.get('conclusion') != "Not applicable"]
            columns = [c['column_header'] for c in relevant_cols]
            
            return json.dumps({"columns": columns, "methods": methods})
        except:
            return json.dumps({"columns": [], "methods": []})
    
    benchmark_df['ground_truth'] = benchmark_df.apply(extract_ground_truth_row, axis=1)
    
    # Rearrange columns (existing first for compatibility)
    existing_cols = ['dataset', 'refined_question', 'relevant_column', 'results', 
                    'ground_truth', 'task', 'difficulty']
    new_cols = ['analysis_spec_id', 'difficulty_axes', 'acceptable_methods', 
                'oracle', 'is_applicable', 'design_metadata', 'oracle_checks']
    
    final_cols = [c for c in existing_cols if c in benchmark_df.columns] + \
                 [c for c in new_cols if c in benchmark_df.columns]
    
    benchmark_df = benchmark_df[final_cols]
    
    # Save
    output_dir = path_config.integ_dataset_path + path_config.balance_path
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    json_path = os.path.join(output_dir, f"{output_name}.json")
    
    benchmark_df.to_csv(csv_path, index=False)
    print(f"[+] Saved CSV: {csv_path}")
    
    # Save JSON
    json_data = benchmark_df.to_json(orient='records', indent=4, force_ascii=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    print(f"[+] Saved JSON: {json_path}")
    
    # Create mini version for testing
    mini_df = benchmark_df.sample(n=min(50, len(benchmark_df)), random_state=42)
    mini_csv = os.path.join(output_dir, f"mini-{output_name}.csv")
    mini_json = os.path.join(output_dir, f"mini-{output_name}.json")
    
    mini_df.to_csv(mini_csv, index=False)
    mini_json_data = mini_df.to_json(orient='records', indent=4, force_ascii=False)
    with open(mini_json, 'w', encoding='utf-8') as f:
        f.write(mini_json_data)
    
    print(f"[+] Saved mini version: {mini_csv}")
    
    print("\n" + "=" * 60)
    print(f"[✓] StressQA generation complete!")
    print(f"    Total cases: {len(benchmark_df)}")
    print(f"    Mini cases: {len(mini_df)}")
    print("=" * 60)
    
    return benchmark_df


if __name__ == "__main__":
    integrate_and_postprocess_stressqa()

