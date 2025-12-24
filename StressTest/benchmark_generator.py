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
from dataclasses import as dict

# Import our new modules
from test_spec_registry import get_registry, TestSpec
from StressTest.modifiers import ModifierPipeline, ModifierResult
from StressTest.oracle_computer import OracleDispatcher, OracleResult
from Construction import question_templates
import utils
import path as path_config


class StressQABenchmarkGenerator:
    """Generates StressQA benchmark with modifiers and oracle computations"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.registry = get_registry()
        self.modifier_pipeline = ModifierPipeline(random_state=random_state)
        self.oracle = OracleDispatcher(alpha=0.05)
        
    def generate_base_cases(self, dataset_df: pd.DataFrame, 
                           dataset_name: str,
                           spec: TestSpec,
                           n_cases: int = 3) -> List[Dict[str, Any]]:
        """
        Generate base test cases for a given dataset and test spec.
        
        Args:
            dataset_df: the dataset to use
            dataset_name: name of the dataset
            spec: test specification
            n_cases: number of cases to generate
        
        Returns:
            List of case dictionaries
        """
        cases = []
        
        # Simple strategy: randomly select columns matching the spec's requirements
        numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = dataset_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for i in range(n_cases):
            case = {
                'dataset': dataset_name,
                'spec_id': spec.id,
                'task': spec.family,
                'method': spec.get_primary_method(),
                'difficulty': 'medium',  # Will be updated based on modifiers
            }
            
            # Select columns based on spec requirements
            # This is a simplified version; real implementation should be more sophisticated
            if spec.family == "Group Comparison":
                if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                    case['dv'] = self.rng.choice(numeric_cols)
                    case['iv'] = self.rng.choice(categorical_cols)
                else:
                    continue  # Skip if columns not available
            elif spec.family == "Correlation Analysis":
                if len(numeric_cols) >= 2:
                    selected = self.rng.choice(numeric_cols, size=2, replace=False)
                    case['dv'] = selected[0]
                    case['iv'] = selected[1]
                else:
                    continue
            elif spec.family == "Regression":
                if len(numeric_cols) >= 2:
                    case['dv'] = self.rng.choice(numeric_cols)
                    remaining = [c for c in numeric_cols if c != case['dv']]
                    n_iv = min(3, len(remaining))
                    case['ivs'] = self.rng.choice(remaining, size=n_iv, replace=False).tolist()
                else:
                    continue
            else:
                # Generic case
                if len(numeric_cols) > 0:
                    case['dv'] = self.rng.choice(numeric_cols)
            
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
            'difficulty_axes': json.dumps(case.get('difficulty_axes', [])),
            'acceptable_methods': json.dumps(case.get('acceptable_methods', [])),
            'oracle': json.dumps(case.get('oracle', {})),
            'is_applicable': case.get('is_applicable', True),
            'design_metadata': json.dumps(case.get('design_metadata', {})),
            'oracle_checks': json.dumps(case.get('oracle_checks', {})),
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

