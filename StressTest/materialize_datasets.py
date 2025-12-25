# -*- coding: utf-8 -*-
"""
Materialize Modified Datasets for StressQA

This script reads the StressQA benchmark JSON and materializes (saves) all modified
datasets that are referenced in the benchmark. This ensures that columns like
condition_A, condition_B, subject_id, etc. that are created by modifiers are
actually available in saved CSV files.

Usage:
    python StressTest/materialize_datasets.py [--benchmark-file path/to/benchmark.json] [--output-dir Data/External Dataset/Modified/]
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import pandas as pd
import numpy as np
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Import StressQA components
from StressTest.modifiers import ModifierPipeline
import path as path_config


def create_dataset_signature(entry: Dict[str, Any]) -> str:
    """
    Create a unique signature for a benchmark entry based on dataset and modifiers.
    
    Args:
        entry: Benchmark entry dictionary
        
    Returns:
        Unique signature string with format: dataset_modifier1_modifier2_hash
    """
    dataset = entry['dataset']
    difficulty_axes = json.loads(entry.get('difficulty_axes', '[]'))
    design_metadata = json.loads(entry.get('design_metadata', '{}'))
    
    # Create a hash from the modifier combination and key parameters
    sig_parts = [dataset] + sorted(difficulty_axes)
    
    # Add key parameters from design_metadata for reproducibility
    if 'target_variable' in design_metadata:
        sig_parts.append(f"target_{design_metadata['target_variable']}")
    if 'group_variable' in design_metadata:
        sig_parts.append(f"group_{design_metadata['group_variable']}")
    if 'variance_ratio_target' in design_metadata:
        sig_parts.append(f"var_ratio_{design_metadata['variance_ratio_target']}")
    if 'n_pairs' in design_metadata:
        sig_parts.append(f"pairs_{design_metadata['n_pairs']}")
    
    sig_string = "_".join(str(p) for p in sig_parts)
    # Create a short hash for uniqueness
    sig_hash = hashlib.md5(sig_string.encode()).hexdigest()[:8]
    
    # Build filename with modifier names included
    if difficulty_axes:
        # Include modifier names in filename: dataset_modifier1_modifier2_hash
        modifier_str = "_".join(sorted(difficulty_axes))
        return f"{dataset}_{modifier_str}_{sig_hash}"
    else:
        # No modifiers, just dataset and hash
        return f"{dataset}_{sig_hash}"


def extract_modifier_params(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract modifier parameters from design_metadata.
    
    Args:
        entry: Benchmark entry dictionary
        
    Returns:
        Dictionary mapping modifier names to their parameters
    """
    difficulty_axes = json.loads(entry.get('difficulty_axes', '[]'))
    design_metadata = json.loads(entry.get('design_metadata', '{}'))
    
    params = {}
    
    for modifier in difficulty_axes:
        if modifier == 'heteroscedasticity':
            params[modifier] = {
                'target_var': design_metadata.get('target_variable'),
                'group_var': design_metadata.get('group_variable'),
                'variance_ratio': design_metadata.get('variance_ratio_target', 4.0),
            }
        elif modifier == 'paired_unpaired_trap':
            # Paired modifier doesn't need explicit params, but we can note the structure
            params[modifier] = {
                'paired_id': design_metadata.get('paired_id', 'subject_id'),
                'condition_var': design_metadata.get('condition_var', 'condition'),
            }
        elif modifier == 'sparse_contingency':
            params[modifier] = {
                'var1': design_metadata.get('var1'),
                'var2': design_metadata.get('var2'),
                'max_n': design_metadata.get('sample_size', 50),
            }
        elif modifier == 'multiple_endpoints':
            endpoint_vars = design_metadata.get('endpoint_variables', [])
            n_endpoints = len(endpoint_vars) if endpoint_vars else 5
            params[modifier] = {
                'n_endpoints': n_endpoints,
                'group_var': design_metadata.get('group_variable'),
                'correlation': design_metadata.get('correlation_between_endpoints', 0.3),
            }
        elif modifier == 'confounding_simpsons':
            params[modifier] = {
                'n_samples': design_metadata.get('n_strata', 200),
            }
        else:
            params[modifier] = {}
    
    return params


def materialize_single_dataset(
    dataset_name: str,
    modifier_names: List[str],
    modifier_params: Dict[str, Dict[str, Any]],
    output_path: str,
    random_state: int = 42
) -> bool:
    """
    Materialize a single modified dataset.
    
    Args:
        dataset_name: Name of the original dataset
        modifier_names: List of modifier names to apply
        modifier_params: Parameters for each modifier
        output_path: Path to save the modified dataset
        random_state: Random seed for reproducibility
        
    Returns:
        True if successful, False otherwise
    """
    # Load original dataset
    dataset_path = f"Data/External Dataset/Origin/{dataset_name}.csv"
    if not os.path.exists(dataset_path):
        # Try alternative path
        dataset_path = f"{path_config.dataset_dir}{dataset_name}.csv"
    
    if not os.path.exists(dataset_path):
        print(f"  [!] Original dataset not found: {dataset_name}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"  [!] Error loading {dataset_name}: {e}")
        return False
    
    # Apply modifiers
    if not modifier_names:
        # No modifiers, just save the original
        df.to_csv(output_path, index=False)
        return True
    
    try:
        modifier_pipeline = ModifierPipeline(random_state=random_state)
        metadata = {'dataset': dataset_name}
        
        # Prepare parameters for modifier pipeline
        # Note: Some modifiers (like paired_unpaired_trap) don't accept parameters
        pipeline_params = {}
        for mod_name in modifier_names:
            if mod_name in modifier_params:
                # Filter out None values
                filtered_params = {
                    k: v for k, v in modifier_params[mod_name].items() 
                    if v is not None
                }
                # Only include params if the modifier actually accepts them
                # paired_unpaired_trap doesn't accept params, so skip them
                if mod_name == 'paired_unpaired_trap':
                    # This modifier doesn't accept parameters
                    continue
                elif mod_name == 'heteroscedasticity':
                    # For heteroscedasticity, if group_var is specified but doesn't exist,
                    # don't pass it (let the modifier create it)
                    if 'group_var' in filtered_params:
                        group_var_name = filtered_params['group_var']
                        if group_var_name not in df.columns:
                            # Group variable doesn't exist, remove it so modifier creates it
                            filtered_params.pop('group_var')
                if filtered_params:
                    pipeline_params[mod_name] = filtered_params
        
        result = modifier_pipeline.apply_modifiers(
            df, metadata, modifier_names, modifier_params=pipeline_params if pipeline_params else None
        )
        
        # Save modified dataset
        result.modified_df.to_csv(output_path, index=False)
        return True
        
    except Exception as e:
        print(f"  [!] Error applying modifiers to {dataset_name}: {e}")
        return False


def materialize_benchmark_datasets(
    benchmark_file: str,
    output_dir: str = "Data/External Dataset/Modified/",
    random_state: int = 42
) -> Dict[str, str]:
    """
    Materialize all modified datasets referenced in the benchmark.
    
    Args:
        benchmark_file: Path to benchmark JSON file
        output_dir: Directory to save materialized datasets
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping dataset signatures to file paths
    """
    print("=" * 60)
    print("StressQA Dataset Materialization")
    print("=" * 60)
    
    # Load benchmark
    print(f"\n[*] Loading benchmark: {benchmark_file}")
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)
    
    print(f"[*] Found {len(benchmark)} benchmark entries")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group entries by unique dataset+modifier combinations
    print("\n[*] Grouping entries by dataset and modifier combinations...")
    unique_combinations = {}
    
    for entry in benchmark:
        sig = create_dataset_signature(entry)
        if sig not in unique_combinations:
            difficulty_axes = json.loads(entry.get('difficulty_axes', '[]'))
            design_metadata = json.loads(entry.get('design_metadata', '{}'))
            
            unique_combinations[sig] = {
                'dataset': entry['dataset'],
                'modifiers': difficulty_axes,
                'params': extract_modifier_params(entry),
                'count': 0,
            }
        unique_combinations[sig]['count'] += 1
    
    print(f"[*] Found {len(unique_combinations)} unique dataset+modifier combinations")
    
    # Materialize each unique combination
    print("\n[*] Materializing datasets...")
    materialized = {}
    failed = []
    
    for sig, combo in unique_combinations.items():
        dataset_name = combo['dataset']
        modifiers = combo['modifiers']
        params = combo['params']
        
        output_filename = f"{sig}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n  [{len(materialized) + 1}/{len(unique_combinations)}] {sig}")
        print(f"    Dataset: {dataset_name}")
        print(f"    Modifiers: {modifiers if modifiers else 'none (original)'}")
        print(f"    Used in {combo['count']} benchmark entries")
        
        success = materialize_single_dataset(
            dataset_name, modifiers, params, output_path, random_state
        )
        
        if success:
            materialized[sig] = output_path
            print(f"    ✓ Saved: {output_path}")
        else:
            failed.append(sig)
            print(f"    ✗ Failed to materialize")
    
    # Summary
    print("\n" + "=" * 60)
    print("Materialization Summary")
    print("=" * 60)
    print(f"  Total combinations: {len(unique_combinations)}")
    print(f"  Successfully materialized: {len(materialized)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\n  Failed signatures: {failed}")
    
    # Save mapping file
    mapping_file = os.path.join(output_dir, "_materialization_mapping.json")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(materialized, f, indent=2)
    print(f"\n[*] Saved materialization mapping: {mapping_file}")
    
    return materialized


def update_benchmark_with_paths(
    benchmark_file: str,
    materialized: Dict[str, str],
    output_file: Optional[str] = None
) -> None:
    """
    Update benchmark entries with paths to materialized datasets.
    
    Args:
        benchmark_file: Path to original benchmark JSON
        materialized: Dictionary mapping signatures to file paths
        output_file: Optional output file path (default: overwrites original)
    """
    print("\n[*] Updating benchmark with materialized dataset paths...")
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)
    
    updated_count = 0
    for entry in benchmark:
        sig = create_dataset_signature(entry)
        if sig in materialized:
            # Add materialized_dataset_path field
            entry['materialized_dataset_path'] = materialized[sig]
            updated_count += 1
    
    output_path = output_file or benchmark_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=4, ensure_ascii=False)
    
    print(f"  ✓ Updated {updated_count} entries in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Materialize modified datasets for StressQA benchmark"
    )
    parser.add_argument(
        '--benchmark-file',
        type=str,
        default='Data/Integrated Dataset/Balanced Benchmark/mini-StressQA.json',
        help='Path to StressQA benchmark JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='Data/External Dataset/Processed/',
        help='Directory to save materialized datasets'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--update-benchmark',
        action='store_true',
        help='Update benchmark JSON with materialized dataset paths'
    )
    parser.add_argument(
        '--output-benchmark',
        type=str,
        default=None,
        help='Output path for updated benchmark (default: overwrites original)'
    )
    
    args = parser.parse_args()
    
    # Materialize datasets
    materialized = materialize_benchmark_datasets(
        args.benchmark_file,
        args.output_dir,
        args.random_state
    )
    
    # Optionally update benchmark
    if args.update_benchmark:
        update_benchmark_with_paths(
            args.benchmark_file,
            materialized,
            args.output_benchmark
        )
    
    print("\n" + "=" * 60)
    print("Materialization complete!")
    print("=" * 60)
    print(f"\nMaterialized datasets saved to: {args.output_dir}")
    print(f"Mapping file: {os.path.join(args.output_dir, '_materialization_mapping.json')}")


if __name__ == "__main__":
    main()

