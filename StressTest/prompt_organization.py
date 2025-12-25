# -*- coding: utf-8 -*-
"""
StressQA Prompt Organization

This module organizes prompts for StressQA benchmark with enhanced JSON contract
supporting audit trails and comprehensive statistical analysis outputs.
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import argparse
import json
from typing import Optional
import utils
import path
import pandas as pd
from random import choice, sample
from prompt_wording import *


def extract_refined_question(row_index, df):
    """Extract the refined question from a row"""
    if row_index >= len(df):
        return f"Row index {row_index} is out of bounds for the dataset with {len(df)} rows."
    return df.at[row_index, 'refined_question']


def get_dataset_metadata(dataset_name: str, materialized_dataset_path: Optional[str] = None):
    """
    Get column metadata for a dataset.
    Handles both regular StatQA datasets and external datasets.
    If materialized_dataset_path is provided, reads the actual CSV and generates metadata.
    """
    # If materialized dataset path is provided, use it to generate metadata
    if materialized_dataset_path and os.path.exists(materialized_dataset_path):
        try:
            # Read the actual materialized dataset
            df = pd.read_csv(materialized_dataset_path)
            
            # Generate metadata from the actual dataset
            meta_rows = []
            for col in df.columns:
                col_data = df[col]
                
                # Determine data type
                if pd.api.types.is_numeric_dtype(col_data):
                    data_type = 'quantitative'
                    # Check normality (simplified - just check if it's numeric)
                    is_normality = False  # Would need actual normality test
                else:
                    data_type = 'categorical'
                    is_normality = False
                
                meta_rows.append({
                    'column_header': col,
                    'data_type': data_type,
                    'num_of_rows': len(df),
                    'is_normality': is_normality
                })
            
            meta_df = pd.DataFrame(meta_rows)
            return meta_df
        except Exception as e:
            print(f"[!] Error reading materialized dataset {materialized_dataset_path}: {e}")
            # Fall through to regular metadata lookup
    
    # Try regular metadata path first
    metadata_path = f"{path.meta_dir}{path.col_meta_dir}{dataset_name}_col_meta.csv"
    
    if not os.path.exists(metadata_path):
        # Try external dataset path
        external_metadata_path = f"Data/Metadata/Column Metadata/{dataset_name}_col_meta.csv"
        if os.path.exists(external_metadata_path):
            metadata_path = external_metadata_path
        else:
            print(f"[!] Warning: Metadata not found for {dataset_name}")
            return pd.DataFrame()
    
    try:
        meta_df = pd.read_csv(metadata_path)
        return meta_df
    except Exception as e:
        print(f"[!] Error reading metadata for {dataset_name}: {e}")
        return pd.DataFrame()


def format_column_metadata(meta_df):
    """
    Format column metadata into the string format used in prompts.
    """
    meta_info_list = []
    
    for i in range(meta_df.shape[0]):
        row = meta_df.iloc[i]
        col_meta_str = ''
        for header in meta_df.columns:
            # Skip dataset and column_description
            if header not in ["dataset", "column_description"]:
                value = row[header]
                if pd.notna(value):
                    col_meta_str += f"{header}: {value}; "
        
        if col_meta_str:
            # Replace semicolon at end with period
            col_meta_str = col_meta_str[:-2] + '.'
            col_meta_str = col_meta_str.replace("cate", "categorical")
            col_meta_str = col_meta_str.replace("quant", "quantitative")
            meta_info_list.append(col_meta_str)
    
    return meta_info_list


def get_stressqa_example(task_family: str):
    """
    Get a few-shot example for StressQA based on task family.
    These examples demonstrate the enhanced JSON contract.
    """
    examples = {
        'Group Comparison': '''# Column Information: 
column_header: sepal length (cm); data_type: quantitative; num_of_rows: 150; is_normality: True.
column_header: target; data_type: categorical; num_of_rows: 150; is_normality: False.
# Statistical Question: Is there a difference in sepal length between different iris species?
# Correct Answer: {
  "columns": ["sepal length (cm)", "target"],
  "methods": ["One-Way ANOVA"],
  "applicability": true,
  "checks": {"normality": true, "equal_variance": true},
  "warnings": [],
  "test_result": {"statistic": 119.26, "df": "2, 147", "p_value": 1.67e-31},
  "effect_size": {"value": 0.619, "type": "eta_squared"},
  "ci": null,
  "post_hoc": {"recommended": ["Tukey HSD"], "reason": "Omnibus test significant"},
  "corrections": null,
  "audit_trail": {
    "prerequisite_checks": "Checked normality (Shapiro-Wilk p>0.05 in all groups) and homogeneity of variance (Levene p=0.24)",
    "method_choice_reason": "Selected one-way ANOVA for comparing means across 3 independent groups with normal distributions",
    "alternatives_rejected": "Kruskal-Wallis not needed; parametric assumptions met"
  }
}''',
        
        'Correlation Analysis': '''# Column Information: 
column_header: TV Ad Budget ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
column_header: Sales ($); data_type: quantitative; num_of_rows: 200; is_normality: False.
# Statistical Question: Is there a linear correlation between the TV advertising budget ($) and sales revenue ($)?
# Correct Answer: {
  "columns": ["TV Ad Budget ($)", "Sales ($)"],
  "methods": ["Pearson Correlation Coefficient", "Spearman Correlation Coefficient"],
  "applicability": true,
  "checks": {"normality": false, "linear_relationship": true},
  "warnings": ["Normality assumption may be violated; Spearman provides robust alternative"],
  "test_result": {"statistic": 0.782, "df": null, "p_value": 2.0e-16},
  "effect_size": {"value": 0.782, "type": "r"},
  "ci": {"lower": 0.72, "upper": 0.83, "level": 0.95},
  "post_hoc": null,
  "corrections": null,
  "audit_trail": {
    "prerequisite_checks": "Checked linearity via scatter plot; normality via Shapiro-Wilk (p<0.05)",
    "method_choice_reason": "Selected both Pearson and Spearman; Pearson for linear relationship, Spearman as robust alternative",
    "alternatives_rejected": "Kendall not selected as Spearman is more commonly used"
  }
}''',
        
        'Regression': '''# Column Information: 
column_header: age; data_type: quantitative; num_of_rows: 442; is_normality: False.
column_header: bmi; data_type: quantitative; num_of_rows: 442; is_normality: False.
column_header: target; data_type: quantitative; num_of_rows: 442; is_normality: False.
# Statistical Question: How do age and BMI predict diabetes progression?
# Correct Answer: {
  "columns": ["age", "bmi"],
  "methods": ["Multiple Linear Regression"],
  "applicability": true,
  "checks": {"normality": false, "linear_relationship": true, "homoscedasticity": true},
  "warnings": ["Residuals may not be normally distributed"],
  "test_result": {"statistic": 45.2, "df": "2, 439", "p_value": 1.2e-18},
  "effect_size": {"value": 0.171, "type": "r_squared"},
  "ci": null,
  "post_hoc": null,
  "corrections": null,
  "audit_trail": {
    "prerequisite_checks": "Checked linearity, normality of residuals (Shapiro-Wilk p<0.05), homoscedasticity (Breusch-Pagan p=0.12)",
    "method_choice_reason": "Selected multiple linear regression for continuous outcome with multiple continuous predictors",
    "alternatives_rejected": "ANOVA not applicable (continuous outcome); non-linear models not needed (linearity assumption met)"
  }
}''',
        
        'Multiple Testing': '''# Column Information: 
column_header: endpoint_1; data_type: quantitative; num_of_rows: 100; is_normality: True.
column_header: endpoint_2; data_type: quantitative; num_of_rows: 100; is_normality: True.
column_header: endpoint_3; data_type: quantitative; num_of_rows: 100; is_normality: True.
column_header: group; data_type: categorical; num_of_rows: 100; is_normality: False.
# Statistical Question: Compare 3 outcomes (endpoint_1, endpoint_2, endpoint_3) between group groups.
# Correct Answer: {
  "columns": ["endpoint_1", "endpoint_2", "endpoint_3", "group"],
  "methods": ["Multiple Endpoints with Benjamini-Hochberg"],
  "applicability": true,
  "checks": {"normality": true, "independence": true},
  "warnings": ["Multiple comparisons require correction to control false discovery rate"],
  "test_result": null,
  "effect_size": null,
  "ci": null,
  "post_hoc": null,
  "corrections": "Benjamini-Hochberg FDR correction applied (alpha=0.05)",
  "audit_trail": {
    "prerequisite_checks": "Checked normality for each endpoint (Shapiro-Wilk p>0.05), independence of observations",
    "method_choice_reason": "Selected BH FDR correction as it is less conservative than Bonferroni while controlling false discovery rate",
    "alternatives_rejected": "Bonferroni rejected (too conservative); Holm-Bonferroni considered but BH preferred for FDR control"
  }
}'''
    }
    
    return examples.get(task_family, examples['Group Comparison'])  # Default to Group Comparison


def prompt_organization_stressqa(row_index, df, curr_dataset: str, trick: str) -> str:
    """
    Organize prompt for StressQA with enhanced JSON contract.
    
    Args:
        row_index: Index of the row in df
        df: DataFrame containing benchmark data
        curr_dataset: Name of the dataset
        trick: Prompting trick (zero-shot, one-shot, etc.)
    
    Returns:
        Formatted prompt string
    """
    # Extract question
    refined_question = extract_refined_question(row_index, df)
    
    # Check if materialized_dataset_path is available
    materialized_path = None
    if 'materialized_dataset_path' in df.columns:
        materialized_path = df.at[row_index, 'materialized_dataset_path']
        if pd.isna(materialized_path) or materialized_path == '':
            materialized_path = None
    
    # Get column metadata (prefer materialized dataset if available)
    meta_df = get_dataset_metadata(curr_dataset, materialized_path)
    meta_info_list = format_column_metadata(meta_df)
    
    # Get task family for example selection
    task_family = df.at[row_index, 'task'] if 'task' in df.columns else 'Group Comparison'
    
    # Get example for this task family
    stressqa_example = get_stressqa_example(task_family)
    
    # Generate prompt based on trick
    if trick == 'zero-shot':
        organized_prompt = "### Task Description: " + STRESSQA_TASK_DESCRIPTION \
                + "\n### Instruction: " + STRESSQA_INSTRUCTION \
                + "\n### Classification List: \n" + PROMPT_CLASSIFICATION \
                + "\n### Column Information: \n" + '\n'.join(meta_info_list) \
                + "\n### Statistical Question: " + refined_question \
                + "\n### Response: " + STRESSQA_RESPONSE
    
    elif trick == 'one-shot':
        organized_prompt = "### Task Description: " + STRESSQA_TASK_DESCRIPTION \
                + "\n### Instruction: " + STRESSQA_INSTRUCTION \
                + "\n### Classification List: \n" + PROMPT_CLASSIFICATION \
                + "\n### Demonstration Example:\n<example start>\n" + stressqa_example + "\n</example end>" \
                + "\n### Column Information: \n" + '\n'.join(meta_info_list) \
                + "\n### Statistical Question: " + refined_question \
                + "\n### Response: " + STRESSQA_RESPONSE
    
    elif trick == 'two-shot':
        # Get two examples (can mix task families)
        example1 = stressqa_example
        example2 = get_stressqa_example('Correlation Analysis') if task_family != 'Correlation Analysis' else get_stressqa_example('Regression')
        
        organized_prompt = "### Task Description: " + STRESSQA_TASK_DESCRIPTION \
                + "\n### Instruction: " + STRESSQA_INSTRUCTION \
                + "\n### Classification List: \n" + PROMPT_CLASSIFICATION \
                + "\n### Here are two demonstration examples:\n<example start>" \
                + "\n## Demonstration Example No.1:\n" + example1 \
                + "\n## Demonstration Example No.2:\n" + example2 + "\n</example end>" \
                + "\n### Column Information: \n" + '\n'.join(meta_info_list) \
                + "\n### Statistical Question: " + refined_question \
                + "\n### Response: " + STRESSQA_RESPONSE
    
    elif trick == 'zero-shot-CoT':
        organized_prompt = "### Task Description: " + STRESSQA_TASK_DESCRIPTION \
                + "\n### Instruction: " + STRESSQA_INSTRUCTION \
                + "\n### Classification List: \n" + PROMPT_CLASSIFICATION \
                + "\n### Column Information: \n" + '\n'.join(meta_info_list) \
                + "\n### Statistical Question: " + refined_question \
                + "\n### Response: " + PROMPT_COT + STRESSQA_RESPONSE
    
    elif trick == 'one-shot-CoT':
        organized_prompt = "### Task Description: " + STRESSQA_TASK_DESCRIPTION \
                + "\n### Instruction: " + STRESSQA_INSTRUCTION \
                + "\n### Classification List: \n" + PROMPT_CLASSIFICATION \
                + "\n### Demonstration Example:\n<example start>\n" + stressqa_example + "\n</example end>" \
                + "\n### Column Information: \n" + '\n'.join(meta_info_list) \
                + "\n### Statistical Question: " + refined_question \
                + "\n### Response: " + PROMPT_COT + STRESSQA_RESPONSE
    
    elif trick == 'stats-prompt':
        organized_prompt = "### Task Description: " + STRESSQA_TASK_DESCRIPTION \
                + "\n### Instruction: " + STRESSQA_INSTRUCTION \
                + "\n### Classification List: \n" + STRESSQA_STATS_PROMPT \
                + "\n### Demonstration Example:\n<example start>\n" + stressqa_example + "\n</example end>" \
                + "\n### Column Information: \n" + '\n'.join(meta_info_list) \
                + "\n### Statistical Question: " + refined_question \
                + "\n### Response: " + STRESSQA_RESPONSE
    
    else:
        raise ValueError("[!] Invalid trick: " + trick)
    
    return organized_prompt


def main():
    parser = argparse.ArgumentParser(description='Organize StressQA prompts with enhanced JSON contract.')
    parser.add_argument('--trick_name', default='zero-shot', type=str, required=True,
                        help="The trick to be applied. Available: 'zero-shot', 'one-shot', 'two-shot', 'zero-shot-CoT', 'one-shot-CoT', 'stats-prompt'.")
    parser.add_argument('--integ_dataset_name', default='StressQA', type=str, required=True,
                        help='The name of the integrated dataset (e.g., StressQA, mini-StressQA).')
    
    args = parser.parse_args()
    
    trick_name = args.trick_name
    integ_dataset_name = args.integ_dataset_name
    
    # Determine output path
    set_path = ''
    if integ_dataset_name.endswith('StatQA') or integ_dataset_name.endswith('test') or integ_dataset_name == 'StressQA':
        set_path = path.test_set_path
    elif integ_dataset_name.endswith('train'):
        set_path = path.training_set_path
    elif integ_dataset_name == 'Balanced Benchmark':
        set_path = ''
    else:
        # For StressQA, default to test set
        set_path = path.test_set_path
    
    # Input file path
    file_path = path.integ_dataset_path + path.balance_path + integ_dataset_name + '.csv'
    
    # Output file path
    output_file_path = path.integ_dataset_path + path.prompt_dataset_path + set_path + integ_dataset_name + f' for {trick_name}.csv'
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    print(f"[*] Organizing StressQA prompts...")
    print(f"    Input: {file_path}")
    print(f"    Output: {output_file_path}")
    print(f"    Trick: {trick_name}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"[*] Loaded {len(df)} rows")
        
        # Add prompt column
        df['prompt'] = ''
        
        # Generate prompts
        for row_index, row in df.iterrows():
            try:
                curr_dataset_name = df['dataset'].iloc[row_index]
                prompt = prompt_organization_stressqa(row_index, df, curr_dataset_name, trick_name)
                df.at[row_index, 'prompt'] = prompt
                
                if (row_index + 1) % 10 == 0:
                    print(f"    Processed {row_index + 1}/{len(df)} rows...")
                    
            except Exception as e:
                print(f"[!] Error processing row {row_index}: {e}")
                df.at[row_index, 'prompt'] = "Error!"
        
        # Select columns to keep (StressQA has additional columns)
        existing_cols = ['dataset', 'refined_question', 'relevant_column', 'task', 'difficulty', 'results']
        new_cols = ['analysis_spec_id', 'difficulty_axes', 'acceptable_methods', 'oracle', 'is_applicable']
        
        # Keep all columns that exist
        cols_to_keep = [c for c in existing_cols + new_cols if c in df.columns] + ['prompt']
        df_output = df[cols_to_keep]
        
        # Save
        df_output.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"[+] Prompts organized successfully!")
        print(f"[+] Saved: {output_file_path}")
        print(f"[+] Total prompts: {len(df_output)}")
        
    except Exception as e:
        print(f"[!] Error during prompt organization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

