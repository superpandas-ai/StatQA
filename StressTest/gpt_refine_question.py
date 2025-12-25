# -*- coding: utf-8 -*-
"""
Optional GPT Question Refinement for StressQA

This module provides optional GPT-based question refinement for StressQA benchmark.
Unlike StatQA, this is OPTIONAL and can be skipped to keep StressQA API-token-free.

Usage:
    - Set enable_refinement=True in benchmark generator to use
    - Or call refine_stressqa_questions() directly on a benchmark file
"""

import sys
import os
main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

import requests
import json
import pandas as pd
import time
import path as path_config
from typing import Optional


# [ATTENTION]: The config file is confidential and can NOT be uploaded or backed up!
# Try to read config, but handle gracefully if not available
_config_list = None
try:
    config_file = open('gpt_config.txt', 'r')
    config_content = config_file.readlines()
    config_file.close()
    _config_list = [line.strip() for line in config_content]  # [url, authorization]
except FileNotFoundError:
    print("[!] Warning: gpt_config.txt not found. GPT refinement will not be available.")
    _config_list = None


# Set prompt template (same as StatQA)
SYSTEM_PROMPT = "I'm a native English-speaking statistician. I will help you refine and improve expressions of statistical sentences without changing the original meaning. Please tell me the sentence you want to refine."

INSTRUCTION_PROMPT = '''Suppose you're a native English-speaking statistician, and I will give you a sentence about a statistical problem. You need to improve the English expression of the given sentence to make it grammatically and semantically correct, statistically rigorous and more coherent in expression. The given sentence will contain the names of the variables to be analyzed, and based on the description, you are encouraged to think carefully and change them to more natural expressions without affecting the meaning. You can be flexible in how you improve the expression, but you must not change the original meaning of the sentence.'''


def is_refinement_available() -> bool:
    """Check if GPT refinement is available (config file exists)."""
    return _config_list is not None


def refine_question_gpt(question: str, variable_descriptions: Optional[str] = None) -> str:
    """
    Refine a question using GPT API.
    
    Args:
        question: The question to refine
        variable_descriptions: Optional string describing variables (e.g., "dv: investment; iv: group")
    
    Returns:
        Refined question string, or 'Error' if refinement fails
    """
    if not is_refinement_available():
        return question  # Return original if refinement not available
    
    url = _config_list[0]
    headers = {
        "Content-Type": "application/json",
        "Authorization": _config_list[1]
    }
    
    # Build user prompt
    if variable_descriptions:
        user_prompt = INSTRUCTION_PROMPT + f'\nVariable description: {variable_descriptions}.' + '\nSentence: {}'
    else:
        user_prompt = INSTRUCTION_PROMPT + '\nSentence: {}'
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt.format(question)}
        ],
        "temperature": 0.7
    }
    
    # 3 attempts in case of internet failure
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                result = response.json()
                refined = result['choices'][0]['message']['content']
                # Remove excessive whitespace and returned prompt structures
                refined = refined.replace('\n', '').replace('The revised sentence is: ', '')
                return refined
            else:
                print(f"[!] Error: {response.status_code} at attempt {attempt+1}.")
        except Exception as e:
            print(f"[!] Error: {e}")
        if attempt < 2:
            time.sleep(10)  # Wait 10 seconds before next attempt
    
    return 'Error'  # Return 'Error' if all attempts fail


def extract_variable_descriptions_from_row(row: pd.Series) -> Optional[str]:
    """
    Extract variable descriptions from a StressQA benchmark row.
    
    Args:
        row: A row from the StressQA benchmark DataFrame
    
    Returns:
        String with variable descriptions, or None if not available
    """
    try:
        # Parse relevant_column JSON
        relevant_cols = json.loads(row['relevant_column'])
        
        # Try to get metadata for the dataset
        dataset_name = row['dataset']
        try:
            import utils
            metadata_df = utils.get_metadata(dataset_name)
            
            # Build description string
            descriptions = []
            for col_info in relevant_cols:
                col_name = col_info.get('column_header', '')
                # Find description in metadata
                col_meta = metadata_df[metadata_df['column_header'] == col_name]
                if not col_meta.empty and 'column_description' in col_meta.columns:
                    desc = col_meta['column_description'].iloc[0]
                    if pd.notna(desc) and desc.strip():
                        descriptions.append(f"{col_name}: {desc}")
                    else:
                        descriptions.append(col_name)
                else:
                    descriptions.append(col_name)
            
            return '; '.join(descriptions) if descriptions else None
            
        except Exception:
            # If metadata not available, just use column names
            col_names = [col_info.get('column_header', '') for col_info in relevant_cols]
            return '; '.join(col_names) if col_names else None
            
    except Exception as e:
        print(f"[!] Warning: Could not extract variable descriptions: {e}")
        return None


def refine_stressqa_questions(
    benchmark_file: str,
    output_file: Optional[str] = None,
    batch_size: int = 20,
    skip_if_no_config: bool = True
) -> bool:
    """
    Refine questions in a StressQA benchmark file using GPT.
    
    Args:
        benchmark_file: Path to StressQA benchmark CSV/JSON file
        output_file: Optional output file path (default: overwrites original)
        batch_size: Number of questions to process before saving
        skip_if_no_config: If True, skip refinement if config not available (default: True)
    
    Returns:
        True if refinement completed, False if skipped or failed
    """
    if not is_refinement_available():
        if skip_if_no_config:
            print("[i] GPT config not available. Skipping question refinement.")
            print("    (StressQA works fine without refinement - this is optional)")
            return False
        else:
            print("[!] Error: GPT config not available but refinement was requested.")
            return False
    
    print("=" * 60)
    print("StressQA Question Refinement (OPTIONAL)")
    print("=" * 60)
    print("[!] This step uses API tokens. Press Ctrl+C to cancel.")
    time.sleep(3)
    
    # Load benchmark
    if benchmark_file.endswith('.json'):
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        df = pd.DataFrame(benchmark_data)
    else:
        df = pd.read_csv(benchmark_file)
    
    if 'refined_question' not in df.columns:
        # If no refined_question column, use the question as base
        if 'question' in df.columns:
            df['refined_question'] = df['question']
        else:
            print("[!] Error: No 'question' or 'refined_question' column found.")
            return False
    
    total_rows = len(df)
    print(f"[*] Refining {total_rows} questions...")
    
    # Process in batches
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        
        for idx in range(i, end_idx):
            row = df.iloc[idx]
            original_question = row['refined_question'] if pd.notna(row['refined_question']) else row.get('question', '')
            
            if not original_question or original_question == 'Error':
                continue
            
            # Extract variable descriptions
            var_descriptions = extract_variable_descriptions_from_row(row)
            
            # Refine question
            refined = refine_question_gpt(original_question, var_descriptions)
            df.at[idx, 'refined_question'] = refined
            
            print(f"[+] Row {idx + 1}/{total_rows}: {original_question[:50]}... -> {refined[:50]}...")
        
        # Save batch
        output_path = output_file or benchmark_file
        # Determine format from output path extension
        if output_path.endswith('.json'):
            # Save as JSON
            json_data = df.to_json(orient='records', indent=4, force_ascii=False)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
        else:
            # Save as CSV
            df.to_csv(output_path, index=False)
        
        print(f"[+] Saved batch {i//batch_size + 1} (rows {i+1}-{end_idx})")
    
    print("\n[+] Question refinement complete!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optional GPT question refinement for StressQA"
    )
    parser.add_argument(
        '--benchmark-file',
        type=str,
        required=True,
        help='Path to StressQA benchmark file (CSV or JSON)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Output file path (default: overwrites input)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Number of questions to process before saving (default: 20)'
    )
    
    args = parser.parse_args()
    
    refine_stressqa_questions(
        args.benchmark_file,
        args.output_file,
        args.batch_size,
        skip_if_no_config=True
    )

