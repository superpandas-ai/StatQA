# -*- coding: utf-8 -*-
"""
Utility functions extracted from repo root utils.py for statqa_analysis independence.
"""

import json
import re
import ast
import pandas as pd
from pathlib import Path


def extract_json_answer(input_string):
    """
    Extract the first balanced JSON object from the input string.
    Supports both inline JSON {...} and fenced ```json blocks.
    
    Args:
        input_string: String containing JSON to extract
        
    Returns:
        str: Valid JSON string or "Invalid Answer"
    """
    # First, try to extract from fenced code block
    json_fence_pattern = r'```json\s*(.*?)\s*```'
    fence_match = re.search(json_fence_pattern, input_string, re.DOTALL)
    
    if fence_match:
        json_candidate = fence_match.group(1).strip()
        try:
            # Validate it's proper JSON
            json.loads(json_candidate)
            return json_candidate
        except ValueError:
            pass  # Fall through to brace-based extraction
    
    # Find the first opening brace
    start_index = input_string.find('{')
    if start_index == -1:
        return "Invalid Answer"
    
    # Now find the matching closing brace using balanced counting
    brace_count = 0
    end_index = -1
    in_string = False
    escape_next = False
    
    for i in range(start_index, len(input_string)):
        char = input_string[i]
        
        # Handle string literals (to avoid counting braces inside strings)
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"':
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        # Count braces
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = i
                break
    
    if end_index == -1:
        return "Invalid Answer"
    
    # Extract the JSON string
    json_str = input_string[start_index:end_index+1]
    
    try:
        # Validate it's proper JSON
        json.loads(json_str)
        return json_str
    except ValueError:
        return "Invalid Answer"


def get_metadata(dataset_name):
    """
    Given a dataset name, returns all information from the corresponding metadata file.
    
    Args:
        dataset_name: The name of the dataset (without .csv extension)
        
    Returns:
        A DataFrame containing the metadata of the dataset.
    """
    # Construct the path to the metadata file based on the dataset name
    metadata_file_path = Path(f'Data/Metadata/Column Metadata/{dataset_name}_col_meta.csv')
    
    # Load and return the metadata file
    try:
        metadata_df = pd.read_csv(metadata_file_path)
        return metadata_df
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {metadata_file_path}")


def extract_ground_truth_for_row(row, col_to_extract: str):
    """
    Extract ground truth of selection of relevant columns or applicable methods from a row.
    
    Args:
        row: String representation of list of dicts (from 'results' or 'relevant_column' column)
        col_to_extract: "results" for methods, or "relevant_column" for relevant columns
        
    Returns:
        List of extracted values (method names or column headers)
    """
    try:
        # Convert the string representation of list of dicts into an actual list of dicts
        results = ast.literal_eval(row.replace("false", "False").replace("true", "True"))
        if col_to_extract == "results":
            # Extract 'method' values where 'conclusion' is not "Not applicable"
            ground_truth_list = [result['method'] for result in results if result['conclusion'] != "Not applicable"]
        elif col_to_extract == "relevant_column":
            # Extract 'column_header' values
            ground_truth_list = [result['column_header'] for result in results]
        else:
            raise ValueError(f"[!] Invalid column to extract: {col_to_extract}. Please use 'results' or 'relevant_column'.")
        return ground_truth_list
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[!] Error decoding or processing row: {e}")
        return []
    except Exception as e:
        print(f"[!] An unexpected error occurred while processing row: {e}")
        return []


def derive_ground_truth(results_str, relevant_column_str):
    """
    Derive ground truth from results and relevant_column strings.
    
    Args:
        results_str: JSON string from 'results' column
        relevant_column_str: JSON string from 'relevant_column' column
        
    Returns:
        dict: Ground truth with 'columns' and 'methods' keys
    """
    methods_ground_truth = extract_ground_truth_for_row(results_str, 'results')
    columns_ground_truth = extract_ground_truth_for_row(relevant_column_str, 'relevant_column')
    
    return {
        "columns": columns_ground_truth,
        "methods": methods_ground_truth
    }

