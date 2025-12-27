# -*- coding: utf-8 -*-
"""
Metadata provider for prompt building.
Adapts existing utils.get_metadata functionality.
"""

from pathlib import Path
from ..utils_extract import get_metadata


def get_dataset_metadata(dataset_name: str) -> list:
    """
    Get formatted metadata strings for all columns in a dataset.
    
    Args:
        dataset_name: Name of the dataset (without .csv extension)
        
    Returns:
        List of formatted metadata strings, one per column
    """
    # Get metadata DataFrame
    meta_df = get_metadata(dataset_name=dataset_name)
    
    meta_info_list = []
    for i in range(meta_df.shape[0]):
        row = meta_df.iloc[i]
        col_meta_str = ''
        for header in meta_df.columns:
            # Skip dataset and column_description columns
            if header not in ["dataset", "column_description"]:
                col_meta_str += f"{header}: {row[header]}; "
        
        # Replace semicolon at end with period
        col_meta_str = col_meta_str[:-2] + '.'
        
        # Replace abbreviations
        col_meta_str = col_meta_str.replace("cate", "categorical")
        col_meta_str = col_meta_str.replace("quant", "quantitative")
        
        meta_info_list.append(col_meta_str)
    
    return meta_info_list

