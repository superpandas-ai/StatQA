# -*- coding: utf-8 -*-
"""
Merge metadata from mini-StatQA.json based on serial_number.
"""

import json
from pathlib import Path
import pandas as pd
from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import safe_parse_json


class MetadataMerge(BaseAnalysis):
    """
    Merges metadata from mini-StatQA.json into the DataFrame based on serial_number.
    Adds columns: task, results, relevant_column, ground_truth, dataset, refined_question, difficulty
    """
    
    @property
    def name(self) -> str:
        return "metadata_merge"
    
    @property
    def requires(self) -> list:
        return ["raw_data"]
    
    @property
    def produces(self) -> list:
        return ["df_with_metadata"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Merge metadata from mini-StatQA.json or StatDatasets."""
        df = context.df.copy()
        
        # Check if serial_number column exists
        if 'serial_number' not in df.columns:
            print("[!] No 'serial_number' column found. Cannot merge metadata from JSON.")
            print("[i] Assuming metadata columns are already present or will be derived from other sources.")
            context.df = df
            context.add_result("df_with_metadata", True)
            return context
        
        # Check if all metadata columns already exist
        required_metadata = ['task', 'results', 'relevant_column', 'ground_truth']
        if all(col in df.columns for col in required_metadata):
            print("[i] All required metadata columns already present, skipping merge")
            context.df = df
            context.add_result("df_with_metadata", True)
            return context
        
        print("[*] Merging metadata from dataset JSON...")
        
        # Try to get dataset_id from config
        dataset_id = context.config.custom_params.get('dataset_id')
        
        if dataset_id:
            # Use StatDatasets/ structure
            json_path = Path("StatDatasets") / "raw" / dataset_id / "data.json"
            if json_path.exists():
                print(f"[i] Using StatDatasets JSON: {json_path}")
            else:
                print(f"[!] StatDatasets JSON not found: {json_path}")
                json_path = None
        else:
            # Fall back to legacy path
            json_path = Path("Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.json")
            if json_path.exists():
                print(f"[i] Using legacy JSON path: {json_path}")
            else:
                print(f"[!] Legacy JSON not found: {json_path}")
                json_path = None
        
        if json_path is None or not json_path.exists():
            print("[!] No metadata JSON found. Skipping metadata merge.")
            print("[i] Hint: Use --dataset-id flag or ensure metadata columns are in input CSV.")
            context.df = df
            context.add_result("df_with_metadata", True)
            return context
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"[!] Error loading metadata JSON: {e}")
            context.df = df
            context.add_result("df_with_metadata", True)
            return context
        
        # Create lookup dictionary by serial_number
        metadata_lookup = {}
        for item in json_data:
            serial_num = item.get('serial_number')
            if serial_num is not None:
                metadata_lookup[serial_num] = {
                    'task': item.get('task'),
                    'results': item.get('results'),
                    'relevant_column': item.get('relevant_column'),
                    'ground_truth': item.get('ground_truth'),
                    'dataset': item.get('dataset'),
                    'refined_question': item.get('refined_question'),
                    'difficulty': item.get('difficulty'),
                }
        
        print(f"[+] Loaded metadata for {len(metadata_lookup)} entries from JSON")
        
        # Merge metadata into DataFrame
        merged_count = 0
        for idx, row in df.iterrows():
            serial_num = row.get('serial_number')
            if serial_num is not None and serial_num in metadata_lookup:
                metadata = metadata_lookup[serial_num]
                
                # Add metadata columns if they don't exist or are missing
                for col, value in metadata.items():
                    if col not in df.columns or pd.isna(row.get(col)):
                        df.at[idx, col] = value
                        merged_count += 1
        
        print(f"[+] Merged metadata for {merged_count} cells")
        
        # Verify required columns
        missing_cols = [col for col in required_metadata if col not in df.columns]
        if missing_cols:
            print(f"[!] Warning: Missing columns after merge: {missing_cols}")
        else:
            print("[+] All required metadata columns present")
        
        context.df = df
        context.add_result("df_with_metadata", True)
        
        return context

