# -*- coding: utf-8 -*-
"""
Derive ground truth from results and relevant_column columns.
"""

import json

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import safe_literal_eval


class GroundTruthDerivation(BaseAnalysis):
    """
    Derives ground_truth column from results and relevant_column.
    Does not mutate input file - only adds column to in-memory DataFrame.
    """
    
    @property
    def name(self) -> str:
        return "ground_truth_derivation"
    
    @property
    def requires(self) -> list:
        return ["df_with_metadata"]
    
    @property
    def produces(self) -> list:
        return ["df_with_ground_truth"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Derive ground truth from results and relevant_column columns."""
        df = context.df.copy()
        
        # Check if ground_truth already exists and is populated
        if 'ground_truth' in df.columns:
            # Check if ground_truth is actually populated (not all NaN)
            if df['ground_truth'].notna().any():
                print("[i] ground_truth column already exists and populated, skipping derivation")
                context.df = df
                context.add_result("df_with_ground_truth", True)
                return context
            else:
                print("[i] ground_truth column exists but is empty, will derive")
        
        # Check if we have the required columns to derive ground_truth
        if 'results' not in df.columns or 'relevant_column' not in df.columns:
            print("[!] Cannot derive ground_truth: missing 'results' or 'relevant_column' columns")
            print("[i] Ensure these columns are present or merge metadata from mini-StatQA.json")
            context.df = df
            context.add_result("df_with_ground_truth", True)
            return context
        
        print("[*] Deriving ground truth from results and relevant_column...")
        
        # Extract methods ground truth
        df['methods_ground_truth'] = df['results'].apply(
            lambda x: self._extract_ground_truth_for_row(x, 'results')
        )
        
        # Extract columns ground truth
        df['columns_ground_truth'] = df['relevant_column'].apply(
            lambda x: self._extract_ground_truth_for_row(x, 'relevant_column')
        )
        
        # Combine into ground_truth column
        df['ground_truth'] = df.apply(
            lambda x: json.dumps({
                "columns": x['columns_ground_truth'],
                "methods": x['methods_ground_truth']
            }),
            axis=1
        )
        
        # Drop intermediate columns
        df.drop(['methods_ground_truth', 'columns_ground_truth'], axis=1, inplace=True)
        
        context.df = df
        context.add_result("df_with_ground_truth", True)
        print(f"[+] Ground truth derived for {len(df)} rows")
        
        return context
    
    def _extract_ground_truth_for_row(self, row_value, col_to_extract: str) -> list:
        """Extract ground truth for a single row."""
        try:
            results = safe_literal_eval(str(row_value))
            if results is None:
                return []
            
            if col_to_extract == "results":
                # Extract 'method' values where 'conclusion' is not "Not applicable"
                ground_truth_list = [
                    result['method'] for result in results 
                    if result.get('conclusion') != "Not applicable"
                ]
            elif col_to_extract == "relevant_column":
                # Extract 'column_header' ground truth
                ground_truth_list = [result['column_header'] for result in results]
            else:
                raise ValueError(f"[!] Invalid column to extract: {col_to_extract}")
            
            return ground_truth_list
        except Exception as e:
            print(f"[!] Error extracting ground truth: {e}")
            return []

