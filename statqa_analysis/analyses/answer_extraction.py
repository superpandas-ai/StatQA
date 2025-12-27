# -*- coding: utf-8 -*-
"""
Extract JSON answers from model output using utils.extract_json_answer.
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext


class AnswerExtraction(BaseAnalysis):
    """
    Extracts JSON answers from model_answer column.
    """
    
    @property
    def name(self) -> str:
        return "answer_extraction"
    
    @property
    def requires(self) -> list:
        return ["df_with_ground_truth"]
    
    @property
    def produces(self) -> list:
        return ["df_with_extracted_answer"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Extract JSON answers from model_answer column, or use existing extracted_answer."""
        df = context.df.copy()
        
        # Check if extracted_answer already exists
        if 'extracted_answer' in df.columns:
            # Check if it's populated
            if df['extracted_answer'].notna().any():
                print("[i] extracted_answer column already exists and populated, skipping extraction")
                context.df = df
                context.add_result("df_with_extracted_answer", True)
                return context
            else:
                print("[i] extracted_answer column exists but is empty, will extract from model_answer")
        
        # Need to extract from model_answer
        if 'model_answer' not in df.columns:
            raise ValueError(
                "[!] No 'model_answer' or 'extracted_answer' column found in input data. "
                "Either provide 'model_answer' for extraction or 'extracted_answer' with pre-extracted data."
            )
        
        print("[*] Extracting JSON answers from model responses...")
        
        # Apply extraction function
        df['extracted_answer'] = df['model_answer'].apply(utils.extract_json_answer)
        
        # Count valid vs invalid answers
        valid_count = (df['extracted_answer'] != 'Invalid Answer').sum()
        invalid_count = len(df) - valid_count
        
        print(f"[+] Extracted answers: {valid_count} valid, {invalid_count} invalid")
        
        context.df = df
        context.add_result("df_with_extracted_answer", True)
        
        return context

