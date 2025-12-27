# -*- coding: utf-8 -*-
"""
Compare extracted answers with ground truth and count correct/wrong/missed.
"""

import json
from ..pipeline import BaseAnalysis
from ..config import AnalysisContext


class CompareAndCount(BaseAnalysis):
    """
    Compares extracted answers with ground truth for methods and columns.
    """
    
    @property
    def name(self) -> str:
        return "compare_and_count"
    
    @property
    def requires(self) -> list:
        return ["df_with_extracted_answer"]
    
    @property
    def produces(self) -> list:
        return ["df_with_comparison"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Compare and count answers for methods and columns."""
        df = context.df.copy()
        
        print("[*] Comparing answers with ground truth...")
        
        # Apply comparison for methods
        df['methods_comparison_result'] = df.apply(
            lambda row: self._compare_and_count_row(
                'methods', row['extracted_answer'], row['ground_truth']
            ),
            axis=1
        )
        
        # Apply comparison for columns
        df['columns_comparison_result'] = df.apply(
            lambda row: self._compare_and_count_row(
                'columns', row['extracted_answer'], row['ground_truth']
            ),
            axis=1
        )
        
        context.df = df
        context.add_result("df_with_comparison", True)
        print("[+] Comparison completed")
        
        return context
    
    def _compare_and_count_row(self, target: str, extracted_answer, ground_truth):
        """Compare and count for a single row."""
        try:
            # Parse model answer
            model_answer_all = json.loads(extracted_answer)
            model_answer_list = model_answer_all.get(target, [])
            
            # Parse ground truth
            ground_truth_all = json.loads(ground_truth.replace('\'', '"'))
            ground_truth_list = ground_truth_all.get(target, [])
            
            # Normalize to lowercase
            model_answer_list = [str(item).lower().strip() for item in model_answer_list]
            ground_truth_list = [str(item).lower().strip() for item in ground_truth_list]
            
            # Calculate correct, wrong, and missed counts
            correct = len(set(model_answer_list) & set(ground_truth_list))
            wrong = len(set(model_answer_list) - set(ground_truth_list))
            missed = len(set(ground_truth_list) - set(model_answer_list))
            
            # Return comparison result as JSON string
            return json.dumps({
                "Correct": correct,
                "Wrong": wrong,
                "Missed": missed
            })
        except Exception:
            return "Invalid Answer"

