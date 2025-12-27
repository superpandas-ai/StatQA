# -*- coding: utf-8 -*-
"""
Calculate scores for methods and columns selection.
"""

import json
from ..pipeline import BaseAnalysis
from ..config import AnalysisContext


class Scoring(BaseAnalysis):
    """
    Calculates scores for methods and columns selection.
    """
    
    @property
    def name(self) -> str:
        return "scoring"
    
    @property
    def requires(self) -> list:
        return ["df_with_comparison"]
    
    @property
    def produces(self) -> list:
        return ["df_with_scores"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Calculate scores for methods and columns."""
        df = context.df.copy()
        method_metric = context.config.method_metric
        
        print(f"[*] Calculating scores (method metric: {method_metric})...")
        
        # Calculate methods score
        df['methods_score'] = df.apply(
            lambda row: self._calculate_score(
                row, 'methods', method_metric
            ),
            axis=1
        )
        
        # Calculate columns score
        df['columns_score'] = df.apply(
            lambda row: self._calculate_score(
                row, 'columns', 'acc'
            ),
            axis=1
        )
        
        # Calculate overall selection score
        df['selection_overall'] = df.apply(
            lambda row: 1 if (row['methods_score'] == 1 and row['columns_score'] == 1) else 0,
            axis=1
        )
        
        # Calculate summary statistics
        total_rows = len(df)
        methods_acc = df['methods_score'].sum() / total_rows if total_rows > 0 else 0
        columns_acc = df['columns_score'].sum() / total_rows if total_rows > 0 else 0
        overall_acc = df['selection_overall'].sum() / total_rows if total_rows > 0 else 0
        
        print(f"[+] Scoring completed:")
        print(f"    Methods accuracy: {methods_acc:.4f}")
        print(f"    Columns accuracy: {columns_acc:.4f}")
        print(f"    Overall accuracy: {overall_acc:.4f}")
        
        context.df = df
        context.add_result("df_with_scores", True)
        context.add_result("methods_accuracy", methods_acc)
        context.add_result("columns_accuracy", columns_acc)
        context.add_result("overall_accuracy", overall_acc)
        
        return context
    
    def _calculate_score(self, row, target: str, method_metric: str = 'acc') -> float:
        """Calculate score for a single row."""
        if target == 'methods':
            target_header = 'methods_comparison_result'
        elif target == 'columns':
            target_header = 'columns_comparison_result'
        else:
            raise ValueError(f"[!] Invalid target: {target}")
        
        comparison_result = row[target_header]
        
        # Check if the comparison_result is a JSON string
        if not (isinstance(comparison_result, str) and 
                comparison_result.startswith('{') and 
                comparison_result.endswith('}')):
            return 0.0
        
        try:
            result = json.loads(comparison_result)
            correct = result.get('Correct', 0)
            wrong = result.get('Wrong', 0)
            missed = result.get('Missed', 0)
            
            if target == 'methods' and method_metric == 'jaccard':
                # Jaccard index
                denominator = correct + wrong + missed
                if denominator == 0:
                    return 0.0
                return correct / denominator
            else:
                # Accuracy: all correct and no wrong/missed
                if correct > 0 and wrong == 0 and missed == 0:
                    return 1.0
                else:
                    return 0.0
        except (ValueError, json.JSONDecodeError):
            return 0.0

