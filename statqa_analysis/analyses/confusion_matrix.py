# -*- coding: utf-8 -*-
"""
Task confusion matrix analysis.
"""

import sys
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add parent directory to path to import mappings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mappings

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import safe_literal_eval


class ConfusionMatrixAnalysis(BaseAnalysis):
    """
    Generates confusion matrix for task classification based on methods selected.
    """
    
    @property
    def name(self) -> str:
        return "confusion_matrix"
    
    @property
    def requires(self) -> list:
        return ["df_with_scores"]
    
    @property
    def produces(self) -> list:
        return ["confusion_matrix_plot"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Generate confusion matrix plot."""
        if not context.config.enable_confusion_matrix:
            print("[i] Confusion matrix disabled, skipping")
            return context
        
        df = context.df.copy()
        output_dir = context.config.get_run_output_dir() / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Generating confusion matrix...")
        
        # Determine answer task from extracted methods
        df['extracted_answer_safe'] = df['extracted_answer'].apply(
            lambda x: safe_literal_eval(x) if x != 'Invalid Answer' else []
        )
        
        df['answer_task'] = df['extracted_answer_safe'].apply(
            lambda x: self._determine_task(x) if isinstance(x, dict) and 'methods' in x 
            else self._determine_task({'methods': x}) if isinstance(x, list)
            else None
        )
        
        # Filter valid answers
        df_valid = df[df['answer_task'].notna()].copy()
        
        if len(df_valid) == 0:
            print("[!] No valid answers for confusion matrix, skipping")
            return context
        
        # Generate confusion matrix
        cm = confusion_matrix(
            df_valid['task'], 
            df_valid['answer_task'], 
            labels=list(mappings.tasks_to_methods.keys())
        )
        
        # Plot
        labels = [mappings.task_abbreviations[task] for task in mappings.tasks_to_methods.keys()]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 14}
        )
        plt.xlabel('Selected Tasks', fontsize=14)
        plt.ylabel('Actual Tasks', fontsize=14)
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"confusion_matrix.{context.config.plot_format}"
        plt.savefig(output_path, format=context.config.plot_format, bbox_inches='tight', dpi=context.config.plot_dpi)
        plt.close()
        
        context.add_artifact("confusion_matrix_plot", output_path)
        context.add_result("confusion_matrix_plot", True)
        print(f"[+] Confusion matrix saved to {output_path}")
        
        return context
    
    def _determine_task(self, answer_dict) -> str:
        """Determine task from methods list."""
        if not isinstance(answer_dict, dict):
            return None
        
        method_list = answer_dict.get('methods', [])
        if not method_list:
            return None
        
        # Normalize methods to lowercase
        method_list = [str(m).lower() for m in method_list]
        
        # Count occurrences
        method_count = Counter(method_list)
        task_scores = {task: 0 for task in mappings.tasks_to_methods.keys()}
        
        for method, count in method_count.items():
            for task, methods in mappings.tasks_to_methods.items():
                # Normalize task methods to lowercase for comparison
                methods_lower = [m.lower() for m in methods]
                if method in methods_lower:
                    task_scores[task] += count
        
        # Return task with highest score
        if max(task_scores.values()) == 0:
            return None
        
        return max(task_scores, key=task_scores.get)

