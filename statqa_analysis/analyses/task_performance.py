# -*- coding: utf-8 -*-
"""
Analyze performance for each task type.
"""

import pandas as pd
from pathlib import Path
from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import save_csv


class TaskPerformance(BaseAnalysis):
    """
    Analyzes performance broken down by task type for methods, columns, and overall.
    """
    
    @property
    def name(self) -> str:
        return "task_performance"
    
    @property
    def requires(self) -> list:
        return ["df_with_scores"]
    
    @property
    def produces(self) -> list:
        return ["task_performance_methods", "task_performance_columns", "task_performance_overall"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Analyze performance for each task type."""
        df = context.df.copy()
        output_dir = context.config.get_run_output_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Analyzing performance by task type...")
        
        # Analyze for methods, columns, and overall
        for target in ['methods', 'columns', 'overall']:
            output_path = self._analyze_for_target(df, target, output_dir)
            context.add_artifact(f"task_performance_{target}", output_path)
            context.add_result(f"task_performance_{target}", True)
        
        print("[+] Task performance analysis completed")
        
        return context
    
    def _analyze_for_target(self, df: pd.DataFrame, target: str, output_dir: Path) -> Path:
        """Analyze performance for a specific target (methods/columns/overall)."""
        df_copy = df.copy()
        
        if target == 'methods':
            # For methods: full score is 1 per row (using accuracy metric)
            df_copy['full_score'] = 1
            df_copy['valid_score'] = df_copy.apply(
                lambda row: row['methods_score'] 
                if row['methods_comparison_result'] != 'Invalid Answer' else 0,
                axis=1
            )
        elif target == 'columns':
            # For columns: full score is 1 per row
            df_copy['full_score'] = 1
            df_copy['valid_score'] = df_copy.apply(
                lambda row: row['columns_score'] 
                if row['columns_comparison_result'] != 'Invalid Answer' else 0,
                axis=1
            )
        elif target == 'overall':
            # For overall selection: full score is 1 per row
            df_copy['full_score'] = 1
            df_copy['valid_score'] = df_copy.apply(
                lambda row: row['selection_overall'] 
                if row['methods_comparison_result'] != 'Invalid Answer' else 0,
                axis=1
            )
        else:
            raise ValueError(f"[!] Invalid target: {target}")
        
        # Group by task and calculate statistics
        task_stats = df_copy.groupby('task').agg(
            total_full_score=pd.NamedAgg(column='full_score', aggfunc='sum'),
            obtained_valid_score=pd.NamedAgg(column='valid_score', aggfunc='sum')
        ).reset_index()
        
        task_stats['score_rate'] = round(
            task_stats['obtained_valid_score'] / task_stats['total_full_score'], 
            5
        )
        
        # Save to file
        output_path = output_dir / f"task_performance_{target}.csv"
        save_csv(task_stats, output_path)
        print(f"[+] Task performance ({target}) saved to {output_path}")
        
        return output_path

