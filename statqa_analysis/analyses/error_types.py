# -*- coding: utf-8 -*-
"""
Error type analysis.
"""

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path to import mappings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import mappings

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import save_csv


class ErrorTypeAnalysis(BaseAnalysis):
    """
    Analyzes types of errors made by the model:
    - Invalid answers
    - Column selection errors
    - Statistical task confusion
    - Applicability errors
    - Mixed errors
    """
    
    @property
    def name(self) -> str:
        return "error_type_analysis"
    
    @property
    def requires(self) -> list:
        return ["df_with_scores"]
    
    @property
    def produces(self) -> list:
        return ["error_analysis_summary"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Analyze error types."""
        if not context.config.enable_error_analysis:
            print("[i] Error analysis disabled, skipping")
            return context
        
        df = context.df.copy()
        output_dir = context.config.get_run_output_dir() / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Analyzing error types...")
        
        # Build list of all valid methods (flattened, lowercase)
        methods_list = [
            method.lower() 
            for methods in mappings.tasks_to_methods.values() 
            for method in methods
        ]
        
        # Initialize counters
        invalid_answers = 0
        column_errors = 0
        statistical_confusion = 0
        applicability_errors = 0
        mixed_errors_column_statistical = 0
        mixed_errors_column_applicability = 0
        mixed_errors_statistical_applicability = 0
        mixed_errors_all = 0
        
        total_cnt = len(df)
        
        for _, row in df.iterrows():
            # Check if extracted_answer contains any valid method
            extracted_str = str(row['extracted_answer']).lower()
            has_valid_method = any(method in extracted_str for method in methods_list)
            
            if not has_valid_method:
                invalid_answers += 1
            else:
                # Check for specific error types
                errors = {
                    'column_error': row['columns_score'] == 0,
                    'statistical_confusion': False,
                    'applicability_error': False
                }
                
                # Analyze methods comparison result
                try:
                    methods_result = json.loads(row['methods_comparison_result'])
                    correct = methods_result.get('Correct', 0)
                    wrong_missed = methods_result.get('Wrong', 0) + methods_result.get('Missed', 0)
                    
                    if correct == 0 and wrong_missed > 0:
                        errors['statistical_confusion'] = True
                    if correct > 0 and wrong_missed > 0:
                        errors['applicability_error'] = True
                except (json.JSONDecodeError, TypeError):
                    errors['statistical_confusion'] = True
                
                # Count error types
                error_count = sum(errors.values())
                
                if error_count == 1:
                    if errors['column_error']:
                        column_errors += 1
                    elif errors['statistical_confusion']:
                        statistical_confusion += 1
                    elif errors['applicability_error']:
                        applicability_errors += 1
                elif error_count == 2:
                    if errors['column_error'] and errors['statistical_confusion']:
                        mixed_errors_column_statistical += 1
                    elif errors['column_error'] and errors['applicability_error']:
                        mixed_errors_column_applicability += 1
                    elif errors['statistical_confusion'] and errors['applicability_error']:
                        mixed_errors_statistical_applicability += 1
                elif error_count == 3:
                    mixed_errors_all += 1
        
        # Create summary
        run_id = context.config.run_id or "default"
        summary = {
            'Model': run_id,
            'Invalid Answer': round(invalid_answers / total_cnt, 5) if total_cnt else 0,
            'Column Selection Error (CSE)': round(column_errors / total_cnt, 5) if total_cnt else 0,
            'Statistical Task Confusion (STC)': round(statistical_confusion / total_cnt, 5) if total_cnt else 0,
            'Applicability Error (AE)': round(applicability_errors / total_cnt, 5) if total_cnt else 0,
            'Mixed Errors (CSE+STC)': round(mixed_errors_column_statistical / total_cnt, 5) if total_cnt else 0,
            'Mixed Errors (CSE+AE)': round(mixed_errors_column_applicability / total_cnt, 5) if total_cnt else 0,
            'Mixed Errors (STC+AE)': round(mixed_errors_statistical_applicability / total_cnt, 5) if total_cnt else 0,
            'Mixed Errors (CSE+STC+AE)': round(mixed_errors_all / total_cnt, 5) if total_cnt else 0
        }
        
        # Save to CSV
        summary_df = pd.DataFrame([summary])
        output_path = output_dir / "error_analysis_summary.csv"
        save_csv(summary_df, output_path)
        
        context.add_artifact("error_analysis_summary", output_path)
        context.add_result("error_analysis_summary", True)
        print(f"[+] Error analysis saved to {output_path}")
        
        # Generate bar chart if plots are enabled
        if context.config.enable_plots:
            self._generate_error_bar_chart(summary_df, context)
        
        return context
    
    def _generate_error_bar_chart(self, summary_df: pd.DataFrame, context: AnalysisContext):
        """Generate stacked bar chart for error analysis."""
        output_dir = context.config.get_run_output_dir() / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Generating error analysis bar chart...")
        
        # Error types in order
        error_types = [
            'Invalid Answer',
            'Column Selection Error (CSE)',
            'Statistical Task Confusion (STC)',
            'Applicability Error (AE)',
            'Mixed Errors (CSE+STC)',
            'Mixed Errors (CSE+AE)',
            'Mixed Errors (STC+AE)',
            'Mixed Errors (CSE+STC+AE)'
        ]
        
        colors = ['#FED9A6', '#fbb4ae', '#b3cde3', '#ccebc5', '#decbe4', '#f3f0ba', '#bce7d7', '#d3cfc7']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get model name
        model_name = summary_df['Model'].iloc[0]
        # Clean up model name for display
        display_name = model_name.replace('_', ' ').replace('zero', '0').replace('one', '1').replace('two', '2')
        
        # Create stacked bar
        bottom = 0
        for j, error_type in enumerate(error_types):
            value = summary_df[error_type].iloc[0]
            ax.bar([display_name], [value], bottom=[bottom], color=colors[j], 
                  edgecolor='white', width=0.6, label=error_type)
            
            # Add value labels if significant
            if value > 0.02:
                ax.text(0, bottom + value / 2, f'{value:.2f}', 
                       ha='center', va='center', fontsize=12, fontweight='bold')
            elif value > 0.01:
                ax.text(0, bottom + value / 2, f'{value:.2f}', 
                       ha='center', va='center', fontsize=10)
            
            bottom += value
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Error Rate', fontsize=14)
        ax.set_title(f'Error Type Analysis: {display_name}', fontsize=16)
        ax.tick_params(axis='x', labelsize=12)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, ncol=2)
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"error_analysis_bar_chart.{context.config.plot_format}"
        plt.savefig(output_path, format=context.config.plot_format, bbox_inches='tight', dpi=context.config.plot_dpi)
        plt.close()
        
        context.add_artifact("error_analysis_bar_chart", output_path)
        print(f"[+] Error analysis bar chart saved to {output_path}")

