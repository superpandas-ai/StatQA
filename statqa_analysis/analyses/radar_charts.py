# -*- coding: utf-8 -*-
"""
Radar chart generation for cohort analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import load_csv
from ..mappings import task_abbreviations


class RadarChartGeneration(BaseAnalysis):
    """
    Generates radar charts for cohort analysis.
    """
    
    @property
    def name(self) -> str:
        return "radar_charts"
    
    @property
    def requires(self) -> list:
        return []  # Operates on cohort-level summary files
    
    @property
    def produces(self) -> list:
        return ["radar_chart_overall"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Generate radar charts from summary tables."""
        # This will be called from CohortAnalyzer when appropriate
        return context
    
    @staticmethod
    def generate_radar_chart(
        summary_csv_path: Path,
        output_path: Path,
        title: str = "Performance Radar Chart",
        dpi: int = 300
    ):
        """
        Generate a radar chart from a summary CSV file.
        
        Args:
            summary_csv_path: Path to summary CSV (e.g., overall_selection_summary_performance.csv)
            output_path: Path to save the radar chart
            title: Chart title
            dpi: Plot DPI
        """
        # Load data
        data = pd.read_csv(summary_csv_path, skip_blank_lines=False)
        
        # Identify blocks by empty rows (for subplots)
        block_indices = data.index[data.isna().all(axis=1)].tolist()
        block_starts = [0] + [index + 1 for index in block_indices]
        block_ends = block_indices + [len(data)]
        
        # Number of blocks (subplots)
        num_blocks = len(block_starts)
        
        if num_blocks == 0:
            print("[!] No data blocks found in summary CSV")
            return
        
        # Setup figure
        if num_blocks > 1:
            fig, axes = plt.subplots(1, num_blocks, figsize=(4 * num_blocks, 4), subplot_kw=dict(polar=True))
            if num_blocks == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(polar=True))
            axes = [axes]
        
        # Determine global color mapping
        unique_models = pd.concat([
            data.iloc[start:end].dropna()['model'] 
            for start, end in zip(block_starts, block_ends)
        ]).unique()
        
        color_map = {model: plt.cm.tab20(i / len(unique_models)) for i, model in enumerate(unique_models)}
        
        # Plot each block
        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            block_df = data.iloc[start:end].dropna()
            ax = axes[i] if num_blocks > 1 else axes[0]
            
            # Plot each model in the block
            for index, row in block_df.iterrows():
                model_name = row['model']
                scores = row[2:]  # Skip 'model' and 'overall_score_rate'
                
                # Map tasks to abbreviations
                tasks = [
                    task_abbreviations.get(task, task) 
                    for task in scores.index.tolist()
                ]
                
                num_vars = len(tasks)
                if num_vars == 0:
                    continue
                
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
                values = scores.values.tolist() + [scores.values[0]]
                color = color_map[model_name]
                
                ax.fill(angles, values, color=color, alpha=0.2)
                ax.plot(angles, values, color=color, linewidth=2, label=model_name)
            
            # Configure axis
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(tasks, fontsize=14)
            ax.set_ylim(0, 1)
        
        # Global legend
        legend_elements = [
            Line2D([0], [0], color=color_map[model], linewidth=2.5, 
                   label=model.replace('_', ' ').replace('zero', '0').replace('one', '1').replace('two', '2'))
            for model in unique_models
        ]
        
        plt.suptitle(title, fontsize=16, y=1.02)
        
        # Adjust layout to make room for legend on the right
        # Leave space on the right side for the legend
        plt.tight_layout(rect=[0, 0, 0.80, 0.98])  # Leave 20% space on right for legend
        
        # Place legend outside the plot area (to the right)
        fig.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.00, 0.5),  # Position legend closer to the plot
            fontsize=12,
            frameon=True
        )
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format=output_path.suffix[1:], bbox_inches='tight', dpi=dpi)
        plt.close()
        
        print(f"[+] Radar chart saved to {output_path}")

