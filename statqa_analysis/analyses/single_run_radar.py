# -*- coding: utf-8 -*-
"""
Single-run radar chart generation showing performance across tasks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..mappings import task_abbreviations


class SingleRunRadarChart(BaseAnalysis):
    """
    Generates a radar chart for a single run showing performance across tasks.
    """
    
    @property
    def name(self) -> str:
        return "single_run_radar_chart"
    
    @property
    def requires(self) -> list:
        return ["task_performance_overall"]
    
    @property
    def produces(self) -> list:
        return ["single_run_radar_chart"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Generate radar chart for single run."""
        if not context.config.enable_plots:
            print("[i] Plots disabled, skipping radar chart")
            return context
        
        # Get task performance file
        task_perf_path = context.get_artifact("task_performance_overall")
        if not task_perf_path or not task_perf_path.exists():
            print("[!] Task performance file not found, skipping radar chart")
            return context
        
        output_dir = context.config.get_run_output_dir() / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Generating single-run radar chart...")
        
        # Load task performance
        df = pd.read_csv(task_perf_path)
        
        # Extract tasks and scores
        tasks = df['task'].tolist()
        scores = df['score_rate'].tolist()
        
        # Map tasks to abbreviations
        task_abbrevs = [task_abbreviations.get(task, task) for task in tasks]
        
        # Create radar chart
        num_vars = len(tasks)
        if num_vars == 0:
            print("[!] No tasks found, skipping radar chart")
            return context
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]
        values = scores + [scores[0]]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.fill(angles, values, color='#1B9CFC', alpha=0.25)
        ax.plot(angles, values, color='#1B9CFC', linewidth=2)
        
        # Add value labels
        for angle, value, task_abbrev in zip(angles[:-1], scores, task_abbrevs):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                   ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(task_abbrevs, fontsize=14)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        
        run_id = context.config.run_id or "default"
        plt.title(f'Task Performance: {run_id.replace("_", " ")}', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"radar_chart.{context.config.plot_format}"
        plt.savefig(output_path, format=context.config.plot_format, bbox_inches='tight', dpi=context.config.plot_dpi)
        plt.close()
        
        context.add_artifact("single_run_radar_chart", output_path)
        print(f"[+] Single-run radar chart saved to {output_path}")
        
        return context

