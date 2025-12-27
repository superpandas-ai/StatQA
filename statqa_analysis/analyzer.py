# -*- coding: utf-8 -*-
"""
Main analyzer classes for single-run and cohort analysis.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

from .config import AnalysisConfig, AnalysisContext
from .pipeline import AnalysisPipeline
from .io import load_csv, save_csv
from .analyses import (
    MetadataMerge,
    GroundTruthDerivation,
    AnswerExtraction,
    CompareAndCount,
    Scoring,
    TaskPerformance,
    ConfusionMatrixAnalysis,
    ErrorTypeAnalysis,
    SingleRunRadarChart,
    RadarChartGeneration,
)


class ModelOutputAnalyzer:
    """
    Analyzes a single model output file (e.g., gpt-5.2_one-shot.csv).
    
    Runs the full pipeline:
    1. Load raw data
    2. Derive ground truth
    3. Extract JSON answers
    4. Compare with ground truth
    5. Calculate scores
    6. Generate per-task performance tables
    7. Generate confusion matrix
    8. Perform error analysis
    """
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str = "AnalysisOutput",
        run_id: Optional[str] = None,
        config: Optional[AnalysisConfig] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            input_csv: Path to the raw model output CSV
            output_dir: Root directory for analysis outputs
            run_id: Unique identifier for this run (defaults to input filename stem)
            config: Optional AnalysisConfig (overrides other params if provided)
        """
        if config is None:
            input_path = Path(input_csv)
            if run_id is None:
                # Try to auto-detect run_id from path structure
                # Look for pattern: .../runs/<run-id>/raw/model_outputs.csv
                parts = input_path.parts
                if 'runs' in parts:
                    runs_idx = parts.index('runs')
                    if runs_idx + 1 < len(parts):
                        # Check if next part after 'runs' is likely a run-id
                        # (i.e., not 'raw' or 'tables' or 'plots')
                        potential_run_id = parts[runs_idx + 1]
                        if potential_run_id not in ['raw', 'tables', 'plots', 'artifacts']:
                            run_id = potential_run_id
                
                # Fallback to filename stem if no run_id detected
                if run_id is None:
                    run_id = input_path.stem
            
            config = AnalysisConfig(
                input_csv=input_path,
                output_dir=Path(output_dir),
                run_id=run_id
            )
        
        self.config = config
        self.context: Optional[AnalysisContext] = None
    
    def run_all(self) -> AnalysisContext:
        """
        Run the complete analysis pipeline.
        
        Returns:
            AnalysisContext with all results and artifacts
        """
        print(f"\n{'='*70}")
        print(f"StatQA Model Output Analysis")
        print(f"{'='*70}")
        print(f"Run ID: {self.config.run_id}")
        print(f"Input: {self.config.input_csv}")
        print(f"Output: {self.config.get_run_output_dir()}")
        print(f"{'='*70}\n")
        
        # Initialize context
        self.context = AnalysisContext(config=self.config)
        
        # Load raw data
        print("[*] Loading raw data...")
        self.context.df = load_csv(self.config.input_csv)
        self.context.add_result("raw_data", True)
        print(f"[+] Loaded {len(self.context.df)} rows")
        
        # Build pipeline
        analyses = [
            MetadataMerge(),          # Merge metadata from mini-StatQA.json
            GroundTruthDerivation(),  # Derive ground_truth if needed
            AnswerExtraction(),       # Extract JSON if needed
            CompareAndCount(),
            Scoring(),
            TaskPerformance(),
            ConfusionMatrixAnalysis(),
            ErrorTypeAnalysis(),      # Includes error bar chart generation
            SingleRunRadarChart(),    # Single-run radar chart
        ]
        
        pipeline = AnalysisPipeline(analyses)
        
        # Run pipeline with initial resources
        initial_resources = {"raw_data"}
        self.context = pipeline.run(self.context, initial_resources=initial_resources)
        
        # Save processed data
        artifacts_dir = self.config.get_run_output_dir() / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        processed_path = artifacts_dir / "processed.csv"
        save_csv(self.context.df, processed_path)
        self.context.add_artifact("processed_csv", processed_path)
        print(f"\n[+] Processed data saved to {processed_path}")
        
        # Print summary
        self._print_summary()
        
        return self.context
    
    def _print_summary(self):
        """Print a summary of the analysis."""
        print(f"\n{'='*70}")
        print("Analysis Summary")
        print(f"{'='*70}")
        
        # Accuracy metrics
        methods_acc = self.context.get_result("methods_accuracy")
        columns_acc = self.context.get_result("columns_accuracy")
        overall_acc = self.context.get_result("overall_accuracy")
        
        if methods_acc is not None:
            print(f"Methods Accuracy:  {methods_acc:.4f}")
        if columns_acc is not None:
            print(f"Columns Accuracy:  {columns_acc:.4f}")
        if overall_acc is not None:
            print(f"Overall Accuracy:  {overall_acc:.4f}")
        
        # Artifacts generated
        print(f"\nArtifacts Generated: {len(self.context.artifacts)}")
        for name, path in self.context.artifacts.items():
            print(f"  - {name}: {path}")
        
        print(f"{'='*70}\n")


class CohortAnalyzer:
    """
    Analyzes multiple model outputs as a cohort.
    
    Generates:
    - Summary performance tables across models
    - Radar charts
    - Batch confusion matrices
    - Aggregated error analysis
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str = "AnalysisOutput",
        cohort_name: str = "default",
        config: Optional[AnalysisConfig] = None,
        filter_pattern: Optional[str] = None
    ):
        """
        Initialize the cohort analyzer.
        
        Args:
            input_dir: Directory containing processed CSV files from individual runs
            output_dir: Root directory for analysis outputs
            cohort_name: Name for this cohort
            config: Optional base AnalysisConfig
        """
        if config is None:
            config = AnalysisConfig(
                output_dir=Path(output_dir)
            )
        
        self.config = config
        self.input_dir = Path(input_dir)
        self.cohort_name = cohort_name
        self.filter_pattern = filter_pattern
        self.context: Optional[AnalysisContext] = None
    
    def run_all(self) -> AnalysisContext:
        """
        Run cohort-level analyses.
        
        Returns:
            AnalysisContext with cohort results
        """
        print(f"\n{'='*70}")
        print(f"StatQA Cohort Analysis")
        print(f"{'='*70}")
        print(f"Cohort: {self.cohort_name}")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.config.get_cohort_output_dir(self.cohort_name)}")
        print(f"{'='*70}\n")
        
        # Initialize context
        self.context = AnalysisContext(config=self.config)
        
        # Find all task performance files
        task_perf_files = self._find_task_performance_files()
        
        if not task_perf_files:
            print("[!] No task performance files found in input directory")
            return self.context
        
        print(f"[*] Found {len(task_perf_files)} runs to analyze")
        
        # Generate summary tables for each target
        for target in ['methods', 'columns', 'overall']:
            self._generate_summary_table(task_perf_files, target)
        
        # Generate radar charts if enabled
        if self.config.enable_radar_charts:
            self._generate_radar_charts()
        
        # Generate error analysis bar charts if enabled
        if self.config.enable_error_analysis:
            self._generate_cohort_error_charts()
        
        print(f"\n[+] Cohort analysis completed")
        
        return self.context
    
    def _find_task_performance_files(self) -> dict:
        """
        Find task performance files from processed runs.
        
        Returns:
            Dict mapping run_id to dict of file paths by target
        """
        files_by_run = {}
        
        # Handle both cases: input_dir is the runs directory itself, or its parent
        if (self.input_dir / "runs").exists():
            # input_dir is the parent (e.g., AnalysisOutput)
            runs_dir = self.input_dir / "runs"
        elif self.input_dir.name == "runs" or any((self.input_dir / d / "tables").exists() for d in self.input_dir.iterdir() if d.is_dir()):
            # input_dir is already the runs directory (e.g., AnalysisOutput/runs)
            runs_dir = self.input_dir
        else:
            return files_by_run
        
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    run_id = run_dir.name
                    # Apply filter pattern if provided
                    if self.filter_pattern and self.filter_pattern not in run_id:
                        continue
                    tables_dir = run_dir / "tables"
                    if tables_dir.exists():
                        files_by_run[run_id] = {}
                        for target in ['methods', 'columns', 'overall']:
                            perf_file = tables_dir / f"task_performance_{target}.csv"
                            if perf_file.exists():
                                files_by_run[run_id][target] = perf_file
        
        return files_by_run
    
    def _generate_summary_table(self, task_perf_files: dict, target: str):
        """Generate summary table for a specific target."""
        print(f"[*] Generating summary table for {target}...")
        
        output_dir = self.config.get_cohort_output_dir(self.cohort_name) / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data from all runs
        model_data_list = []
        
        for run_id, files in task_perf_files.items():
            if target not in files:
                continue
            
            df = load_csv(files[target])
            
            # Calculate overall score rate
            total_obtained = df['obtained_valid_score'].sum()
            total_full = df['total_full_score'].sum()
            overall_score_rate = total_obtained / total_full if total_full else 0
            
            # Build row for this model
            model_data = {
                'model': run_id,
                'overall_score_rate': round(overall_score_rate, 5)
            }
            
            # Add per-task scores
            for _, row in df.iterrows():
                model_data[row['task']] = row['score_rate']
            
            model_data_list.append(model_data)
        
        if not model_data_list:
            print(f"[!] No data for {target}, skipping")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(model_data_list)
        
        # Reorder columns if tasks are present
        if 'task' in summary_df.columns or len(summary_df.columns) > 2:
            # Standard task order from StatQA
            standard_tasks = [
                "Correlation Analysis",
                "Contingency Table Test",
                "Distribution Compliance Test",
                "Variance Test",
                "Descriptive Statistics"
            ]
            
            # Build column order: model, overall_score_rate, then tasks
            col_order = ['model', 'overall_score_rate']
            for task in standard_tasks:
                if task in summary_df.columns:
                    col_order.append(task)
            
            # Add any other columns not in standard list
            for col in summary_df.columns:
                if col not in col_order:
                    col_order.append(col)
            
            summary_df = summary_df.reindex(columns=col_order)
        
        # Save
        output_path = output_dir / f"{target}_selection_summary_performance.csv"
        save_csv(summary_df, output_path)
        
        self.context.add_artifact(f"summary_{target}", output_path)
        print(f"[+] Summary for {target} saved to {output_path}")
    
    def _generate_radar_charts(self):
        """Generate radar charts from summary tables."""
        print("[*] Generating radar charts...")
        
        output_dir = self.config.get_cohort_output_dir(self.cohort_name) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for target in ['methods', 'columns', 'overall']:
            summary_artifact = self.context.get_artifact(f"summary_{target}")
            if summary_artifact and summary_artifact.exists():
                output_path = output_dir / f"radar_{target}.{self.config.plot_format}"
                try:
                    RadarChartGeneration.generate_radar_chart(
                        summary_csv_path=summary_artifact,
                        output_path=output_path,
                        title=f"{target.title()} Selection Performance",
                        dpi=self.config.plot_dpi
                    )
                    self.context.add_artifact(f"radar_{target}", output_path)
                except Exception as e:
                    print(f"[!] Error generating radar chart for {target}: {e}")
    
    def _generate_cohort_error_charts(self):
        """Generate error analysis bar charts for cohort."""
        print("[*] Generating cohort error analysis bar charts...")
        
        # Collect error analysis summaries from all runs
        error_summaries = []
        runs_dir = Path(self.input_dir) / "runs"
        
        if not runs_dir.exists():
            # Try alternative structure
            runs_dir = Path(self.input_dir)
            if not runs_dir.exists():
                print("[!] Cannot find runs directory for error analysis")
                return
        
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                error_file = run_dir / "tables" / "error_analysis_summary.csv"
                if error_file.exists():
                    try:
                        df = pd.read_csv(error_file)
                        error_summaries.append(df)
                    except Exception as e:
                        print(f"[!] Error reading {error_file}: {e}")
        
        if not error_summaries:
            print("[!] No error analysis summaries found")
            return
        
        # Combine all summaries
        combined_df = pd.concat(error_summaries, ignore_index=True)
        
        # Save combined summary
        output_dir = self.config.get_cohort_output_dir(self.cohort_name) / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_path = output_dir / "error_analysis_summary.csv"
        combined_df.to_csv(combined_path, index=False)
        self.context.add_artifact("cohort_error_summary", combined_path)
        
        # Generate bar chart
        if self.config.enable_plots:
            self._plot_error_bar_chart(combined_df)
        
        print(f"[+] Cohort error analysis saved to {combined_path}")
    
    def _plot_error_bar_chart(self, summary_df: pd.DataFrame):
        """Plot error analysis bar chart for cohort."""
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt
        
        output_dir = self.config.get_cohort_output_dir(self.cohort_name) / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Error types
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
        
        # Process model names for display
        summary_df = summary_df.copy()
        summary_df['DisplayName'] = summary_df['Model'].apply(
            lambda x: x.replace('_', ' ').replace('zero', '0').replace('one', '1').replace('two', '2')
                      .replace('instruct', 'inst').replace('gpt-3.5-turbo', 'GPT-3.5T')
                      .replace('gpt-4', 'GPT-4').replace('stats-prompt', '1-shot+DK')
        )
        
        models = summary_df['DisplayName'].tolist()
        
        fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.8), 8))
        
        # Create stacked bars
        bottom = np.zeros(len(models))
        for j, error_type in enumerate(error_types):
            values = summary_df[error_type].values
            ax.bar(models, values, bottom=bottom, color=colors[j], 
                  edgecolor='white', width=0.8, label=error_type)
            
            # Add value labels
            for i, (val, btm) in enumerate(zip(values, bottom)):
                if val > 0.02:
                    ax.text(i, btm + val / 2, f'{val:.2f}', 
                           ha='center', va='center', fontsize=10, fontweight='bold')
                elif val > 0.01:
                    ax.text(i, btm + val / 2, f'{val:.2f}', 
                           ha='center', va='center', fontsize=8)
            
            bottom += values
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Error Rate', fontsize=14)
        ax.set_title(f'Error Type Analysis: {self.cohort_name}', fontsize=16)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', fontsize=10, ncol=2)
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f"error_analysis_bar_chart.{self.config.plot_format}"
        plt.savefig(output_path, format=self.config.plot_format, bbox_inches='tight', dpi=self.config.plot_dpi)
        plt.close()
        
        self.context.add_artifact("cohort_error_bar_chart", output_path)
        print(f"[+] Cohort error analysis bar chart saved to {output_path}")

