# -*- coding: utf-8 -*-
"""
Command-line interface for StatQA analysis framework.
"""

import argparse
import sys
from pathlib import Path

from .analyzer import ModelOutputAnalyzer, CohortAnalyzer
from .config import AnalysisConfig


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="StatQA Model Answer Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single model output
  python -m statqa_analysis run --input gpt-5.2_one-shot.csv --out AnalysisOutput
  
  # Analyze a cohort of runs
  python -m statqa_analysis cohort --input-dir AnalysisOutput/runs --out AnalysisOutput
  
  # Run with specific options
  python -m statqa_analysis run --input model.csv --no-plots --method-metric jaccard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single-run analysis command
    run_parser = subparsers.add_parser('run', help='Analyze a single model output')
    run_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to raw model output CSV'
    )
    run_parser.add_argument(
        '--out', '-o',
        default='AnalysisOutput',
        help='Root output directory (default: AnalysisOutput)'
    )
    run_parser.add_argument(
        '--run-id',
        help='Unique identifier for this run (default: input filename stem)'
    )
    run_parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    run_parser.add_argument(
        '--no-confusion-matrix',
        action='store_true',
        help='Disable confusion matrix generation'
    )
    run_parser.add_argument(
        '--no-error-analysis',
        action='store_true',
        help='Disable error type analysis'
    )
    run_parser.add_argument(
        '--method-metric',
        choices=['acc', 'jaccard'],
        default='acc',
        help='Metric for methods scoring (default: acc)'
    )
    run_parser.add_argument(
        '--plot-dpi',
        type=int,
        default=300,
        help='DPI for plots (default: 300)'
    )
    run_parser.add_argument(
        '--plot-format',
        choices=['pdf', 'png', 'svg'],
        default='pdf',
        help='Plot format (default: pdf)'
    )
    
    # Cohort analysis command
    cohort_parser = subparsers.add_parser('cohort', help='Analyze multiple runs as a cohort')
    cohort_parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Directory containing processed runs (e.g., AnalysisOutput/runs)'
    )
    cohort_parser.add_argument(
        '--out', '-o',
        default='AnalysisOutput',
        help='Root output directory (default: AnalysisOutput)'
    )
    cohort_parser.add_argument(
        '--cohort-name',
        default='default',
        help='Name for this cohort (default: default)'
    )
    cohort_parser.add_argument(
        '--no-radar-charts',
        action='store_true',
        help='Disable radar chart generation'
    )
    cohort_parser.add_argument(
        '--plot-dpi',
        type=int,
        default=300,
        help='DPI for plots (default: 300)'
    )
    cohort_parser.add_argument(
        '--plot-format',
        choices=['pdf', 'png', 'svg'],
        default='pdf',
        help='Plot format (default: pdf)'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'run':
            run_single_analysis(args)
        elif args.command == 'cohort':
            run_cohort_analysis(args)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        sys.exit(1)


def run_single_analysis(args):
    """Run single model output analysis."""
    config = AnalysisConfig(
        input_csv=Path(args.input),
        output_dir=Path(args.out),
        run_id=args.run_id,
        enable_plots=not args.no_plots,
        enable_confusion_matrix=not args.no_confusion_matrix,
        enable_error_analysis=not args.no_error_analysis,
        method_metric=args.method_metric,
        plot_dpi=args.plot_dpi,
        plot_format=args.plot_format
    )
    
    analyzer = ModelOutputAnalyzer(
        input_csv=str(args.input),
        output_dir=str(args.out),
        run_id=args.run_id,
        config=config
    )
    
    analyzer.run_all()


def run_cohort_analysis(args):
    """Run cohort analysis."""
    config = AnalysisConfig(
        output_dir=Path(args.out),
        enable_radar_charts=not args.no_radar_charts,
        plot_dpi=args.plot_dpi,
        plot_format=args.plot_format
    )
    
    analyzer = CohortAnalyzer(
        input_dir=str(args.input_dir),
        output_dir=str(args.out),
        cohort_name=args.cohort_name,
        config=config
    )
    
    analyzer.run_all()


if __name__ == '__main__':
    main()

