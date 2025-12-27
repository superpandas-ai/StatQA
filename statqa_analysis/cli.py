# -*- coding: utf-8 -*-
"""
Command-line interface for StatQA analysis framework.
"""

import argparse
import sys
from pathlib import Path

from .analyzer import ModelOutputAnalyzer, CohortAnalyzer
from .config import AnalysisConfig
from .datasets import DatasetImporter
from .prompts import PromptBuilder
from .inference import AzureOpenAIRunner


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
    
    # Dataset import command
    dataset_parser = subparsers.add_parser('dataset', help='Dataset management commands')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command', help='Dataset subcommands')
    
    import_parser = dataset_subparsers.add_parser('import', help='Import a dataset into StatDatasets/')
    import_parser.add_argument(
        '--dataset-id',
        required=True,
        help='Unique identifier for the dataset (e.g., mini-statqa, statqa-full)'
    )
    import_parser.add_argument(
        '--from-json',
        required=True,
        help='Path to source JSON file'
    )
    import_parser.add_argument(
        '--no-ensure-serial-number',
        action='store_true',
        help='Do not synthesize serial_number if missing (default: synthesize)'
    )
    import_parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing dataset'
    )
    
    # Prompts build command
    prompts_parser = subparsers.add_parser('prompts', help='Prompt generation commands')
    prompts_subparsers = prompts_parser.add_subparsers(dest='prompts_command', help='Prompts subcommands')
    
    build_parser = prompts_subparsers.add_parser('build', help='Build prompts for a dataset')
    build_parser.add_argument(
        '--dataset-id',
        required=True,
        help='Dataset identifier'
    )
    build_parser.add_argument(
        '--prompt-version',
        required=True,
        help='Prompt version identifier (e.g., v1, paper_v1, 2025-12-27)'
    )
    build_parser.add_argument(
        '--trick',
        required=True,
        choices=['zero-shot', 'one-shot', 'two-shot', 'zero-shot-CoT', 'one-shot-CoT', 'stats-prompt', 'manual'],
        help='Prompting strategy (use "manual" with --manual-template for f-string templates)'
    )
    build_parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for deterministic few-shot selection'
    )
    build_parser.add_argument(
        '--template',
        type=str,
        help='Path to custom prompt template JSON file (stored with prompt version)'
    )
    build_parser.add_argument(
        '--manual-template',
        type=str,
        help='Path to text file containing f-string template for manual trick (use {meta_info_list} and {refined_question})'
    )
    
    # Model run command
    model_parser = subparsers.add_parser('model', help='Model inference commands')
    model_subparsers = model_parser.add_subparsers(dest='model_command', help='Model subcommands')
    
    run_model_parser = model_subparsers.add_parser('run', help='Run model inference on prompts')
    run_model_parser.add_argument(
        '--model',
        required=True,
        help='Azure deployment name (e.g., gpt-4, gpt-3.5-turbo, gpt-4o)'
    )
    run_model_parser.add_argument(
        '--dataset-id',
        required=True,
        help='Dataset identifier'
    )
    run_model_parser.add_argument(
        '--prompt-version',
        required=True,
        help='Prompt version identifier'
    )
    run_model_parser.add_argument(
        '--trick',
        required=True,
        choices=['zero-shot', 'one-shot', 'two-shot', 'zero-shot-CoT', 'one-shot-CoT', 'stats-prompt'],
        help='Prompting strategy'
    )
    run_model_parser.add_argument(
        '--run-id',
        help='Custom run identifier (default: auto-generated)'
    )
    run_model_parser.add_argument(
        '--max-retries',
        type=int,
        default=2,
        help='Max retry attempts per prompt (default: 2)'
    )
    run_model_parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Save progress every N prompts (default: 5)'
    )
    
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
        '--dataset-id',
        help='Dataset ID for metadata merge (uses StatDatasets/raw/<dataset_id>/data.json)'
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
        '--filter',
        help='Filter run directories by name pattern (e.g., "stats-prompt" to only include runs with "stats-prompt" in the name)'
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
        if args.command == 'dataset':
            if args.dataset_command == 'import':
                run_dataset_import(args)
            else:
                dataset_parser.print_help()
                sys.exit(1)
        elif args.command == 'prompts':
            if args.prompts_command == 'build':
                run_prompts_build(args)
            else:
                prompts_parser.print_help()
                sys.exit(1)
        elif args.command == 'model':
            if args.model_command == 'run':
                run_model_inference(args)
            else:
                model_parser.print_help()
                sys.exit(1)
        elif args.command == 'run':
            run_single_analysis(args)
        elif args.command == 'cohort':
            run_cohort_analysis(args)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        sys.exit(1)


def run_dataset_import(args):
    """Import a dataset into StatDatasets/."""
    importer = DatasetImporter()
    
    summary = importer.import_from_json(
        dataset_id=args.dataset_id,
        json_path=Path(args.from_json),
        ensure_serial_number=not args.no_ensure_serial_number,
        force=args.force
    )
    
    print(f"\n{'='*70}")
    print("Dataset Import Summary")
    print(f"{'='*70}")
    print(f"Dataset ID: {summary['dataset_id']}")
    print(f"Row count: {summary['row_count']}")
    print(f"Data path: {summary['data_path']}")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"{'='*70}\n")


def run_prompts_build(args):
    """Build prompts for a dataset."""
    builder = PromptBuilder()
    
    template_file = Path(args.template) if args.template else None
    manual_template_file = Path(args.manual_template) if args.manual_template else None
    
    summary = builder.build_prompts(
        dataset_id=args.dataset_id,
        prompt_version=args.prompt_version,
        trick=args.trick,
        seed=args.seed,
        template_file=template_file,
        manual_template_file=manual_template_file
    )
    
    print(f"\n{'='*70}")
    print("Prompt Build Summary")
    print(f"{'='*70}")
    print(f"Dataset ID: {summary['dataset_id']}")
    print(f"Prompt version: {summary['prompt_version']}")
    print(f"Trick: {summary['trick']}")
    print(f"Row count: {summary['row_count']}")
    print(f"Output: {summary['output_path']}")
    print(f"{'='*70}\n")


def run_model_inference(args):
    """Run model inference on prompts."""
    runner = AzureOpenAIRunner()
    
    summary = runner.run_inference(
        model=args.model,
        dataset_id=args.dataset_id,
        prompt_version=args.prompt_version,
        trick=args.trick,
        run_id=args.run_id,
        max_retries=args.max_retries,
        batch_size=args.batch_size
    )
    
    print(f"\n{'='*70}")
    print("Model Inference Summary")
    print(f"{'='*70}")
    print(f"Run ID: {summary['run_id']}")
    print(f"Model: {summary['model']}")
    print(f"Total prompts: {summary['total_prompts']}")
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Output: {summary['output_path']}")
    print(f"{'='*70}\n")


def run_single_analysis(args):
    """Run single model output analysis."""
    custom_params = {}
    if hasattr(args, 'dataset_id') and args.dataset_id:
        custom_params['dataset_id'] = args.dataset_id
    
    # Auto-detect run_id from path if not provided
    run_id = args.run_id
    if run_id is None:
        input_path = Path(args.input)
        # Look for pattern: .../runs/<run-id>/raw/model_outputs.csv
        parts = input_path.parts
        if 'runs' in parts:
            runs_idx = parts.index('runs')
            if runs_idx + 1 < len(parts):
                potential_run_id = parts[runs_idx + 1]
                if potential_run_id not in ['raw', 'tables', 'plots', 'artifacts']:
                    run_id = potential_run_id
    
    config = AnalysisConfig(
        input_csv=Path(args.input),
        output_dir=Path(args.out),
        run_id=run_id,
        enable_plots=not args.no_plots,
        enable_confusion_matrix=not args.no_confusion_matrix,
        enable_error_analysis=not args.no_error_analysis,
        method_metric=args.method_metric,
        plot_dpi=args.plot_dpi,
        plot_format=args.plot_format,
        custom_params=custom_params
    )
    
    analyzer = ModelOutputAnalyzer(
        input_csv=str(args.input),
        output_dir=str(args.out),
        run_id=run_id,
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
        config=config,
        filter_pattern=getattr(args, 'filter', None)
    )
    
    analyzer.run_all()


if __name__ == '__main__':
    main()

