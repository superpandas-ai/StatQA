# -*- coding: utf-8 -*-
"""
Configuration dataclasses for the analysis framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


@dataclass
class AnalysisConfig:
    """Configuration for analysis runs."""
    
    # Input/output paths
    input_csv: Optional[Path] = None
    output_dir: Path = Path("AnalysisOutput")
    run_id: Optional[str] = None
    
    # Analysis control flags
    enable_plots: bool = True
    enable_confusion_matrix: bool = True
    enable_error_analysis: bool = True
    enable_radar_charts: bool = True
    
    # Scoring method metric ('acc' or 'jaccard')
    method_metric: str = 'acc'
    
    # Plot parameters
    plot_dpi: int = 300
    plot_format: str = 'pdf'
    
    # Custom parameters for analyses (extensible)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_run_output_dir(self) -> Path:
        """Get the output directory for this run."""
        if self.run_id:
            return self.output_dir / "runs" / self.run_id
        return self.output_dir / "runs" / "default"
    
    def get_cohort_output_dir(self, cohort_name: str) -> Path:
        """Get the output directory for a cohort."""
        return self.output_dir / "cohorts" / cohort_name


@dataclass
class AnalysisContext:
    """
    Context object that carries state through the analysis pipeline.
    Holds DataFrames, paths, and intermediate results.
    """
    
    config: AnalysisConfig
    
    # Core dataframes
    df: Optional[pd.DataFrame] = None
    df_processed: Optional[pd.DataFrame] = None
    
    # Paths to generated artifacts
    artifacts: Dict[str, Path] = field(default_factory=dict)
    
    # Intermediate results
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_artifact(self, name: str, path: Path) -> None:
        """Register an artifact (file) produced by an analysis."""
        self.artifacts[name] = path
    
    def add_result(self, name: str, value: Any) -> None:
        """Store an intermediate result."""
        self.results[name] = value
    
    def get_artifact(self, name: str) -> Optional[Path]:
        """Retrieve path to a named artifact."""
        return self.artifacts.get(name)
    
    def get_result(self, name: str) -> Any:
        """Retrieve a named result."""
        return self.results.get(name)

