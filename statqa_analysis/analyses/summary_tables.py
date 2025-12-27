# -*- coding: utf-8 -*-
"""
Summary table generation for cohort analysis.
"""

import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..pipeline import BaseAnalysis
from ..config import AnalysisContext
from ..io import load_csv, save_csv


class SummaryTableGeneration(BaseAnalysis):
    """
    Generates summary performance tables across multiple runs.
    """
    
    @property
    def name(self) -> str:
        return "summary_tables"
    
    @property
    def requires(self) -> list:
        return []  # Operates on cohort-level inputs
    
    @property
    def produces(self) -> list:
        return ["summary_methods", "summary_columns", "summary_overall"]
    
    def run(self, context: AnalysisContext) -> AnalysisContext:
        """Generate summary tables."""
        # This is handled directly in CohortAnalyzer for now
        # Could be refactored to use pipeline if needed
        return context

