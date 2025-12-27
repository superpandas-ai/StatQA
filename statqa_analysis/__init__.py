"""
StatQA Model Answer Analysis Framework

A unified, extensible framework for analyzing LLM outputs on StatQA benchmark.
"""

from .analyzer import ModelOutputAnalyzer, CohortAnalyzer
from .config import AnalysisConfig, AnalysisContext

__all__ = [
    'ModelOutputAnalyzer',
    'CohortAnalyzer',
    'AnalysisConfig',
    'AnalysisContext',
]

__version__ = '1.0.0'

