"""
StatQA Model Answer Analysis Framework

A unified, extensible framework for analyzing LLM outputs on StatQA benchmark.
Includes dataset management, prompt generation, model inference, and analysis.
"""

from .analyzer import ModelOutputAnalyzer, CohortAnalyzer
from .config import AnalysisConfig, AnalysisContext
from .datasets import DatasetImporter, DatasetRegistry
from .prompts import PromptBuilder
from .inference import AzureOpenAIRunner

__all__ = [
    'ModelOutputAnalyzer',
    'CohortAnalyzer',
    'AnalysisConfig',
    'AnalysisContext',
    'DatasetImporter',
    'DatasetRegistry',
    'PromptBuilder',
    'AzureOpenAIRunner',
]

__version__ = '2.0.0'

