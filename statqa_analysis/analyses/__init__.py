"""
Pluggable analysis modules for StatQA model output processing.
"""

from .metadata_merge import MetadataMerge
from .ground_truth import GroundTruthDerivation
from .answer_extraction import AnswerExtraction
from .compare_count import CompareAndCount
from .scoring import Scoring
from .task_performance import TaskPerformance
from .confusion_matrix import ConfusionMatrixAnalysis
from .error_types import ErrorTypeAnalysis
from .single_run_radar import SingleRunRadarChart
from .summary_tables import SummaryTableGeneration
from .radar_charts import RadarChartGeneration

__all__ = [
    'MetadataMerge',
    'GroundTruthDerivation',
    'AnswerExtraction',
    'CompareAndCount',
    'Scoring',
    'TaskPerformance',
    'ConfusionMatrixAnalysis',
    'ErrorTypeAnalysis',
    'SingleRunRadarChart',
    'SummaryTableGeneration',
    'RadarChartGeneration',
]

