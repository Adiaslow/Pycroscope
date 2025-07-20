"""
Analysis engines and processing components for Pycroscope.

Provides multi-pass analysis of profiling data to extract insights,
detect patterns, and generate optimization recommendations.
"""

from .base_analyzer import BaseAnalyzer
from .complexity_detector import AlgorithmComplexityDetector
from .correlation_analyzer import CrossCorrelationAnalyzer
from .dynamic_analyzer import DynamicAnalyzer
from .engine import AnalysisEngine
from .optimization_engine import OptimizationRecommendationEngine
from .pattern_detector import AdvancedPatternDetector
from .static_analyzer import StaticAnalyzer

__all__ = [
    "AnalysisEngine",
    "StaticAnalyzer",
    "DynamicAnalyzer",
    "BaseAnalyzer",
    "AdvancedPatternDetector",
    "CrossCorrelationAnalyzer",
    "AlgorithmComplexityDetector",
    "OptimizationRecommendationEngine",
]
