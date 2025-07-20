"""
Analysis engines and processing components for Pycroscope.

Provides multi-pass analysis of profiling data to extract insights,
detect patterns, and generate optimization recommendations.
"""

from .engine import AnalysisEngine
from .static_analyzer import StaticAnalyzer
from .dynamic_analyzer import DynamicAnalyzer
from .base_analyzer import BaseAnalyzer
from .pattern_detector import AdvancedPatternDetector
from .correlation_analyzer import CrossCorrelationAnalyzer
from .complexity_detector import AlgorithmComplexityDetector
from .optimization_engine import OptimizationRecommendationEngine

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
