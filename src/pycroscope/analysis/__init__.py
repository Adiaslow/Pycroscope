"""
Performance Anti-Pattern Analysis Module for Pycroscope.

This module provides algorithmic pattern analysis and performance anti-pattern
identification by leveraging established static analysis tools while integrating
seamlessly with Pycroscope's profiling data.

Key Features:
- Algorithmic complexity detection and O(n) analysis
- Common anti-pattern identification (nested loops, inefficient data structures)
- Dead code and unused import detection
- Code maintainability analysis
- Scientific computing anti-pattern detection (vectorization, array operations, etc.)
- Integration with profiling data for hot-spot correlation
"""

from .interfaces import (
    PatternAnalyzer,
    AntiPatternDetector,
    ComplexityAnalyzer,
    AnalysisResult,
    PatternType,
)
from .detectors import (
    AlgorithmicComplexityDetector,
    DeadCodeDetector,
    AntiPatternDetector as AntiPatternDetectorImpl,
    MaintainabilityAnalyzer,
    ScientificComputingDetector,
)
from .orchestrator import AnalysisOrchestrator
from .config import AnalysisConfig

__all__ = [
    "PatternAnalyzer",
    "AntiPatternDetector",
    "ComplexityAnalyzer",
    "AnalysisResult",
    "PatternType",
    "AlgorithmicComplexityDetector",
    "DeadCodeDetector",
    "AntiPatternDetectorImpl",
    "MaintainabilityAnalyzer",
    "ScientificComputingDetector",
    "AnalysisOrchestrator",
    "AnalysisConfig",
]
