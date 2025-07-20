"""
Pycroscope: Development Optimization Framework

A comprehensive Python profiling system designed for development-time package optimization.
Provides complete performance analysis through multi-dimensional data collection.
"""

__version__ = "1.0.0"
__author__ = "Pycroscope Contributors"

from .core.config import CollectorConfig, CollectorType, ProfileConfig
from .core.models import AnalysisResult, ProfileSession

# Core public API - "One Way, Many Options"
from .core.profiler_suite import ProfilerSuite


# Main convenience function for simple usage
def enable_profiling(config=None):
    """
    Enable comprehensive profiling with single function call.

    Args:
        config: Optional ProfileConfig instance for customization

    Returns:
        ProfilerSuite instance for advanced control
    """
    suite = ProfilerSuite(config or ProfileConfig())
    suite.enable()
    return suite


# Public API exports
__all__ = [
    "ProfilerSuite",
    "ProfileConfig",
    "CollectorConfig",
    "CollectorType",
    "ProfileSession",
    "AnalysisResult",
    "enable_profiling",
]
