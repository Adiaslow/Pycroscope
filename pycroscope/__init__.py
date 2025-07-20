"""
Pycroscope: Development Optimization Framework

A comprehensive Python profiling system designed for development-time package optimization.
Provides complete performance analysis through multi-dimensional data collection.
"""

__version__ = "0.1.0"
__author__ = "Pycroscope Contributors"

# Core public API - "One Way, Many Options"
from .core.profiler_suite import ProfilerSuite
from .core.config import ProfileConfig, CollectorConfig, CollectorType
from .core.models import ProfileSession, AnalysisResult


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
