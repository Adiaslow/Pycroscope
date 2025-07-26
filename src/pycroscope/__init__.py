"""
Pycroscope: Python performance analysis and visualization using established profiling tools.

A modern Python profiling framework that leverages battle-tested profiling packages
to provide comprehensive performance analysis and beautiful visualizations.

Instead of reinventing profiling infrastructure, Pycroscope focuses on:
- Clean abstractions over existing profilers (cProfile, line_profiler, memory_profiler, py-spy)
- Advanced analysis and correlation of profiling data
- Beautiful visualizations and reports
- Unified API for multiple profiling approaches
"""

__version__ = "2.0.0"
__author__ = "Adam Murray"

# Core public API - "One Way, Many Options"
from .core import ProfileSession, ProfileConfig
from .application.services import ProfilingService
from .infrastructure.profilers.orchestra import ProfilerOrchestra

# Initialize the application service as the main entry point
_profiling_service = ProfilingService()


# Main convenience function for simple usage
def profile(
    func=None,
    *,
    line_profiling=True,
    memory_profiling=True,
    call_profiling=True,
    output_dir=None,
    generate_reports=True,
    create_visualizations=True,
    analyze_patterns=True,
    **kwargs,
):
    """
    Profile a function or code block with comprehensive analysis.

    Can be used as a decorator or context manager:

    @profile()
    def my_function():
        # Code to profile
        pass

    # Or as context manager:
    with profile() as session:
        # Code to profile
        pass

    Args:
        func: Function to profile (when used as decorator)
        line_profiling: Enable line-by-line profiling
        memory_profiling: Enable memory usage profiling
        call_profiling: Enable function call profiling

        output_dir: Directory to save results
        generate_reports: Generate comprehensive analysis reports
        create_visualizations: Create charts and visualizations
        analyze_patterns: Perform pattern analysis on profiling data
        **kwargs: Additional configuration options

    Returns:
        ProfileSession: Session containing all profiling results
    """
    config_kwargs = {
        "line_profiling": line_profiling,
        "memory_profiling": memory_profiling,
        "call_profiling": call_profiling,
        "generate_reports": generate_reports,
        "create_visualizations": create_visualizations,
        "analyze_patterns": analyze_patterns,
        **kwargs,
    }

    # Only include output_dir if it's not None
    if output_dir is not None:
        config_kwargs["output_dir"] = output_dir

    if func is not None:
        # Used as @profile (without parentheses)
        return _profiling_service.profile_function(func, **config_kwargs)
    else:
        # Used as @profile() or context manager
        return _profiling_service.profile_context(**config_kwargs)


__all__ = [
    "profile",
    "ProfileSession",
    "ProfileConfig",
    "ProfilingService",
    "ProfilerOrchestra",
]
