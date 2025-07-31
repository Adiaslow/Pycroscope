"""
Chart generation coordinator for profiling sessions.

Coordinates specialized plotters to generate industry-standard performance
analysis visualizations following clean architecture principles.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.session import ProfileSession
from ...core.exceptions import ConfigurationError, ValidationError
from .interfaces import ProfilerPlotter, StyleManager
from .plotters import (
    CallProfilerPlotter,
    LineProfilerPlotter,
    MemoryProfilerPlotter,
)


class ChartGenerator:
    """
    Coordinates chart generation across all profiler types.

    Implements the Facade pattern to provide a unified interface for
    generating industry-standard performance analysis visualizations.
    """

    def __init__(self, session: ProfileSession):
        self.session = session
        self._validate_session()

        # Initialize specialized plotters
        self._plotters: Dict[str, ProfilerPlotter] = {
            "call": CallProfilerPlotter(),
            "line": LineProfilerPlotter(),
            "memory": MemoryProfilerPlotter(),
        }

    def _validate_session(self) -> None:
        """Validate session before chart generation."""
        if not self.session.is_complete:
            raise ValidationError("Cannot generate charts for incomplete session")

        if not self.session.results:
            raise ValidationError("Cannot generate charts for session with no results")

    def generate_all_charts(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Generate all available charts based on profiling data.

        Args:
            output_dir: Directory for output files (uses session config if None)

        Returns:
            Dictionary mapping chart names to their file paths
        """
        if output_dir is None:
            if self.session.config.output_dir is None:
                raise ConfigurationError(
                    "Output directory must be configured for chart generation",
                    config_key="output_dir",
                )
            output_dir = self.session.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Style is now applied in each plotter with proper isolation
        generated_charts = {}

        # Generate charts for each profiler type with available data
        for profiler_type, profiler_result in self.session.results.items():
            if profiler_type in self._plotters and profiler_result.success:
                plotter = self._plotters[profiler_type]

                if plotter.can_plot(profiler_result.data):
                    charts = plotter.generate_plots(profiler_result.data, output_dir)

                    # Prefix chart names with profiler type to avoid conflicts
                    for chart_name, chart_path in charts.items():
                        prefixed_name = f"{profiler_type}_{chart_name}"
                        generated_charts[prefixed_name] = chart_path

        return generated_charts

    def get_available_chart_types(self) -> Dict[str, List[str]]:
        """
        Get available chart types for each profiler.

        Returns:
            Dictionary mapping profiler types to their available chart types
        """
        available_types = {}

        for profiler_type, plotter in self._plotters.items():
            available_types[profiler_type] = plotter.get_plot_types()

        return available_types

    def generate_specific_charts(
        self, profiler_types: List[str], output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Generate charts for specific profiler types only.

        Args:
            profiler_types: List of profiler types to generate charts for
            output_dir: Output directory (uses session config if None)

        Returns:
            Dictionary mapping chart names to their file paths
        """
        if output_dir is None:
            if self.session.config.output_dir is None:
                raise ConfigurationError(
                    "Output directory must be configured for chart generation",
                    config_key="output_dir",
                )
            output_dir = self.session.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Style is now applied in each plotter with proper isolation
        generated_charts = {}

        for profiler_type in profiler_types:
            if (
                profiler_type in self._plotters
                and profiler_type in self.session.results
            ):

                profiler_result = self.session.results[profiler_type]

                if profiler_result.success:
                    plotter = self._plotters[profiler_type]

                    if plotter.can_plot(profiler_result.data):
                        charts = plotter.generate_plots(
                            profiler_result.data, output_dir
                        )

                        for chart_name, chart_path in charts.items():
                            prefixed_name = f"{profiler_type}_{chart_name}"
                            generated_charts[prefixed_name] = chart_path

        return generated_charts
