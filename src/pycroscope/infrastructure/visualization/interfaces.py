"""
Visualization interfaces and contracts.

Defines abstract base classes for profiler-specific visualization components
following the Interface Segregation Principle.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ProfilerPlotter(ABC):
    """
    Abstract base class for profiler-specific visualization.

    Each profiler type (call, line, memory, sampling) implements this interface
    to provide industry-standard visualization capabilities.
    """

    @abstractmethod
    def can_plot(self, profiler_data: Dict[str, Any]) -> bool:
        """
        Check if this plotter can handle the given profiler data.

        Args:
            profiler_data: Raw profiling data from a specific profiler

        Returns:
            True if this plotter can generate visualizations from the data
        """
        pass

    @abstractmethod
    def generate_plots(
        self, profiler_data: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generate all standard plots for this profiler type.

        Args:
            profiler_data: Raw profiling data from the profiler
            output_dir: Directory to save generated plots

        Returns:
            Dictionary mapping plot names to their file paths
        """
        pass

    @abstractmethod
    def get_plot_types(self) -> List[str]:
        """
        Get list of plot types this plotter can generate.

        Returns:
            List of standard plot type names for this profiler
        """
        pass


class StyleManager:
    """
    Manages consistent professional styling across all visualizations.

    Implements the Single Responsibility Principle for visual styling.
    """

    @staticmethod
    def apply_professional_style() -> None:
        """Apply consistent professional styling to matplotlib."""
        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 13,
                "figure.titlesize": 14,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 10,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )

    @staticmethod
    def get_color_palette() -> List[str]:
        """Get consistent color palette for all charts."""
        return [
            "#2E86AB",  # Blue
            "#A23B72",  # Purple
            "#F18F01",  # Orange
            "#C73E1D",  # Red
            "#6A994E",  # Green
            "#8B5A3C",  # Brown
            "#4C956C",  # Teal
            "#F26419",  # Bright Orange
            "#2F9599",  # Cyan
            "#C7522A",  # Rust
        ]

    @staticmethod
    def save_figure(fig: Figure, filepath: Path, **kwargs) -> None:
        """
        Save figure with consistent high-quality settings.

        Args:
            fig: Matplotlib figure to save
            filepath: Output file path
            **kwargs: Additional arguments for savefig
        """
        default_kwargs = {
            "dpi": 300,
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none",
        }
        default_kwargs.update(kwargs)

        fig.savefig(filepath, **default_kwargs)
        plt.close(fig)
