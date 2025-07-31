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

    Each profiler type (call, line, memory) implements this interface
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

    Implements the Single Responsibility Principle for visual styling,
    with proper isolation from the target application's matplotlib configuration.
    """

    @staticmethod
    def get_style_dict() -> dict:
        """Get professional style settings dictionary."""
        return {
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

    @staticmethod
    def apply_professional_style():
        """
        Get a context manager for professional styling.

        This method now returns a context manager to isolate styling
        changes from the global matplotlib state.

        Usage:
            with StyleManager.apply_professional_style():
                # Create plots here

        For backward compatibility, this can still be called directly,
        but it will return the context manager without applying it.
        """
        # Use a combination of style and rc_context to ensure all styles are applied properly
        # This provides stronger isolation from global state
        style_dict = StyleManager.get_style_dict()

        # Create a custom context manager that combines style context and explicit rcParams
        class CombinedStyleContext:
            def __enter__(self):
                self.style_context = plt.style.context(style_dict)
                self.style_context.__enter__()
                # Set additional critical parameters that might not be captured by style context
                self.original_params = plt.rcParams.copy()
                plt.rcParams.update(style_dict)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore original parameters
                plt.rcParams.update(self.original_params)
                return self.style_context.__exit__(exc_type, exc_val, exc_tb)

        return CombinedStyleContext()

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

        # Apply comprehensive style settings directly to the figure
        # This ensures consistent styling regardless of the global state

        # Get the original figure size to check if it's been customized
        orig_width, orig_height = fig.get_size_inches()

        # Only set default size if the figure doesn't already have a custom size
        # (Custom size is detected by checking if dimensions are significantly different from defaults)
        if (
            abs(orig_width - 6.4) < 0.1 and abs(orig_height - 4.8) < 0.1
        ):  # Default matplotlib size
            fig.set_figheight(8)
            fig.set_figwidth(12)

        fig.set_facecolor("white")

        for ax in fig.get_axes():
            # Apply font styling
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_fontsize(9)

            # Title and label styling
            if ax.get_title():
                ax.title.set_fontsize(13)
            if ax.get_xlabel():
                ax.xaxis.label.set_fontsize(11)
            if ax.get_ylabel():
                ax.yaxis.label.set_fontsize(11)

            # Grid and spines
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("white")

            # Legend styling if present
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(10)

        fig.savefig(filepath, **default_kwargs)
        plt.close(fig)
