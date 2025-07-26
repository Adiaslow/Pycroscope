"""
Line profiler visualization implementation.

Implements industry-standard line_profiler visualization:
Line-by-Line Heatmap - Essential for line-level performance analysis
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .interfaces import ProfilerPlotter, StyleManager


class LineProfilerPlotter(ProfilerPlotter):
    """
    Professional line_profiler data visualization.

    Generates industry-standard line-by-line timing heatmaps used by
    developers for function-level optimization.
    """

    def can_plot(self, profiler_data: Dict[str, Any]) -> bool:
        """Check if data contains line profiler results."""
        return "line_stats" in profiler_data

    def get_plot_types(self) -> List[str]:
        """Get available plot types for line profiler."""
        return ["line_heatmap"]

    def generate_plots(
        self, profiler_data: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Path]:
        """Generate all line profiler visualizations."""
        StyleManager.apply_professional_style()

        if not self.can_plot(profiler_data):
            raise RuntimeError(
                f"Cannot generate line profiler plots: Invalid data structure. "
                f"Data keys: {list(profiler_data.keys())}"
            )

        plots = {}

        # Line-by-Line Heatmap
        heatmap_fig = self._plot_line_heatmap(profiler_data)
        heatmap_path = output_dir / "line_heatmap.png"
        StyleManager.save_figure(heatmap_fig, heatmap_path)
        plots["line_heatmap"] = heatmap_path

        return plots

    def _plot_line_heatmap(self, profiler_data: Dict[str, Any]) -> Figure:
        """
        Create line-by-line timing heatmap from line_profiler output.

        Industry standard for line-level performance analysis used by
        developers optimizing specific functions.
        """
        # Extract line timing data
        line_timings = self._extract_line_timings(profiler_data)

        if not line_timings:
            raise RuntimeError(
                "Line profiler generated no timing data. This indicates the profiler "
                "was not properly capturing function execution or no functions were executed."
            )

        df = pd.DataFrame(line_timings)

        # Create figure with appropriate height
        fig_height = max(8, len(df) * 0.25)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        # Create heatmap matrix
        time_values = np.array(df["time"].values).reshape(-1, 1)

        # Color mapping - use professional red gradient
        im = ax.imshow(time_values, cmap="Reds", aspect="auto", alpha=0.8)

        # Set labels with script name, line number, and hits count
        ax.set_yticks(range(len(df)))
        labels = []
        for _, row in df.iterrows():
            script_name = row.get("script_name", "unknown")
            line_number = row["line_number"]
            hits = row["hits"]
            line_content = row["line_contents"]

            # Truncate line content to fit
            if len(line_content) > 35:
                line_content = line_content[:32] + "..."

            # Format: script.py:123 (42 hits) | code content
            label = f"{script_name}:{line_number:3d} ({hits:,} hits) | {line_content}"
            labels.append(label)

        ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
        ax.set_xticks([])
        ax.set_title("Line-by-Line Execution Time Heatmap", fontweight="bold", pad=20)

        # Add time annotations on the heatmap
        max_time = df["time"].max()
        for i, (_, row) in enumerate(df.iterrows()):
            time_val = row["time"]

            # Time annotation with better contrast
            color = "white" if time_val > max_time * 0.5 else "black"
            ax.text(
                0,
                i,
                f"{time_val:.3f}s",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=8,
            )

        # Single colorbar with proper positioning
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label("Execution Time (seconds)", rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=8)

        # Add summary statistics positioned to avoid colorbar overlap
        total_time = df["time"].sum()
        total_hits = df["hits"].sum()
        avg_time_per_hit = total_time / total_hits if total_hits > 0 else 0

        stats_text = f"""Performance Summary:
Total Time: {total_time:.3f}s
Total Hits: {total_hits:,}
Avg/Hit: {avg_time_per_hit:.6f}s
Lines: {len(df)}

Format: script.py:line (hits) | code"""

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
            fontsize=8,
        )

        plt.tight_layout()
        return fig

    def _extract_line_timings(
        self, profiler_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract line timing data from profiler results."""
        line_timings = []

        # Process line_stats format (standardized format from line profiler)
        if "line_stats" in profiler_data:
            for line_key, stats in profiler_data["line_stats"].items():
                # Extract filename from line_key if available
                if isinstance(line_key, str) and ":" in line_key:
                    filename = line_key.split(":")[0]
                    script_name = filename.split("/")[-1]
                else:
                    script_name = "unknown"

                # Extract line content if available
                line_content = f"Line {stats.get('line_number', 'N/A')}"

                line_timings.append(
                    {
                        "script_name": script_name,
                        "line_number": stats.get("line_number", 0),
                        "hits": stats.get("hits", 0),
                        "time": stats.get("time", 0),
                        "per_hit": stats.get("time_per_hit", 0),
                        "line_contents": line_content,
                    }
                )
        else:
            # Fail fast if expected data format is not present
            raise RuntimeError(
                f"Line profiler data missing expected 'line_stats' format. "
                f"Available keys: {list(profiler_data.keys())}"
            )

        # Sort by execution time (hottest lines first)
        line_timings.sort(key=lambda x: x["time"], reverse=True)

        # Limit to top 50 lines for readability
        return line_timings[:50]
