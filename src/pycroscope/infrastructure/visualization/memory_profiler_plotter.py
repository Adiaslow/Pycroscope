"""
Memory profiler visualization implementation.

Implements industry-standard memory_profiler visualization:
Memory Timeline - Critical for memory leak detection and allocation analysis
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .interfaces import ProfilerPlotter, StyleManager


class MemoryProfilerPlotter(ProfilerPlotter):
    """
    Professional memory_profiler data visualization.

    Generates industry-standard memory timeline charts used by
    SREs and performance engineers for memory analysis.
    """

    def can_plot(self, profiler_data: Dict[str, Any]) -> bool:
        """Check if data contains memory profiler results."""
        return "samples" in profiler_data and profiler_data["samples"]

    def get_plot_types(self) -> List[str]:
        """Get available plot types for memory profiler."""
        return ["memory_timeline"]

    def generate_plots(
        self, profiler_data: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Path]:
        """Generate all memory profiler visualizations."""
        if not self.can_plot(profiler_data):
            raise RuntimeError(
                f"Cannot generate memory profiler plots: Invalid data structure. "
                f"Data keys: {list(profiler_data.keys())}"
            )

        plots = {}

        # Use style context manager to isolate styling
        with StyleManager.apply_professional_style():
            # Memory Timeline
            timeline_fig = self._plot_memory_timeline(profiler_data)
            timeline_path = output_dir / "memory_timeline.png"
            StyleManager.save_figure(timeline_fig, timeline_path)
            plots["memory_timeline"] = timeline_path

        return plots

    def _plot_memory_timeline(self, profiler_data: Dict[str, Any]) -> Figure:
        """
        Create memory usage timeline from memory_profiler output.

        Industry standard for memory leak detection and allocation
        pattern analysis used by SREs and performance engineers.
        """
        samples = profiler_data["samples"]

        if not samples:
            raise RuntimeError(
                "Memory profiler generated no sample data. This indicates the profiler "
                "was not properly capturing memory usage or monitoring failed."
            )

        # Extract memory data
        memory_data = []
        for i, sample in enumerate(samples):
            memory_mb = sample.get("rss_mb", 0)
            timestamp = sample.get("timestamp", i)
            memory_data.append(
                {"index": i, "memory_mb": memory_mb, "timestamp": timestamp}
            )

        df = pd.DataFrame(memory_data)

        if len(df) < 2:
            raise RuntimeError(
                f"Memory profiler captured insufficient data: {len(df)} samples. "
                "Need at least 2 samples to generate timeline visualization."
            )

        # Calculate memory increments
        df["increment"] = df["memory_mb"].diff().fillna(0)

        # Normalize timestamps to start from 0
        start_time = df["timestamp"].iloc[0]
        df["time_delta"] = df["timestamp"] - start_time

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Memory usage plot
        colors = StyleManager.get_color_palette()
        ax1.plot(
            df["time_delta"],
            df["memory_mb"],
            color=colors[0],
            linewidth=2,
            label="Memory Usage",
        )
        ax1.fill_between(df["time_delta"], df["memory_mb"], alpha=0.3, color=colors[0])

        # Highlight memory spikes
        mean_memory = df["memory_mb"].mean()
        std_memory = df["memory_mb"].std()
        spike_threshold = mean_memory + 2 * std_memory

        spikes = df[df["memory_mb"] > spike_threshold]
        if not spikes.empty:
            ax1.scatter(
                spikes["time_delta"],
                spikes["memory_mb"],
                color="red",
                s=50,
                alpha=0.8,
                label="Memory Spikes",
                zorder=5,
            )

        ax1.set_ylabel("Memory Usage (MB)")
        ax1.set_title("Memory Usage Over Execution", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add peak memory annotation
        peak_idx = df["memory_mb"].idxmax()
        peak_memory = df.loc[peak_idx, "memory_mb"]
        peak_time = df.loc[peak_idx, "time_delta"]

        ax1.annotate(
            f"Peak: {peak_memory:.1f} MB",
            xy=(peak_time, peak_memory),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            fontsize=10,
            fontweight="bold",
        )

        # Create candlestick-style cumulative memory change bars with binning
        n_bins = 40  # Fixed number of bins for cleaner visualization
        max_time = df["time_delta"].max()
        bin_width = max_time / n_bins

        # Initialize cumulative tracking
        cumulative_position = 0
        bin_data = []

        for i in range(n_bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width
            bin_center = bin_start + bin_width / 2

            # Find all increments in this time bin
            bin_mask = (df["time_delta"] >= bin_start) & (df["time_delta"] < bin_end)
            bin_increments = df[bin_mask]["increment"].sum()

            if abs(bin_increments) > 0.001:  # Only include bins with meaningful changes
                bin_data.append(
                    {
                        "center": bin_center,
                        "increment": bin_increments,
                        "cumulative_start": cumulative_position,
                        "cumulative_end": cumulative_position + bin_increments,
                    }
                )
                cumulative_position += bin_increments

        # Plot candlestick-style bars
        for bin_info in bin_data:
            bar_bottom = bin_info["cumulative_start"]
            bar_height = bin_info["increment"]
            bar_color = "red" if bar_height > 0 else "green"
            bar_alpha = 0.8

            # Create the bar starting from cumulative position
            ax2.bar(
                bin_info["center"],
                abs(bar_height),  # Always positive height
                bottom=bar_bottom if bar_height > 0 else bar_bottom + bar_height,
                color=bar_color,
                alpha=bar_alpha,
                width=bin_width * 0.8,  # Slightly narrower than bin for clarity
                edgecolor="black",
                linewidth=0.5,
            )

        # Add legend entries manually since we're using custom plotting
        ax2.bar([], [], color="red", alpha=0.8, label="Memory Increase")
        ax2.bar([], [], color="green", alpha=0.8, label="Memory Decrease")

        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=1)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Cumulative Memory Change (MB)")
        ax2.set_title(
            "Cumulative Memory Changes (Candlestick Style)", fontweight="bold"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Annotate the final cumulative change if significant
        if bin_data:
            final_cumulative = bin_data[-1]["cumulative_end"]
            if abs(final_cumulative) > 1.0:  # Only annotate if > 1MB total change
                ax2.annotate(
                    f"Net Change: {final_cumulative:+.1f} MB",
                    xy=(bin_data[-1]["center"], final_cumulative),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8
                    ),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
                )

        # Add summary statistics
        total_time = df["time_delta"].iloc[-1]
        min_memory = df["memory_mb"].min()
        max_memory = df["memory_mb"].max()
        avg_memory = df["memory_mb"].mean()
        memory_delta = max_memory - min_memory

        stats_text = f"""Memory Summary:
Duration: {total_time:.2f}s
Min: {min_memory:.1f} MB
Max: {max_memory:.1f} MB  
Avg: {avg_memory:.1f} MB
Delta: {memory_delta:.1f} MB
Samples: {len(df)}"""

        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            fontsize=9,
        )

        plt.tight_layout()
        return fig
