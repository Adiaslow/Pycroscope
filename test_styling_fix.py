#!/usr/bin/env python3
"""
Test script to verify styling fixes for matplotlib plots.
"""

import json
from pathlib import Path
from src.pycroscope.infrastructure.visualization.call_profiler_plotter import (
    CallProfilerPlotter,
)
from src.pycroscope.infrastructure.visualization.line_profiler_plotter import (
    LineProfilerPlotter,
)
from src.pycroscope.infrastructure.visualization.memory_profiler_plotter import (
    MemoryProfilerPlotter,
)


def test_plot_styling():
    """Test plot styling fixes with real profiling data."""

    # Load the session data
    session_path = Path("profiling_results/session.json")
    if not session_path.exists():
        print("❌ Session data not found. Please run profiling first.")
        return False

    with open(session_path, "r") as f:
        session_data = json.load(f)

    results = session_data.get("results", {})
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    success = True

    # Test call profiler styling
    if "call" in results:
        print("\n📊 Testing call profiler styling...")
        call_data = results["call"].get("data", {})
        plotter = CallProfilerPlotter()

        if plotter.can_plot(call_data):
            try:
                charts = plotter.generate_plots(call_data, output_dir)
                print(f"✅ Call profiler charts generated: {', '.join(charts.keys())}")
            except Exception as e:
                print(f"❌ Error generating call profiler charts: {e}")
                success = False

    # Test line profiler styling
    if "line" in results:
        print("\n📊 Testing line profiler styling...")
        line_data = results["line"].get("data", {})
        plotter = LineProfilerPlotter()

        if plotter.can_plot(line_data):
            try:
                charts = plotter.generate_plots(line_data, output_dir)
                print(f"✅ Line profiler charts generated: {', '.join(charts.keys())}")
            except Exception as e:
                print(f"❌ Error generating line profiler charts: {e}")
                success = False

    # Test memory profiler styling
    if "memory" in results:
        print("\n📊 Testing memory profiler styling...")
        memory_data = results["memory"].get("data", {})
        plotter = MemoryProfilerPlotter()

        if plotter.can_plot(memory_data):
            try:
                charts = plotter.generate_plots(memory_data, output_dir)
                print(
                    f"✅ Memory profiler charts generated: {', '.join(charts.keys())}"
                )
            except Exception as e:
                print(f"❌ Error generating memory profiler charts: {e}")
                success = False

    return success


if __name__ == "__main__":
    print("🧪 Testing plot styling fixes...")
    success = test_plot_styling()

    if success:
        print(
            "\n🎉 Styling fixes are working! Check the test_output directory for the generated plots."
        )
    else:
        print("\n💥 Styling fixes need more work.")
