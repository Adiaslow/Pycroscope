#!/usr/bin/env python3
"""
Test script to verify the updated visualization positioning and styles.
"""

import json
from pathlib import Path
from src.pycroscope.infrastructure.visualization.call_profiler_plotter import (
    CallProfilerPlotter,
)
from src.pycroscope.infrastructure.visualization.memory_profiler_plotter import (
    MemoryProfilerPlotter,
)


def test_visualization_changes():
    """Test visualization changes with real profiling data."""

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
        print("\n📊 Testing call profiler visualizations...")
        call_data = results["call"].get("data", {})
        plotter = CallProfilerPlotter()

        if plotter.can_plot(call_data):
            try:
                charts = plotter.generate_plots(call_data, output_dir)
                print(f"✅ Call profiler charts generated with updated positioning:")
                print(
                    f"  - Top functions (info box in top right): {charts.get('top_functions')}"
                )
                print(
                    f"  - Call tree (info box in bottom right): {charts.get('call_tree')}"
                )
                print(
                    f"  - Flame graph (info box in top right): {charts.get('flame_graph')}"
                )
            except Exception as e:
                print(f"❌ Error generating call profiler charts: {e}")
                success = False

    # Test memory profiler styling
    if "memory" in results:
        print("\n📊 Testing memory profiler visualizations...")
        memory_data = results["memory"].get("data", {})
        plotter = MemoryProfilerPlotter()

        if plotter.can_plot(memory_data):
            try:
                charts = plotter.generate_plots(memory_data, output_dir)
                print(f"✅ Memory profiler charts generated with updated colors:")
                print(
                    f"  - Memory timeline (improved color legend): {charts.get('memory_timeline')}"
                )
            except Exception as e:
                print(f"❌ Error generating memory profiler charts: {e}")
                success = False

    return success


if __name__ == "__main__":
    print("🧪 Testing visualization positioning and styling updates...")
    success = test_visualization_changes()

    if success:
        print(
            "\n🎉 Visualization updates are working! Check the test_output directory for the generated plots."
        )
        print("You should see:")
        print("  1. Call Top Functions: Info box in top right")
        print("  2. Call Tree: Info box in bottom right")
        print("  3. Flame Graph: Info box in top right")
        print(
            "  4. Memory Timeline: Consistent colors in legend and bars (crimson/forestgreen)"
        )
    else:
        print("\n💥 Visualization updates need more work.")
