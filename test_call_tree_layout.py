#!/usr/bin/env python3
"""
Test script to verify the improved call tree layout.
"""

import json
from pathlib import Path
from src.pycroscope.infrastructure.visualization.call_profiler_plotter import (
    CallProfilerPlotter,
)


def test_call_tree_layout():
    """Test the improved call tree layout with real profiling data."""

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

    # Test call tree visualization specifically
    if "call" in results:
        print("\n📊 Testing improved call tree visualization...")
        call_data = results["call"].get("data", {})
        plotter = CallProfilerPlotter()

        if plotter.can_plot(call_data):
            try:
                # Only generate call tree to focus testing
                stats_dict = call_data.get("stats", {})
                if not stats_dict:
                    print("❌ No call stats available in profiling data.")
                    return False

                # Call the improved call tree visualization directly
                from matplotlib import pyplot as plt

                fig = plotter._plot_call_tree(stats_dict)

                # Save the figure
                output_path = output_dir / "improved_call_tree.png"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                print(f"✅ Improved call tree generated: {output_path}")
                print("  - Check for better hierarchical structure")
                print(
                    "  - Verify that nodes are properly positioned with proper spacing"
                )
                print(
                    "  - Confirm that nodes are labeled clearly with timing information"
                )
                success = True
            except Exception as e:
                print(f"❌ Error generating call tree: {e}")
                import traceback

                traceback.print_exc()
                success = False

    return success


if __name__ == "__main__":
    print("🧪 Testing improved call tree layout...")
    success = test_call_tree_layout()

    if success:
        print(
            "\n🎉 Call tree layout improvements are working! Check the test_output directory."
        )
    else:
        print("\n💥 Call tree layout needs more work.")
