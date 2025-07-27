#!/usr/bin/env python3
"""
Pycroscope 1.0 Usage Example

This example shows how to use Pycroscope to profile real code with integrated pattern analysis.
Users can easily modify this template for their own profiling needs.

QUICK START:
1. Replace the sample workload import/execution with your own code
2. Adjust profiling configuration as needed
3. Run: python usage_example.py
4. Check the 'profiling_results' directory for reports, charts, and pattern analysis

NEW IN 1.0: Pattern analysis runs automatically alongside profiling!
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pycroscope


def run_sample_workload():
    """
    REPLACE THIS SECTION WITH YOUR OWN CODE

    This function runs our sample workload script.
    Users should replace this with their own code to profile.

    Examples of what you might put here:
    - Import and call your own functions
    - Run your data processing pipeline
    - Execute your machine learning training
    - Run your web scraping script
    - Call your API processing logic
    """
    # Import the sample workload (replace this with your own imports)
    from sample_workload import mixed_workload

    print("Executing workload to be profiled...")
    print("   (Replace this section with your own code)")
    print()

    # Execute the workload (replace this with your own function calls)
    results = mixed_workload()

    return results


def main():
    """Main profiling workflow - users typically won't need to modify this."""
    print("=" * 70)
    print("🔬 PYCROSCOPE 1.0 USAGE EXAMPLE")
    print("=" * 70)
    print("This example demonstrates comprehensive profiling with pattern analysis.")
    print("Pattern analysis runs automatically - no extra setup required!")
    print("See comments in the code for customization instructions.")
    print("-" * 70)

    # Configure output directory
    output_dir = Path("profiling_results")

    # PROFILING CONFIGURATION
    # Pycroscope 1.0 includes pattern analysis by default!
    # Users can adjust these settings based on their needs:
    print("Profiling Configuration (Pattern Analysis Enabled by Default):")
    print("   [ENABLED] line_profiling: Detailed line-by-line execution timing")
    print("   [ENABLED] call_profiling: Function call timing and hierarchy")
    print("   [ENABLED] memory_profiling: Memory usage tracking over time")
    print("   [NEW] analyze_patterns: Anti-pattern detection")
    print("   [ENABLED] generate_reports: Comprehensive reports with analysis")
    print("   [ENABLED] create_visualizations: Charts and graphs")
    print()

    # CREATE OUTPUT DIRECTORY
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    # RUN PROFILING WITH INTEGRATED PATTERN ANALYSIS
    print("Starting comprehensive profiling session...")
    print("   • Performance profiling: Measuring execution speed and memory")
    print("   • Pattern analysis: Detecting code quality issues")
    print("   • Hotspot correlation: Linking patterns to performance impact")
    print()

    session_start = time.time()

    try:
        # THE MAIN PROFILING - Pattern analysis happens automatically!
        @pycroscope.profile(output_dir=output_dir)
        def run_profiling_session():
            print(
                "✨ Profilers and pattern analysis are active - executing your code..."
            )
            print()

            # EXECUTE YOUR CODE HERE
            # Replace run_sample_workload() with your own function calls
            workload_results = run_sample_workload()

            print()
            print("Code execution completed!")
            return workload_results

        # Run the profiling session
        workload_results = run_profiling_session()

        session_time = time.time() - session_start

        # RESULTS SUMMARY
        print("-" * 70)
        print("PROFILING RESULTS")
        print(f"Total profiling time: {session_time:.3f} seconds")
        print(f"Session output directory: {output_dir}")
        print()

        # Check generated outputs
        all_files = list(output_dir.iterdir()) if output_dir.exists() else []
        generated_files = [f for f in all_files if f.is_file()]

        if not generated_files:
            print("WARNING: No output files were generated")
        else:
            print("Generated Files:")
            for generated_file in sorted(generated_files):
                if "pattern_analysis" in generated_file.name:
                    print(f"   [ANALYSIS] {generated_file.name} (Pattern Analysis)")
                elif "integrated" in generated_file.name:
                    print(f"   [REPORT] {generated_file.name} (Combined Analysis)")
                elif generated_file.suffix == ".json":
                    print(f"   [DATA] {generated_file.name} (Data)")
                elif generated_file.suffix == ".md":
                    print(f"   [REPORT] {generated_file.name} (Report)")
                elif generated_file.suffix == ".png":
                    print(f"   [CHART] {generated_file.name} (Chart)")
                else:
                    print(f"   [FILE] {generated_file.name}")

        print()
        print("Your Code Results:")
        if workload_results:
            print("   [SUCCESS] Workload completed successfully")
            if isinstance(workload_results, dict):
                for key, value in workload_results.items():
                    if isinstance(value, (int, float)):
                        if "time" in key:
                            print(f"   [TIME] {key}: {value:.3f}s")
                        else:
                            print(f"   [DATA] {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"   [LIST] {key}: {len(value)} items")
                    else:
                        print(f"   [OTHER] {key}: {value}")
        else:
            print("   [WARNING] No results returned from workload")

        # Show pattern analysis insights if available
        pattern_report_path = output_dir / "pattern_analysis_report.json"
        integrated_report_path = output_dir / "integrated_analysis_report.json"

        if pattern_report_path.exists():
            print()
            print("PATTERN ANALYSIS INSIGHTS:")
            try:
                import json

                with open(pattern_report_path, "r") as f:
                    pattern_data = json.load(f)

                summary = pattern_data.get("summary", {})
                total_patterns = summary.get("total_patterns_detected", 0)

                if total_patterns > 0:
                    print(
                        f"   [WARNING] Found {total_patterns} code patterns across {summary.get('total_files_analyzed', 0)} files"
                    )

                    # Show pattern distribution
                    pattern_dist = summary.get("pattern_distribution", {})
                    if pattern_dist:
                        print("   Pattern types detected:")
                        for pattern_type, count in sorted(
                            pattern_dist.items(), key=lambda x: x[1], reverse=True
                        ):
                            print(f"      • {pattern_type}: {count}")

                    # Show top issues
                    top_issues = pattern_data.get("top_issues", [])
                    if top_issues:
                        print("   [PRIORITY] Top priority issues:")
                        for i, issue in enumerate(top_issues[:3], 1):
                            severity_prefix = {
                                "low": "[LOW]",
                                "medium": "[MEDIUM]",
                                "high": "[HIGH]",
                                "critical": "[CRITICAL]",
                            }.get(issue.get("severity", "medium"), "[MEDIUM]")
                            correlated = (
                                " [PERF-CORRELATED]"
                                if issue.get("performance_correlated")
                                else ""
                            )
                            print(
                                f"      {i}. {severity_prefix} {issue.get('pattern_type', 'Unknown')}{correlated}"
                            )

                    # Show recommendations
                    recommendations = pattern_data.get("recommendations", [])
                    if recommendations:
                        print("   [RECOMMENDATIONS] Key recommendations:")
                        for i, rec in enumerate(recommendations[:2], 1):
                            print(f"      {i}. {rec}")
                else:
                    print(
                        "   [SUCCESS] No significant code patterns detected - good job!"
                    )

            except Exception as e:
                print(f"   [ERROR] Could not parse pattern analysis results: {e}")

        print()
        print("=" * 70)
        print("COMPREHENSIVE ANALYSIS COMPLETE!")
        print()
        print("WHAT YOU GET WITH PYCROSCOPE 1.0:")
        print("   [FEATURE] Performance hotspot identification")
        print("   [FEATURE] Detailed execution timing and memory usage")
        print("   [NEW] Anti-pattern detection")
        print("   [FEATURE] Correlation between patterns and performance issues")
        print("   [FEATURE] Beautiful visualizations and comprehensive reports")
        print()
        print("NEXT STEPS:")
        print(f"   1. Open the '{output_dir}' directory to explore results")
        print("   2. Review pattern_analysis_report.json for code quality insights")
        print("   3. Check integrated_analysis_report.json for combined findings")
        print("   4. View .png chart files to visualize performance data")
        print("   5. Replace sample_workload with your own code to profile")
        print()
        print("PATTERN ANALYSIS FEATURES:")
        print("   • Detects nested loops and O(n²) complexity issues")
        print("   • Identifies functions with high complexity or too many parameters")
        print("   • Finds dead code and unused imports")
        print("   • Highlights inefficient data structure usage")
        print("   • Prioritizes issues found in performance hotspots")
        print()
        print("CUSTOMIZATION:")
        print("   - Modify run_sample_workload() to execute your own code")
        print("   - Use @pycroscope.profile() decorator on specific functions")
        print("   - Adjust pattern detection thresholds in ProfileConfig")
        print(
            "   - Focus analysis with .with_performance_focus() or .with_maintainability_focus()"
        )
        print("=" * 70)

        return True

    except Exception as e:
        print()
        print("PROFILING FAILED!")
        print()
        print("Full error details:")
        # Re-raise the exception to show the full traceback and let the program fail
        raise


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
