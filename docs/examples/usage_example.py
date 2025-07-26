#!/usr/bin/env python3
"""
Pycroscope 2.0 Usage Example

This example shows how to use Pycroscope to profile real code.
Users can easily modify this template for their own profiling needs.

QUICK START:
1. Replace the sample workload import/execution with your own code
2. Adjust profiling configuration as needed
3. Run: python usage_example.py
4. Check the 'profiling_results' directory for reports and charts
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pycroscope import profile


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

    print("🎯 Executing workload to be profiled...")
    print("   (Replace this section with your own code)")
    print()

    # Execute the workload (replace this with your own function calls)
    results = mixed_workload()

    return results


def main():
    """Main profiling workflow - users typically won't need to modify this."""
    print("=" * 60)
    print("🔬 PYCROSCOPE USAGE EXAMPLE")
    print("=" * 60)
    print("This example demonstrates how to profile code with Pycroscope.")
    print("See comments in the code for customization instructions.")
    print("-" * 60)

    # Configure output directory
    output_dir = "profiling_results"

    # PROFILING CONFIGURATION
    # Users can adjust these settings based on their needs:
    profiling_config = {
        # Enable/disable specific profilers
        "line_profiling": True,  # Line-by-line execution timing
        "call_profiling": True,  # Function call timing and hierarchy
        "memory_profiling": True,  # Memory usage over time
        "sampling_profiling": False,  # Sampling profiler removed
        # Output configuration
        "output_dir": output_dir,
        "generate_reports": True,
        "generate_charts": True,
        # Profiler-specific settings
        "line_profile_functions": [],  # Empty = profile all functions
        "memory_interval": 0.1,  # Memory sampling interval in seconds
        "sampling_interval": 0.01,  # CPU sampling interval in seconds
    }

    print("📊 Profiling Configuration:")
    for key, value in profiling_config.items():
        if isinstance(value, bool) and value:
            print(f"   ✅ {key}: {value}")
        elif isinstance(value, bool):
            print(f"   ❌ {key}: {value}")
        else:
            print(f"   🔧 {key}: {value}")
    print()

    # CREATE OUTPUT DIRECTORY
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}")
    print()

    # RUN PROFILING
    print("🚀 Starting profiling session...")
    session_start = time.time()

    try:
        # THE MAIN PROFILING CONTEXT
        # This is where the magic happens!
        with profile(**profiling_config) as session:
            print("✨ Profilers are active - executing your code...")
            print()

            # EXECUTE YOUR CODE HERE
            # Replace run_sample_workload() with your own function calls
            workload_results = run_sample_workload()

            print()
            print("✅ Code execution completed!")

        session_time = time.time() - session_start

        # RESULTS SUMMARY
        print("-" * 60)
        print("📈 PROFILING RESULTS")
        print(f"⏱️  Total profiling time: {session_time:.3f} seconds")
        print(f"📁 Session output directory: {session.config.output_dir}")
        print()

        # Verify generated outputs
        output_dir_path = Path(session.config.output_dir)

        # Get actual generated files from session - FAIL FAST if expected outputs are missing
        all_files = list(output_dir_path.iterdir()) if output_dir_path.exists() else []
        generated_files = [f for f in all_files if f.is_file()]

        # Check if we got the expected outputs based on configuration
        expected_reports = profiling_config["generate_reports"]
        expected_charts = profiling_config["generate_charts"]

        if expected_reports and not generated_files:
            raise RuntimeError(
                "FAIL FAST: Reports were requested but no output files were generated"
            )

        if expected_charts and not generated_files:
            raise RuntimeError(
                "FAIL FAST: Charts were requested but no output files were generated"
            )

        # Show what was actually generated
        print("📄 Generated Files:")
        for generated_file in sorted(generated_files):
            print(f"   📄 {generated_file.name}")

        print()
        print("🎯 Your Code Results:")
        if workload_results:
            print("   ✅ Workload completed successfully")
            if isinstance(workload_results, dict):
                for key, value in workload_results.items():
                    if isinstance(value, (int, float)):
                        if "time" in key:
                            print(f"   ⏱️  {key}: {value:.3f}s")
                        else:
                            print(f"   📊 {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"   📋 {key}: {len(value)} items")
                    else:
                        print(f"   📄 {key}: {value}")
        else:
            print("   ⚠️  No results returned from workload")

        print()
        print("=" * 60)
        print("🎉 PROFILING COMPLETE!")
        print()
        print("💡 NEXT STEPS:")
        print(f"   1. Check the '{output_dir}' directory for detailed reports")
        print("   2. Open the .png chart files to visualize performance")
        print("   3. Review .txt reports for detailed profiling data")
        print("   4. Replace sample_workload with your own code to profile")
        print()
        print("📚 CUSTOMIZATION:")
        print("   - Modify run_sample_workload() to execute your own code")
        print("   - Adjust profiling_config to enable/disable profilers")
        print("   - Change output_dir to your preferred location")
        print("=" * 60)

        return True

    except Exception as e:
        print()
        print("❌ PROFILING FAILED!")
        print()
        print("🔧 Full error details:")
        # Re-raise the exception to show the full traceback and let the program fail
        raise


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
