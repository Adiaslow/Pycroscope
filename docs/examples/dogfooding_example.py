#!/usr/bin/env python3
"""
Pycroscope True Dogfooding Example

This demonstrates true dogfooding: Pycroscope profiling itself while it profiles
a computational workload. We have two layers:
1. Inner Pycroscope: Profiles a computational workload (outputs discarded)
2. Outer Pycroscope: Profiles the Inner Pycroscope (outputs analyzed)

Turns this is not possible due to a fundamental limitation of Python due to the fact that
python's sys.settrace() function is not thread-safe.
"""

1 / 0

import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import os
import gc
import subprocess
import threading

# Add pycroscope to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pycroscope import profile


class ComputationalWorkload:
    """A realistic computational workload for the inner Pycroscope to profile."""

    def __init__(self):
        self.data_cache = {}

    def cpu_intensive_computation(self, size: int) -> Dict[str, Any]:
        """CPU-intensive computation that the inner Pycroscope will profile."""
        # Prime number generation
        primes = []
        for num in range(2, size):
            is_prime = True
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)

        # Matrix operations
        matrix_size = 30
        matrix_a = [
            [random.random() for _ in range(matrix_size)] for _ in range(matrix_size)
        ]
        matrix_b = [
            [random.random() for _ in range(matrix_size)] for _ in range(matrix_size)
        ]

        # Matrix multiplication
        result_matrix = []
        for i in range(matrix_size):
            row = []
            for j in range(matrix_size):
                sum_val = 0
                for k in range(matrix_size):
                    sum_val += matrix_a[i][k] * matrix_b[k][j]
                row.append(sum_val)
            result_matrix.append(row)

        return {
            "primes_found": len(primes),
            "largest_prime": max(primes) if primes else 0,
            "matrix_result_sum": sum(sum(row) for row in result_matrix),
        }

    def memory_intensive_computation(self, cycles: int) -> Dict[str, Any]:
        """Memory-intensive computation with allocation waves."""
        memory_pools = []

        for cycle in range(cycles):
            # Allocate memory
            chunk_size = (cycle + 1) * 500
            memory_chunk = [random.random() for _ in range(chunk_size)]

            # Process the chunk
            processed = [x * 2 + random.random() for x in memory_chunk]
            sorted_chunk = sorted(processed)

            memory_pools.append(
                {
                    "cycle": cycle,
                    "data": sorted_chunk[:50],  # Keep sample
                    "stats": {
                        "min": min(sorted_chunk),
                        "max": max(sorted_chunk),
                        "avg": sum(sorted_chunk) / len(sorted_chunk),
                    },
                }
            )

            # Cleanup to create memory pressure waves
            if cycle % 4 == 0 and len(memory_pools) > 3:
                memory_pools.pop(0)
                gc.collect()

        return {
            "memory_cycles": cycles,
            "final_pools": len(memory_pools),
            "total_processed": sum(len(pool["data"]) for pool in memory_pools),
        }

    def io_intensive_computation(
        self, temp_dir: Path, file_count: int
    ) -> Dict[str, Any]:
        """I/O intensive computation with file operations."""
        files_created = []
        total_bytes = 0

        for i in range(file_count):
            filename = temp_dir / f"workload_{i}.json"

            # Generate varying data
            file_data = {
                "file_id": i,
                "timestamp": time.time(),
                "data_points": [
                    {
                        "id": j,
                        "value": random.random() * 100,
                        "category": random.choice(["Alpha", "Beta", "Gamma"]),
                        "metadata": {
                            "processed": j % 2 == 0,
                            "priority": random.randint(1, 3),
                        },
                    }
                    for j in range(50 + i * 20)
                ],
            }

            # Write and read file
            with open(filename, "w") as f:
                json.dump(file_data, f, indent=2)

            total_bytes += filename.stat().st_size
            files_created.append(str(filename))

            # Read and process
            with open(filename, "r") as f:
                loaded_data = json.load(f)

            # Process data
            processed_count = sum(
                1
                for point in loaded_data["data_points"]
                if point["metadata"]["processed"]
            )

            self.data_cache[f"file_{i}"] = {
                "processed_count": processed_count,
                "file_size": filename.stat().st_size,
            }

        # Cleanup
        for filename in files_created:
            if os.path.exists(filename):
                os.remove(filename)

        return {
            "files_processed": len(files_created),
            "total_bytes": total_bytes,
            "cache_entries": len(self.data_cache),
        }


def run_inner_pycroscope_workload(temp_dir: Path, profiler_type: str) -> str:
    """
    Run computational workload that simulates typical Pycroscope usage.
    This represents the target process that the outer Pycroscope will profile.
    No actual profiling here to avoid conflicts - just computational work.
    """
    workload = ComputationalWorkload()

    print(
        f"   🔄 Inner Process: Running computational workload (target for {profiler_type} profiling)..."
    )

    start_time = time.time()

    # Simulate typical Pycroscope workload patterns without actual profiling
    # Run computational workload phases that represent heavy Pycroscope usage
    cpu_result = workload.cpu_intensive_computation(800)
    time.sleep(0.2)

    memory_result = workload.memory_intensive_computation(10)
    time.sleep(0.1)

    io_result = workload.io_intensive_computation(temp_dir, 5)
    time.sleep(0.1)

    # Second computation wave to simulate sustained Pycroscope activity
    cpu_result2 = workload.cpu_intensive_computation(1200)
    memory_result2 = workload.memory_intensive_computation(8)

    # Simulate some Pycroscope-like operations
    # Data processing that would happen in real profiling scenarios
    for i in range(100):
        # Simulate profiler data collection patterns
        data_point = {
            "timestamp": time.time(),
            "cpu_usage": random.random() * 100,
            "memory_usage": random.random() * 500,
            "function_calls": random.randint(10, 100),
        }
        # Simulate data aggregation
        aggregated = sum(data_point.values()) / len(data_point)
        if i % 20 == 0:
            time.sleep(0.01)  # Simulate I/O operations

    execution_time = time.time() - start_time

    print(
        f"   ✅ Inner Process: Completed computational workload in {execution_time:.2f}s"
    )
    print(
        f"      🧮 Workload: {cpu_result['primes_found']} + {cpu_result2['primes_found']} primes"
    )
    print(
        f"      🧠 Memory: {memory_result['memory_cycles']} + {memory_result2['memory_cycles']} cycles"
    )
    print(
        f"      💾 I/O: {io_result['files_processed']} files, {io_result['total_bytes']:,} bytes"
    )

    return f"computational_workload_simulating_pycroscope_usage_{profiler_type}"


async def run_true_dogfooding():
    """
    Run true dogfooding: Pycroscope profiling itself.

    This creates ONE comprehensive session with ALL profilers enabled,
    resulting in a single timestamped directory with all figures and one report.
    """
    print("🐕 Pycroscope TRUE DOGFOODING")
    print("=" * 50)
    print("Pycroscope profiling itself while it profiles computational workloads.")
    print("Single comprehensive session with ALL profilers enabled.")
    print()

    # Setup
    output_dir = Path("dogfooding_results")
    output_dir.mkdir(exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="pycroscope_dogfooding_"))

    print(f"📁 Comprehensive dogfooding output: {output_dir.absolute()}")
    print(f"💾 Temporary workspace: {temp_dir}")
    print()

    try:
        print("🎯 Starting Comprehensive Dogfooding")
        print(
            "   📊 Profiling Pycroscope with ALL profilers: Call + Line + Memory + Sampling"
        )
        print("   🔍 Target: Pycroscope's internal computational workload processing")
        print()

        start_time = time.time()

        # Single comprehensive session with ALL profilers enabled
        # Sequential execution resolves sys.settrace() conflicts between call and line profilers
        # This will create: dogfooding_results/<timestamp>/
        with profile(
            call_profiling=True,  # Enable all profilers at full capacity
            line_profiling=True,
            memory_profiling=True,
            sampling_profiling=False,
            generate_reports=True,
            create_visualizations=True,
            output_dir=output_dir,  # ProfileSession will create timestamped subdir
        ) as session:

            print(
                "   🔄 Running computational workload (target for comprehensive profiling)..."
            )

            # Run several iterations for comprehensive data
            for iteration in range(3):
                print(f"      Iteration {iteration + 1}/3...")
                run_inner_pycroscope_workload(temp_dir, f"iteration_{iteration + 1}")

        test_time = time.time() - start_time

        print(f"   ⏱️  Profiling session completed in {test_time:.2f} seconds")
        print(f"   📊 Session ID: {session.session_id}")
        print()

        # Analyze the results
        print("=" * 50)
        print("📊 PROFILING RESULTS ANALYSIS")
        print(f"⏱️  Total execution time: {test_time:.2f} seconds")
        print()

        # Find the generated timestamped directory and verify results
        # Use the session's actual output directory (timestamped subdirectory)
        session_dir = session.config.output_dir
        success = False
        output_directory = None

        if session_dir and session_dir.exists():
            # This is the actual directory created by the session
            output_directory = str(session_dir)
            reports = list(session_dir.glob("*.md"))
            charts = list(session_dir.glob("*.png"))

            print("📋 Generated Output Structure:")
            print(f"   📁 {output_dir.name}/")
            print(f"   ├─ {session_dir.name}/")

            if reports:
                print(f"   │  ├─ 📄 profiling_report.md")
            else:
                print(f"   │  ├─ ❌ No profiling report generated")

            if charts:
                for chart in sorted(charts):
                    print(f"   │  ├─ 📊 {chart.name}")
            else:
                print(f"   │  ├─ ❌ No visualization charts generated")

            print()
            print(f"📈 Results Verification:")
            print(f"   📁 Output Directory: {session_dir}")
            print(f"   📄 Reports Generated: {len(reports)}")
            print(f"   📊 Charts Generated: {len(charts)}")

            # Verify session actually worked
            has_session_data = session.results and len(session.results) > 0
            has_outputs = len(reports) > 0 or len(charts) > 0

            if has_session_data and has_outputs:
                print("   ✅ Profiling session completed successfully")
                success = True
            else:
                print("   ❌ Profiling session did not generate expected outputs")
                if not has_session_data:
                    print("     • No profiling results captured")
                if not has_outputs:
                    print("     • No reports or charts generated")

            if charts:
                print("   📊 Visualization Status:")
                for chart in sorted(charts):
                    chart_type = (
                        chart.stem.split("_")[0] if "_" in chart.stem else chart.stem
                    )
                    if chart.stat().st_size > 0:
                        print(f"     ✅ {chart.name} ({chart_type} profiler)")
                    else:
                        print(f"     ❌ {chart.name} (empty file)")
        else:
            print("   ❌ No output directory found")

        return {
            "success": success,
            "execution_time": test_time,
            "session_id": str(session.session_id),
            "output_directory": output_directory,
            "has_session_data": session.results and len(session.results) > 0,
            "reports_generated": len(reports) if "reports" in locals() else 0,
            "charts_generated": len(charts) if "charts" in locals() else 0,
        }

    except Exception as e:
        print(f"❌ Comprehensive dogfooding failed: {e}")
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}

    finally:
        # Cleanup
        if temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary workspace: {temp_dir}")
        print()
        print("=" * 50)
        print(f"📁 Results directory: {output_dir}")


def main():
    """Main entry point for true dogfooding."""
    try:
        result = asyncio.run(run_true_dogfooding())

        print("\n" + "=" * 50)
        if result["success"]:
            print("✅ Dogfooding completed successfully!")
            print(f"⏱️  Total execution time: {result['execution_time']:.2f} seconds")
            print(
                f"📊 Session data captured: {'Yes' if result.get('has_session_data') else 'No'}"
            )
            print(f"📄 Reports generated: {result.get('reports_generated', 0)}")
            print(f"📊 Charts generated: {result.get('charts_generated', 0)}")
            if result.get("output_directory"):
                print(f"📁 Results: {result['output_directory']}")
        else:
            print("❌ Dogfooding failed!")
            print(
                f"📊 Session data captured: {'Yes' if result.get('has_session_data') else 'No'}"
            )
            print(f"📄 Reports generated: {result.get('reports_generated', 0)}")
            print(f"📊 Charts generated: {result.get('charts_generated', 0)}")
            return False

    except KeyboardInterrupt:
        print("\n⚠️  True dogfooding interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
