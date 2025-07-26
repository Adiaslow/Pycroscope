#!/usr/bin/env python3
"""
Sample Workload for Pycroscope Profiling

This script demonstrates various computational patterns that are interesting to profile:
- CPU-intensive calculations
- Memory allocation patterns
- File I/O operations
- Function call hierarchies
- Loops and recursion
- Data structure operations

Users can replace this entire script with their own code to profile.
"""

import time
import math
import random
import json
import tempfile
import os
from typing import List, Dict, Any


def fibonacci_recursive(n: int) -> int:
    """Recursive fibonacci - creates deep call stack for call profiling."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Iterative fibonacci - more efficient, shows different call pattern."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def cpu_intensive_calculation() -> float:
    """CPU-bound task - good for sampling profiler hotspot detection."""
    print("🔥 Running CPU-intensive calculations...")

    # Matrix multiplication simulation
    size = 200
    matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]

    result = 0.0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result += matrix_a[i][k] * matrix_b[k][j]

    # Mathematical computations
    for i in range(10000):
        result += math.sin(i) * math.cos(i) * math.sqrt(i + 1)

    return result


def memory_intensive_operations() -> List[Dict[str, Any]]:
    """Memory allocation patterns - good for memory profiler."""
    print("💾 Running memory-intensive operations...")

    # Large list allocation
    large_list = []
    for i in range(50000):
        large_list.append(
            {
                "id": i,
                "data": [random.random() for _ in range(20)],
                "metadata": {
                    "timestamp": time.time(),
                    "category": f"category_{i % 10}",
                    "tags": [f"tag_{j}" for j in range(i % 5 + 1)],
                },
            }
        )

    # Dictionary operations
    lookup_dict = {}
    for item in large_list:
        category = item["metadata"]["category"]
        if category not in lookup_dict:
            lookup_dict[category] = []
        lookup_dict[category].append(item)

    # Memory churn - allocate and deallocate
    temp_lists = []
    for i in range(100):
        temp_list = [random.random() for _ in range(1000)]
        temp_lists.append(temp_list)
        if len(temp_lists) > 10:
            temp_lists.pop(0)  # Remove oldest

    return large_list


def file_io_operations() -> Dict[str, Any]:
    """File I/O operations - shows I/O patterns."""
    print("📁 Running file I/O operations...")

    # Create temporary files for I/O testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Write multiple files
        file_data = {}
        for i in range(10):
            filename = os.path.join(temp_dir, f"test_file_{i}.json")
            data = {
                "file_id": i,
                "content": [random.random() for _ in range(1000)],
                "metadata": {"created": time.time()},
            }

            with open(filename, "w") as f:
                json.dump(data, f)

            file_data[filename] = data

        # Read files back
        read_data = {}
        for filename in file_data:
            with open(filename, "r") as f:
                read_data[filename] = json.load(f)

        # Process data
        total_values = 0
        for data in read_data.values():
            total_values += len(data["content"])

        return {"files_processed": len(read_data), "total_values": total_values}

    finally:
        # Cleanup
        for filename in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)


def nested_function_calls() -> int:
    """Complex function call hierarchy - good for call profiler."""
    print("🌳 Running nested function calls...")

    def level_1(n: int) -> int:
        if n <= 0:
            return 1
        return level_2(n - 1) + level_3(n - 1)

    def level_2(n: int) -> int:
        if n <= 0:
            return 2
        return level_4(n - 1) * 2

    def level_3(n: int) -> int:
        if n <= 0:
            return 3
        return level_4(n - 1) + level_5(n - 1)

    def level_4(n: int) -> int:
        if n <= 0:
            return 4
        return level_5(n - 1) + 1

    def level_5(n: int) -> int:
        return 5 + (n * 2 if n > 0 else 0)

    return level_1(8)


def data_processing_pipeline() -> Dict[str, float]:
    """Data processing with multiple stages - shows line-by-line execution."""
    print("⚙️  Running data processing pipeline...")

    # Stage 1: Generate raw data
    raw_data = [random.random() * 1000 for _ in range(10000)]

    # Stage 2: Filter data
    filtered_data = [x for x in raw_data if x > 100]

    # Stage 3: Transform data
    transformed_data = [math.log(x) if x > 1 else 0 for x in filtered_data]

    # Stage 4: Aggregate data
    sum_data = sum(transformed_data)
    avg_data = sum_data / len(transformed_data) if transformed_data else 0
    max_data = max(transformed_data) if transformed_data else 0
    min_data = min(transformed_data) if transformed_data else 0

    # Stage 5: Statistical calculations
    variance = sum((x - avg_data) ** 2 for x in transformed_data) / len(
        transformed_data
    )
    std_dev = math.sqrt(variance)

    return {
        "count": len(transformed_data),
        "sum": sum_data,
        "average": avg_data,
        "maximum": max_data,
        "minimum": min_data,
        "std_dev": std_dev,
    }


def mixed_workload() -> Dict[str, Any]:
    """Combined workload showing all patterns together."""
    print("🚀 Running mixed workload demonstration...")

    results = {}

    # CPU work
    start_time = time.time()
    cpu_result = cpu_intensive_calculation()
    results["cpu_time"] = time.time() - start_time
    results["cpu_result_sample"] = cpu_result

    # Memory work
    start_time = time.time()
    memory_data = memory_intensive_operations()
    results["memory_time"] = time.time() - start_time
    results["memory_objects_created"] = len(memory_data)

    # I/O work
    start_time = time.time()
    io_result = file_io_operations()
    results["io_time"] = time.time() - start_time
    results["io_operations"] = io_result

    # Function calls
    start_time = time.time()
    call_result = nested_function_calls()
    results["call_time"] = time.time() - start_time
    results["call_result"] = call_result

    # Data processing
    start_time = time.time()
    processing_result = data_processing_pipeline()
    results["processing_time"] = time.time() - start_time
    results["processing_stats"] = processing_result

    # Fibonacci comparison
    start_time = time.time()
    fib_recursive = fibonacci_recursive(25)  # Small enough to not take forever
    results["fib_recursive_time"] = time.time() - start_time

    start_time = time.time()
    fib_iterative = fibonacci_iterative(25)
    results["fib_iterative_time"] = time.time() - start_time

    results["fib_results_match"] = fib_recursive == fib_iterative

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 PYCROSCOPE SAMPLE WORKLOAD")
    print("=" * 60)
    print("This script demonstrates various computational patterns.")
    print("Perfect for profiling with Pycroscope!")
    print("-" * 60)

    overall_start = time.time()

    # Run the mixed workload
    final_results = mixed_workload()

    overall_time = time.time() - overall_start

    print("-" * 60)
    print("📊 WORKLOAD COMPLETED")
    print(f"⏱️  Total execution time: {overall_time:.3f} seconds")
    print(f"🔥 CPU work took: {final_results['cpu_time']:.3f}s")
    print(f"💾 Memory work took: {final_results['memory_time']:.3f}s")
    print(f"📁 I/O work took: {final_results['io_time']:.3f}s")
    print(f"🌳 Function calls took: {final_results['call_time']:.3f}s")
    print(f"⚙️  Data processing took: {final_results['processing_time']:.3f}s")
    print(f"🧮 Recursive fibonacci took: {final_results['fib_recursive_time']:.3f}s")
    print(f"🔢 Iterative fibonacci took: {final_results['fib_iterative_time']:.3f}s")
    print(f"✅ Memory objects created: {final_results['memory_objects_created']:,}")
    print(f"📄 Files processed: {final_results['io_operations']['files_processed']}")
    print("=" * 60)
