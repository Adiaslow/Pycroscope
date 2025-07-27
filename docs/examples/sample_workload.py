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


def inefficient_nested_search(data_list: List[int], targets: List[int]) -> List[int]:
    """
    Example of O(n²) complexity - will be detected by pattern analysis.
    This shows a nested loop anti-pattern.
    """
    found_items = []

    # Nested loops create O(n²) complexity - pattern analysis will detect this
    for target in targets:
        for item in data_list:
            if item == target:
                found_items.append(item)
                break  # At least we break early

    return found_items


def overly_complex_function(param1, param2, param3, param4, param5, param6, param7):
    """
    Function with too many parameters and high cyclomatic complexity.
    Pattern analysis will detect both issues.
    """
    result = 0

    # High cyclomatic complexity - deeply nested conditions
    if param1 > 0:
        if param2 > 0:
            if param3 > 0:
                if param4 > 0:
                    if param5 > 0:
                        if param6 > 0:
                            if param7 > 0:
                                result = (
                                    param1
                                    + param2
                                    + param3
                                    + param4
                                    + param5
                                    + param6
                                    + param7
                                )
                            else:
                                result = (
                                    param1 + param2 + param3 + param4 + param5 + param6
                                )
                        else:
                            result = param1 + param2 + param3 + param4 + param5
                    else:
                        result = param1 + param2 + param3 + param4
                else:
                    result = param1 + param2 + param3
            else:
                result = param1 + param2
        else:
            result = param1

    # More unnecessary complexity
    for i in range(10):
        for j in range(10):
            if i + j == result % 10:
                result += 1

    return result


def inefficient_data_operations(items: List[str]) -> List[str]:
    """
    Example of inefficient data structure usage.
    Using list for membership testing instead of set.
    """
    # Inefficient: using list for membership testing (O(n) per lookup)
    processed_items = []
    unique_items = []

    for item in items:
        if item not in processed_items:  # O(n) operation - should use set
            processed_items.append(item)
            unique_items.append(f"processed_{item}")

    return unique_items


def unused_helper_function():
    """
    This function is never called - dead code that pattern analysis will detect.
    """
    unused_variable = "This will be flagged as unused"
    return unused_variable


def cpu_intensive_calculation() -> float:
    """CPU-bound task - good for sampling profiler hotspot detection."""
    print("Running CPU-intensive calculations...")

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
    print("Running memory-intensive operations...")

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
    print("Running file I/O operations...")

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
    print("Running nested function calls...")

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
    print("Running data processing pipeline...")

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


def demonstrate_anti_patterns():
    """Demonstrate various anti-patterns for detection."""
    print("Running anti-pattern demonstrations...")

    # Scientific computing anti-patterns
    import numpy as np

    # Pattern: inefficient array creation
    bad_array = np.array(range(1000))  # Should use np.arange()

    # Pattern: inefficient boolean array sum
    test_array = np.random.random(1000)
    count_result = (test_array > 0.5).sum()  # Should use np.count_nonzero()

    # Pattern: suboptimal matrix operations
    matrix_a = np.random.random((100, 50))
    matrix_b = np.random.random((50, 100))
    result_tensordot = np.tensordot(matrix_a, matrix_b, axes=(1, 0))  # Could use .dot()

    # Pattern: unnecessary array copy
    original_array = np.random.random(1000)
    copied_array = original_array.copy()  # Might be unnecessary

    # Existing anti-patterns
    data = list(range(1000))

    # Nested loops (O(n²) complexity)
    result = inefficient_nested_search(data, [500, 750, 999])

    # Overly complex function
    complex_result = overly_complex_function(10, 20, 30, 40, 50, 60, 70)

    # Inefficient data operations
    string_data = [f"item_{i}" for i in range(100)]
    processed_data = inefficient_data_operations(string_data)

    # Demonstrate recursion without memoization
    fib_recursive_result = fibonacci_recursive(30)
    fib_iterative_result = fibonacci_iterative(30)

    return {
        "nested_search": result,
        "complex_function": complex_result,
        "data_operations": len(processed_data),
        "fibonacci_recursive": fib_recursive_result,
        "fibonacci_iterative": fib_iterative_result,
        "fibonacci_match": fib_recursive_result == fib_iterative_result,
        # Scientific computing results
        "bad_array_size": len(bad_array),
        "count_result": count_result,
        "tensordot_shape": result_tensordot.shape,
        "copied_array_size": len(copied_array),
    }


def mixed_workload() -> Dict[str, Any]:
    """Combined workload showing all patterns together."""
    print("Running mixed workload demonstration...")

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

    # Anti-pattern demonstrations (new)
    start_time = time.time()
    antipattern_result = demonstrate_anti_patterns()
    results["antipattern_time"] = time.time() - start_time
    results["antipattern_stats"] = antipattern_result

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
    print("Includes examples of common anti-patterns for analysis.")
    print("-" * 60)

    overall_start = time.time()

    # Run the mixed workload
    final_results = mixed_workload()

    overall_time = time.time() - overall_start

    print("-" * 60)
    print("WORKLOAD COMPLETED")
    print(f"Total execution time: {overall_time:.3f} seconds")
    print(f"CPU work took: {final_results['cpu_time']:.3f}s")
    print(f"Memory work took: {final_results['memory_time']:.3f}s")
    print(f"I/O work took: {final_results['io_time']:.3f}s")
    print(f"Function calls took: {final_results['call_time']:.3f}s")
    print(f"Data processing took: {final_results['processing_time']:.3f}s")
    print(f"Anti-pattern demos took: {final_results['antipattern_time']:.3f}s")
    print(f"Recursive fibonacci took: {final_results['fib_recursive_time']:.3f}s")
    print(f"Iterative fibonacci took: {final_results['fib_iterative_time']:.3f}s")
    print(f"Memory objects created: {final_results['memory_objects_created']:,}")
    print(f"Files processed: {final_results['io_operations']['files_processed']}")
    print("=" * 60)
