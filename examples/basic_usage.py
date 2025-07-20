#!/usr/bin/env python3
"""
Basic usage example for Pycroscope profiling.

This example demonstrates how to use Pycroscope to profile a simple
Python program and collect performance data.
"""

import time
import random
from pycroscope import enable_profiling, ProfileConfig, CollectorType


def fibonacci(n):
    """Calculate fibonacci number (inefficient recursive implementation)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def matrix_multiply(size: int = 100):
    """Perform matrix multiplication (memory intensive)."""
    # Create two random matrices
    matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]

    # Multiply matrices
    result = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def io_operations():
    """Perform some I/O operations."""
    import tempfile
    import os

    # Write some data to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        temp_filename = f.name
        for i in range(1000):
            f.write(f"Line {i}: This is some test data for I/O profiling\n")

    # Read the data back
    with open(temp_filename, "r") as f:
        lines = f.readlines()

    # Clean up
    os.unlink(temp_filename)

    return len(lines)


def demo_basic_usage():
    """Demonstrate basic Pycroscope usage with default configuration."""
    print("=== Basic Usage Demo ===")

    # Enable profiling with default configuration
    profiler = enable_profiling()

    try:
        # Run some code to profile
        print("Running fibonacci calculation...")
        result1 = fibonacci(10)
        print(f"fibonacci(10) = {result1}")

        print("Running matrix multiplication...")
        result2 = matrix_multiply(50)
        print(f"Matrix result size: {len(result2)}x{len(result2[0])}")

        print("Running I/O operations...")
        result3 = io_operations()
        print(f"I/O operations completed, processed {result3} lines")

    finally:
        # Disable profiling and get results
        profiler.disable()

        # Get current session data
        session = profiler.current_session
        if session:
            print(f"\nProfiling completed for session: {session.session_id}")
            print(f"Total events collected: {session.total_events}")
            print(f"Peak memory usage: {session.peak_memory / 1024 / 1024:.2f} MB")


def demo_custom_configuration():
    """Demonstrate Pycroscope with custom configuration."""
    print("\n=== Custom Configuration Demo ===")

    # Create custom configuration
    config = ProfileConfig()

    # Enable only specific collectors
    config.disable_collector(CollectorType.IO)  # Disable I/O profiling
    config.disable_collector(CollectorType.CPU)  # Disable CPU profiling

    # Configure sampling
    # TODO: Fix sampling rate configuration
    # config.collectors[CollectorType.LINE].sampling_rate = 0.5  # Sample 50% of line events
    config.collectors[CollectorType.MEMORY].buffer_size = 5000  # Smaller buffer

    # Enable verbose output
    config.verbose = True

    # Create profiler with custom config
    profiler = enable_profiling(config)

    try:
        print("Running with custom configuration...")

        # Run some intensive operations
        for i in range(3):
            print(f"Iteration {i + 1}")
            fibonacci(8)
            time.sleep(0.1)  # Small delay to see timing

    finally:
        profiler.disable()
        print("Custom configuration demo completed")


def demo_context_manager():
    """Demonstrate using Pycroscope as a context manager."""
    print("\n=== Context Manager Demo ===")

    # Use Pycroscope as a context manager
    config = ProfileConfig()
    config.target_package = "example_demo"

    with enable_profiling(config) as profiler:
        print("Inside profiling context...")

        # Run code to profile
        start_time = time.time()
        fibonacci(12)
        matrix_multiply(30)
        end_time = time.time()

        print(f"Operations completed in {end_time - start_time:.2f} seconds")

        # Session is automatically ended when exiting context

    print("Context manager demo completed")


def demo_session_management():
    """Demonstrate manual session management."""
    print("\n=== Session Management Demo ===")

    config = ProfileConfig()
    profiler = enable_profiling(config)

    try:
        # Begin a named session
        session_id = profiler.begin_session("manual_session_demo")
        print(f"Started session: {session_id}")

        # Run some operations
        fibonacci(9)

        # End the session and get data
        session_data = profiler.end_session()

        if session_data:
            print(f"Session {session_data.session_id} completed")
            print(f"Target package: {session_data.target_package}")
            if (
                session_data.execution_context
                and session_data.execution_context.duration
            ):
                print(
                    f"Session duration: {session_data.execution_context.duration:.3f}s"
                )

            # Begin another session
            session_id_2 = profiler.begin_session("second_session")
            print(f"Started second session: {session_id_2}")

            matrix_multiply(25)

            session_data_2 = profiler.end_session()
            if session_data_2:
                print(f"Second session {session_data_2.session_id} completed")

    finally:
        profiler.disable()

    print("Session management demo completed")


if __name__ == "__main__":
    print("Pycroscope Basic Usage Examples")
    print("=" * 40)

    # Run all demos
    demo_basic_usage()
    demo_custom_configuration()
    demo_context_manager()
    demo_session_management()

    print("\n" + "=" * 40)
    print("All demos completed successfully!")
    print("\nTry running this script and observe the profiling output.")
    print("In a real application, you would analyze the collected data")
    print("to identify performance bottlenecks and optimization opportunities.")
