#!/usr/bin/env python3
"""Installation verification for CI workflow."""

import pycroscope
from pycroscope import enable_profiling


def main():
    """Verify installation is working correctly."""
    print(f"Pycroscope version: {pycroscope.__version__}")

    # Test basic import
    print("Import successful")

    # Test basic functionality
    try:
        profiler = enable_profiling()
        profiler.disable()
        print("[PASS] Basic functionality verified")
    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        raise


if __name__ == "__main__":
    main()
