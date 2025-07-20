#!/usr/bin/env python3
"""Integration test for CI workflow."""

import time
import pycroscope
from pycroscope import enable_profiling, ProfileConfig, CollectorType


def main():
    """Run integration test."""
    print("🔬 Running Integration Test")

    config = ProfileConfig()
    config.enable_collector(CollectorType.LINE)
    config.enable_collector(CollectorType.MEMORY)

    profiler = enable_profiling(config)

    # Simple test workload
    data = [i**2 for i in range(1000)]
    result = sum(data)

    time.sleep(0.1)
    session = profiler.end_session()
    profiler.disable()

    assert session is not None, "Session should be created"
    assert len(session.execution_events) > 0, "Events should be collected"
    print(f"✅ Integration test passed! Events: {len(session.execution_events)}")


if __name__ == "__main__":
    main()
