#!/usr/bin/env python3
"""
Test runner for Pycroscope test suite.

Provides convenient interface for running tests with different configurations
following our principles and best practices.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_unit_tests() -> int:
    """Run unit tests only."""
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-m", "unit", "--verbose", "tests/unit/"]
    )


def run_integration_tests() -> int:
    """Run integration tests only."""
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "integration",
            "--verbose",
            "tests/integration/",
        ]
    )


def run_core_tests() -> int:
    """Run core layer tests only."""
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-m", "core", "--verbose", "tests/unit/core/"]
    )


def run_application_tests() -> int:
    """Run application layer tests only."""
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "application",
            "--verbose",
            "tests/unit/application/",
        ]
    )


def run_infrastructure_tests() -> int:
    """Run infrastructure layer tests only."""
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "infrastructure",
            "--verbose",
            "tests/unit/infrastructure/",
        ]
    )


def run_fast_tests() -> int:
    """Run fast tests (exclude slow tests)."""
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-m", "not slow", "--verbose"]
    )


def run_all_tests() -> int:
    """Run complete test suite."""
    return subprocess.call([sys.executable, "-m", "pytest", "--verbose"])


def run_tests_with_coverage() -> int:
    """Run tests with coverage reporting."""
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src/pycroscope",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "--verbose",
        ]
    )


def main():
    """Main test runner interface."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <command>")
        print("Commands:")
        print("  unit         - Run unit tests only")
        print("  integration  - Run integration tests only")
        print("  core         - Run core layer tests only")
        print("  application  - Run application layer tests only")
        print("  infrastructure - Run infrastructure layer tests only")
        print("  fast         - Run fast tests (exclude slow)")
        print("  all          - Run complete test suite")
        print("  coverage     - Run tests with coverage")
        return 1

    command = sys.argv[1].lower()

    # Change to test directory
    test_dir = Path(__file__).parent
    original_cwd = Path.cwd()

    try:
        # Ensure we're in the project root for pytest
        project_root = test_dir.parent
        sys.path.insert(0, str(project_root))

        if command == "unit":
            return run_unit_tests()
        elif command == "integration":
            return run_integration_tests()
        elif command == "core":
            return run_core_tests()
        elif command == "application":
            return run_application_tests()
        elif command == "infrastructure":
            return run_infrastructure_tests()
        elif command == "fast":
            return run_fast_tests()
        elif command == "all":
            return run_all_tests()
        elif command == "coverage":
            return run_tests_with_coverage()
        else:
            print(f"Unknown command: {command}")
            return 1

    finally:
        # Restore original directory
        pass


if __name__ == "__main__":
    sys.exit(main())
