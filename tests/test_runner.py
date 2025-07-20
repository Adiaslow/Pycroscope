#!/usr/bin/env python3
"""
Test runner for Pycroscope testing framework.

Provides convenient test execution with coverage reporting and filtering.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(
    test_path: str = "tests/",
    coverage: bool = True,
    markers: str = None,
    verbose: bool = False,
    html_report: bool = False,
    parallel: bool = False,
) -> int:
    """
    Run tests with pytest.

    Args:
        test_path: Path to tests directory or specific test file
        coverage: Whether to generate coverage report
        markers: Pytest markers to filter tests (e.g., "unit", "integration")
        verbose: Verbose output
        html_report: Generate HTML coverage report
        parallel: Run tests in parallel

    Returns:
        Exit code from pytest
    """
    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test path
    cmd.append(test_path)

    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=pycroscope", "--cov-report=term-missing"])
        if html_report:
            cmd.append("--cov-report=html")

    # Add markers filter
    if markers:
        cmd.extend(["-m", markers])

    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])

    # Add other useful options
    cmd.extend(
        [
            "--tb=short",  # Short traceback format
            "--strict-markers",  # Strict marker validation
            "--strict-config",  # Strict config validation
        ]
    )

    print(f"Running command: {' '.join(cmd)}")

    # Execute tests
    result = subprocess.run(cmd)
    return result.returncode


def run_specific_tests():
    """Run specific test categories."""
    test_categories = {
        "core": "tests/core/",
        "storage": "tests/storage/",
        "cli": "tests/cli/",
        "collectors": "tests/collectors/",
        "analysis": "tests/analysis/",
        "integration": "tests/integration/",
    }

    print("Available test categories:")
    for category, path in test_categories.items():
        if Path(path).exists():
            print(f"  {category}: {path}")
        else:
            print(f"  {category}: {path} (not yet implemented)")

    return 0


def run_performance_tests():
    """Run performance and stress tests."""
    return run_tests(
        test_path="tests/",
        markers="slow or performance",
        verbose=True,
        parallel=False,  # Performance tests should run sequentially
    )


def run_quick_tests():
    """Run quick unit tests only."""
    return run_tests(
        test_path="tests/",
        markers="unit and not slow",
        coverage=False,
        verbose=False,
        parallel=True,
    )


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Pycroscope test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with coverage
  python tests/test_runner.py
  
  # Run only unit tests
  python tests/test_runner.py --markers unit
  
  # Run core component tests
  python tests/test_runner.py --path tests/core/
  
  # Run tests with HTML coverage report
  python tests/test_runner.py --html-report
  
  # Run quick tests (no coverage)
  python tests/test_runner.py --quick
  
  # Run performance tests
  python tests/test_runner.py --performance
  
  # List test categories
  python tests/test_runner.py --list-categories
        """,
    )

    parser.add_argument(
        "--path", default="tests/", help="Path to tests directory or specific test file"
    )

    parser.add_argument(
        "--markers", help="Pytest markers to filter tests (e.g., 'unit', 'integration')"
    )

    parser.add_argument(
        "--no-coverage", action="store_true", help="Disable coverage reporting"
    )

    parser.add_argument(
        "--html-report", action="store_true", help="Generate HTML coverage report"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--parallel", "-j", action="store_true", help="Run tests in parallel"
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick unit tests only"
    )

    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )

    parser.add_argument(
        "--list-categories", action="store_true", help="List available test categories"
    )

    args = parser.parse_args()

    # Handle special modes
    if args.list_categories:
        return run_specific_tests()

    if args.quick:
        return run_quick_tests()

    if args.performance:
        return run_performance_tests()

    # Run normal tests
    return run_tests(
        test_path=args.path,
        coverage=not args.no_coverage,
        markers=args.markers,
        verbose=args.verbose,
        html_report=args.html_report,
        parallel=args.parallel,
    )


if __name__ == "__main__":
    sys.exit(main())
