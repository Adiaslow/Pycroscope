#!/usr/bin/env python3
"""
Workflow validation script for GitHub Actions.

This script validates that the workflow configuration is correct and that
all necessary components are in place for successful CI/CD execution.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def validate_workflow_file():
    """Validate the GitHub Actions workflow YAML file."""
    workflow_path = Path(".github/workflows/tests.yml")

    if not workflow_path.exists():
        print("❌ Workflow file not found: .github/workflows/tests.yml")
        return False

    try:
        with open(workflow_path, "r") as f:
            workflow = yaml.safe_load(f)

        # Check required jobs
        required_jobs = [
            "test",
            "architecture-validation",
            "security",
            "code-quality",
            "release-check",
        ]

        for job in required_jobs:
            if job not in workflow.get("jobs", {}):
                print(f"❌ Missing required job: {job}")
                return False

        print("✅ Workflow YAML file is valid")
        return True

    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error in workflow file: {e}")
        return False


def validate_test_runner():
    """Validate that the test runner exists and is structured correctly."""
    try:
        # Check if test runner exists
        runner_path = Path("tests/run_tests.py")
        if not runner_path.exists():
            print("❌ Test runner not found: tests/run_tests.py")
            return False

        # Check that it's a valid Python file by trying to parse it
        with open(runner_path, "r") as f:
            content = f.read()

        # Look for key indicators that it's the right test runner
        if "def main(" not in content:
            print("❌ Test runner missing main function")
            return False

        if "pytest" not in content:
            print("❌ Test runner doesn't appear to use pytest")
            return False

        print("✅ Test runner structure is correct")
        return True

    except Exception as e:
        print(f"❌ Error validating test runner: {e}")
        return False


def validate_dependencies():
    """Validate that required dependencies configuration is correct."""
    try:
        # Check pyproject.toml exists
        if not Path("pyproject.toml").exists():
            print("❌ pyproject.toml not found")
            return False

        # Check that source files exist
        src_files = [
            "src/pycroscope/__init__.py",
            "src/pycroscope/core/config.py",
            "src/pycroscope/application/services.py",
        ]

        for src_file in src_files:
            if not Path(src_file).exists():
                print(f"❌ Missing source file: {src_file}")
                return False

        print("✅ Core source files are present")
        return True

    except Exception as e:
        print(f"❌ Error validating dependencies: {e}")
        return False


def validate_coverage_config():
    """Validate coverage configuration."""
    try:
        # Check codecov.yml
        codecov_path = Path("codecov.yml")
        if not codecov_path.exists():
            print("❌ codecov.yml not found")
            return False

        with open(codecov_path, "r") as f:
            codecov_config = yaml.safe_load(f)

        # Check coverage target
        target = (
            codecov_config.get("coverage", {})
            .get("status", {})
            .get("project", {})
            .get("default", {})
            .get("target")
        )
        if target != "50%":
            print(f"❌ Coverage target should be 50%, found: {target}")
            return False

        print("✅ Coverage configuration is correct")
        return True

    except Exception as e:
        print(f"❌ Error validating coverage config: {e}")
        return False


def validate_project_structure():
    """Validate that the project structure is correct."""
    required_paths = [
        "src/pycroscope",
        "tests/unit/core",
        "tests/unit/application",
        "tests/unit/infrastructure",
        "tests/integration",
        ".github/workflows",
    ]

    for path in required_paths:
        if not Path(path).exists():
            print(f"❌ Missing required path: {path}")
            return False

    print("✅ Project structure is correct")
    return True


def main():
    """Run all validations."""
    print("🔍 Validating GitHub Actions workflow configuration...\n")

    validations = [
        ("Project Structure", validate_project_structure),
        ("Workflow YAML", validate_workflow_file),
        ("Test Runner", validate_test_runner),
        ("Dependencies", validate_dependencies),
        ("Coverage Config", validate_coverage_config),
    ]

    all_passed = True

    for name, validation_func in validations:
        print(f"📋 Validating {name}...")
        if not validation_func():
            all_passed = False
        print()

    if all_passed:
        print("🎉 All validations passed! Workflow is ready for GitHub Actions.")
        return 0
    else:
        print("❌ Some validations failed. Please fix the issues before pushing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
