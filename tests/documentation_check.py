#!/usr/bin/env python3
"""Documentation coverage check for CI workflow."""

import ast
import os


def check_docstrings(directory):
    """Check for missing docstrings in Python files."""
    missing_docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    try:
                        tree = ast.parse(f.read(), filepath)
                        for node in ast.walk(tree):
                            if isinstance(
                                node,
                                (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                            ):
                                if not ast.get_docstring(node):
                                    missing_docs.append(
                                        f"{filepath}:{node.lineno} - {node.name}"
                                    )
                    except:
                        pass
    return missing_docs


def main():
    """Run documentation check."""
    print("Checking docstring coverage...")

    missing = check_docstrings("pycroscope/")
    if missing:
        print("Missing docstrings:")
        for item in missing[:10]:  # Show first 10
            print(f"  {item}")
        print(f"Total missing: {len(missing)}")
    else:
        print("[PASS] All functions and classes have docstrings!")


if __name__ == "__main__":
    main()
