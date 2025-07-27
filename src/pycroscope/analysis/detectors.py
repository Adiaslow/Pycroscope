"""
Pattern detectors leveraging validated external packages.

Implements specific detectors for different types of performance anti-patterns
and code quality issues using established tools like radon, vulture, and ast.
"""

import ast
import sys
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from .interfaces import (
    PatternType,
    AnalysisResult,
    PatternAnalyzer,
    ComplexityAnalyzer,
    AntiPatternDetector,
)
from .config import AnalysisConfig


class AlgorithmicComplexityDetector:
    """Detects algorithmic complexity issues using radon and AST analysis."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzer_type = "algorithmic_complexity"

    @property
    def supported_patterns(self) -> List[PatternType]:
        return [
            PatternType.QUADRATIC_COMPLEXITY,
            PatternType.EXPONENTIAL_COMPLEXITY,
            PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
            PatternType.RECURSIVE_WITHOUT_MEMOIZATION,
        ]

    def analyze(self, code: str, file_path: Path, **kwargs) -> List[AnalysisResult]:
        """Analyze code for algorithmic complexity issues."""
        results = []

        # Use radon for cyclomatic complexity
        results.extend(self._analyze_cyclomatic_complexity(code, file_path))

        # Custom AST analysis for algorithmic patterns
        results.extend(self._analyze_nested_loops(code, file_path))
        results.extend(self._analyze_recursive_patterns(code, file_path))

        return results

    def _analyze_cyclomatic_complexity(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Use radon to analyze cyclomatic complexity."""
        results = []

        try:
            # Save code to temporary file for radon analysis
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # Run radon complexity analysis
            cmd = [sys.executable, "-m", "radon", "cc", temp_file_path, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                import json

                radon_data = json.loads(result.stdout)

                for file_data in radon_data.values():
                    for func_data in file_data:
                        complexity = func_data.get("complexity", 0)
                        if complexity > self.config.complexity_threshold:
                            severity = self._get_complexity_severity(complexity)

                            results.append(
                                AnalysisResult(
                                    pattern_type=PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
                                    severity=severity,
                                    confidence=0.9,
                                    description=f"Function '{func_data['name']}' has high cyclomatic complexity: {complexity}",
                                    location={
                                        "file": str(file_path),
                                        "line": func_data.get("lineno", 0),
                                        "function": func_data["name"],
                                    },
                                    suggestion=f"Consider breaking down this function. Target complexity: <= {self.config.complexity_threshold}",
                                    metrics={"cyclomatic_complexity": complexity},
                                )
                            )

            # Clean up temporary file
            Path(temp_file_path).unlink()

        except Exception as e:
            print(f"Warning: Radon analysis failed: {e}")

        return results

    def _analyze_nested_loops(self, code: str, file_path: Path) -> List[AnalysisResult]:
        """Analyze for nested loops that might indicate O(n²) or worse complexity."""
        results = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    loop_depth = self._calculate_max_loop_depth(node)
                    if (
                        loop_depth >= 3
                    ):  # 3+ nested loops often indicate complexity issues
                        severity = "critical" if loop_depth >= 4 else "high"

                        results.append(
                            AnalysisResult(
                                pattern_type=PatternType.NESTED_LOOPS,
                                severity=severity,
                                confidence=0.8,
                                description=f"Function '{node.name}' has {loop_depth} levels of nested loops",
                                location={
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "function": node.name,
                                },
                                suggestion="Consider extracting inner loops into separate functions or using more efficient algorithms",
                                metrics={"loop_depth": loop_depth},
                            )
                        )

        except SyntaxError as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return results

    def _analyze_recursive_patterns(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Analyze for recursive functions that might benefit from memoization."""
        results = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if self._is_recursive_function(node):
                        if not self._has_memoization(node):
                            results.append(
                                AnalysisResult(
                                    pattern_type=PatternType.RECURSIVE_WITHOUT_MEMOIZATION,
                                    severity="medium",
                                    confidence=0.7,
                                    description=f"Recursive function '{node.name}' without memoization",
                                    location={
                                        "file": str(file_path),
                                        "line": node.lineno,
                                        "function": node.name,
                                    },
                                    suggestion="Consider adding memoization with @lru_cache or manual caching",
                                    metrics={
                                        "is_recursive": True,
                                        "has_memoization": False,
                                    },
                                )
                            )

        except SyntaxError as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return results

    def _calculate_max_loop_depth(self, node: ast.AST) -> int:
        """Calculate maximum loop nesting depth."""
        max_depth = 0

        def visit_node(n, current_depth=0):
            nonlocal max_depth
            if isinstance(n, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)

            for child in ast.iter_child_nodes(n):
                visit_node(child, current_depth)

        visit_node(node)
        return max_depth

    def _is_recursive_function(self, func_node: ast.FunctionDef) -> bool:
        """Check if function calls itself recursively."""
        func_name = func_node.name

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == func_name:
                    return True

        return False

    def _has_memoization(self, func_node: ast.FunctionDef) -> bool:
        """Check if function uses memoization decorators."""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ["lru_cache", "cache"]:
                    return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in ["lru_cache", "cache"]:
                    return True
        return False

    def _get_complexity_severity(self, complexity: int) -> str:
        """Get severity level based on complexity score."""
        if complexity >= 20:
            return "critical"
        elif complexity >= 15:
            return "high"
        elif complexity >= 10:
            return "medium"
        else:
            return "low"


class DeadCodeDetector:
    """Detects dead code using vulture and custom analysis."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzer_type = "dead_code"

    @property
    def supported_patterns(self) -> List[PatternType]:
        return [
            PatternType.DEAD_CODE,
            PatternType.UNUSED_IMPORTS,
        ]

    def analyze(self, code: str, file_path: Path, **kwargs) -> List[AnalysisResult]:
        """Analyze code for dead code and unused imports."""
        results = []

        # Skip test files if configured
        if self.config.exclude_test_files and self._is_test_file(file_path):
            return results

        # Use vulture for dead code detection
        results.extend(self._analyze_with_vulture(code, file_path))

        # Custom analysis for unused imports
        results.extend(self._analyze_unused_imports(code, file_path))

        return results

    def _analyze_with_vulture(self, code: str, file_path: Path) -> List[AnalysisResult]:
        """Use vulture to detect dead code."""
        results = []

        try:
            # Save code to temporary file for vulture analysis
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # Run vulture
            cmd = [sys.executable, "-m", "vulture", temp_file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Parse vulture output
            for line in result.stdout.split("\n"):
                if line.strip() and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 3:
                        line_no = int(parts[1]) if parts[1].isdigit() else 0
                        message = ":".join(parts[2:]).strip()

                        results.append(
                            AnalysisResult(
                                pattern_type=PatternType.DEAD_CODE,
                                severity="low",
                                confidence=0.8,
                                description=f"Unused code detected: {message}",
                                location={
                                    "file": str(file_path),
                                    "line": line_no,
                                    "function": None,
                                },
                                suggestion="Consider removing this unused code to improve maintainability",
                                metrics={"vulture_detected": True},
                            )
                        )

            # Clean up temporary file
            Path(temp_file_path).unlink()

        except Exception as e:
            print(f"Warning: Vulture analysis failed: {e}")

        return results

    def _analyze_unused_imports(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Analyze for unused imports using AST."""
        results = []

        try:
            tree = ast.parse(code)
            imports = []
            used_names = set()

            # Collect all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name
                        imports.append((import_name, node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name
                        imports.append(
                            (import_name, node.lineno, f"{node.module}.{alias.name}")
                        )

            # Collect all used names
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Handle cases like module.function
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Find unused imports
            for import_name, line_no, full_name in imports:
                if import_name not in used_names:
                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.UNUSED_IMPORTS,
                            severity="low",
                            confidence=0.9,
                            description=f"Unused import: {full_name}",
                            location={
                                "file": str(file_path),
                                "line": line_no,
                                "function": None,
                            },
                            suggestion=f"Remove unused import: {full_name}",
                            metrics={
                                "import_name": import_name,
                                "full_name": full_name,
                            },
                        )
                    )

        except SyntaxError as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return results

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file based on patterns."""
        file_name = file_path.name
        return any(
            file_path.match(pattern) for pattern in self.config.test_file_patterns
        )


class MaintainabilityAnalyzer:
    """Analyzes code maintainability using radon and custom metrics."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzer_type = "maintainability"

    @property
    def supported_patterns(self) -> List[PatternType]:
        return [
            PatternType.LOW_MAINTAINABILITY_INDEX,
            PatternType.LONG_FUNCTION,
            PatternType.TOO_MANY_PARAMETERS,
        ]

    def analyze(self, code: str, file_path: Path, **kwargs) -> List[AnalysisResult]:
        """Analyze code maintainability."""
        results = []

        # Use radon for maintainability index
        results.extend(self._analyze_maintainability_index(code, file_path))

        # Custom analysis for function characteristics
        results.extend(self._analyze_function_characteristics(code, file_path))

        return results

    def _analyze_maintainability_index(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Use radon to analyze maintainability index."""
        results = []

        try:
            # Save code to temporary file for radon analysis
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # Run radon maintainability analysis
            cmd = [sys.executable, "-m", "radon", "mi", temp_file_path, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                import json

                radon_data = json.loads(result.stdout)

                for file_data in radon_data.values():
                    mi_score = file_data.get("mi", 100)
                    if mi_score < self.config.maintainability_threshold:
                        severity = self._get_maintainability_severity(mi_score)

                        results.append(
                            AnalysisResult(
                                pattern_type=PatternType.LOW_MAINTAINABILITY_INDEX,
                                severity=severity,
                                confidence=0.8,
                                description=f"Low maintainability index: {mi_score:.1f}",
                                location={
                                    "file": str(file_path),
                                    "line": 1,
                                    "function": None,
                                },
                                suggestion=f"Improve code structure. Target MI: >= {self.config.maintainability_threshold}",
                                metrics={"maintainability_index": mi_score},
                            )
                        )

            # Clean up temporary file
            Path(temp_file_path).unlink()

        except Exception as e:
            print(f"Warning: Radon MI analysis failed: {e}")

        return results

    def _analyze_function_characteristics(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Analyze function length and parameter count."""
        results = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Calculate function length
                    func_lines = self._get_function_lines(node, code)
                    if func_lines > self.config.max_function_lines:
                        results.append(
                            AnalysisResult(
                                pattern_type=PatternType.LONG_FUNCTION,
                                severity="medium",
                                confidence=0.9,
                                description=f"Function '{node.name}' is too long: {func_lines} lines",
                                location={
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "function": node.name,
                                },
                                suggestion=f"Consider breaking down this function. Target: <= {self.config.max_function_lines} lines",
                                metrics={"function_lines": func_lines},
                            )
                        )

                    # Check parameter count
                    param_count = len(node.args.args)
                    if param_count > self.config.max_function_parameters:
                        results.append(
                            AnalysisResult(
                                pattern_type=PatternType.TOO_MANY_PARAMETERS,
                                severity="medium",
                                confidence=0.9,
                                description=f"Function '{node.name}' has too many parameters: {param_count}",
                                location={
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "function": node.name,
                                },
                                suggestion=f"Consider using data classes or reducing parameters. Target: <= {self.config.max_function_parameters}",
                                metrics={"parameter_count": param_count},
                            )
                        )

        except SyntaxError as e:
            print(f"Warning: Could not parse {file_path}: {e}")

        return results

    def _get_function_lines(self, func_node: ast.FunctionDef, code: str) -> int:
        """Calculate the number of lines in a function."""
        lines = code.split("\n")
        start_line = func_node.lineno - 1  # Convert to 0-based indexing

        # Find the end of the function by looking for the next function or class at the same indentation
        end_line = len(lines)
        func_indentation = len(lines[start_line]) - len(lines[start_line].lstrip())

        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indentation = len(line) - len(line.lstrip())
                if current_indentation <= func_indentation and line.strip().startswith(
                    ("def ", "class ", "@")
                ):
                    end_line = i
                    break

        return end_line - start_line

    def _get_maintainability_severity(self, mi_score: float) -> str:
        """Get severity level based on maintainability index."""
        if mi_score < 10:
            return "critical"
        elif mi_score < 15:
            return "high"
        elif mi_score < 20:
            return "medium"
        else:
            return "low"


class AntiPatternDetector:
    """Main anti-pattern detector that orchestrates all specific detectors."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.complexity_detector = AlgorithmicComplexityDetector(config)
        self.dead_code_detector = DeadCodeDetector(config)
        self.maintainability_analyzer = MaintainabilityAnalyzer(config)

    def detect_performance_antipatterns(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect performance anti-patterns in code."""
        results = []

        # Run complexity analysis
        if any(
            p in self.config.enabled_patterns
            for p in self.complexity_detector.supported_patterns
        ):
            results.extend(self.complexity_detector.analyze(code, file_path))

        return results

    def detect_maintainability_issues(
        self, code: str, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect maintainability issues in code."""
        results = []

        # Run dead code detection
        if any(
            p in self.config.enabled_patterns
            for p in self.dead_code_detector.supported_patterns
        ):
            results.extend(self.dead_code_detector.analyze(code, file_path))

        # Run maintainability analysis
        if any(
            p in self.config.enabled_patterns
            for p in self.maintainability_analyzer.supported_patterns
        ):
            results.extend(self.maintainability_analyzer.analyze(code, file_path))

        return results

    def correlate_with_hotspots(
        self, patterns: List[AnalysisResult], profiling_data: Dict[str, Any]
    ) -> List[AnalysisResult]:
        """Correlate detected patterns with profiling hotspots."""
        if not self.config.correlate_with_profiling or not profiling_data:
            return patterns

        # Extract hotspot information from profiling data
        hotspots = self._extract_hotspots(profiling_data)

        # Update patterns with correlation information
        for pattern in patterns:
            correlation = self._find_correlation(pattern, hotspots)
            if correlation:
                pattern.profiling_correlation = correlation
                if self.config.prioritize_hotspots:
                    pattern.severity = self._boost_severity(pattern.severity)

        return patterns

    def _extract_hotspots(self, profiling_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract performance hotspots from profiling data."""
        hotspots = []

        # Extract from different profiler types
        if "call" in profiling_data:
            call_data = profiling_data["call"]
            if "data" in call_data and "stats" in call_data["data"]:
                stats = call_data["data"]["stats"]
                for func_key, func_stats in stats.items():
                    if isinstance(func_stats, dict) and "total_time" in func_stats:
                        if (
                            func_stats["total_time"]
                            > self.config.hotspot_correlation_threshold
                        ):
                            hotspots.append(
                                {
                                    "type": "call",
                                    "function": func_key,
                                    "total_time": func_stats["total_time"],
                                    "call_count": func_stats.get("count", 0),
                                }
                            )

        if "line" in profiling_data:
            line_data = profiling_data["line"]
            if "data" in line_data and "functions" in line_data["data"]:
                functions = line_data["data"]["functions"]
                for func_info in functions:
                    if (
                        "total_time" in func_info
                        and func_info["total_time"]
                        > self.config.hotspot_correlation_threshold
                    ):
                        hotspots.append(
                            {
                                "type": "line",
                                "function": func_info.get("name", ""),
                                "file": func_info.get("filename", ""),
                                "total_time": func_info["total_time"],
                            }
                        )

        return hotspots

    def _find_correlation(
        self, pattern: AnalysisResult, hotspots: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find correlation between a pattern and performance hotspots."""
        pattern_file = pattern.location.get("file", "")
        pattern_function = pattern.location.get("function", "")

        for hotspot in hotspots:
            # Check file and function matching
            hotspot_file = hotspot.get("file", "")
            hotspot_function = hotspot.get("function", "")

            if pattern_file and hotspot_file and pattern_file in hotspot_file:
                if (
                    pattern_function
                    and hotspot_function
                    and pattern_function in hotspot_function
                ):
                    return {
                        "hotspot_type": hotspot["type"],
                        "performance_impact": hotspot.get("total_time", 0),
                        "correlation_strength": "high",
                    }
                elif not pattern_function:  # File-level pattern
                    return {
                        "hotspot_type": hotspot["type"],
                        "performance_impact": hotspot.get("total_time", 0),
                        "correlation_strength": "medium",
                    }

        return None

    def _boost_severity(self, current_severity: str) -> str:
        """Boost severity for patterns found in hotspots."""
        severity_hierarchy = ["low", "medium", "high", "critical"]
        current_index = (
            severity_hierarchy.index(current_severity)
            if current_severity in severity_hierarchy
            else 0
        )
        boosted_index = min(current_index + 1, len(severity_hierarchy) - 1)
        return severity_hierarchy[boosted_index]


class ScientificComputingDetector:
    """Detects scientific computing anti-patterns using validated approaches from external tools.

    Based on patterns from:
    - perflint: Python performance linter (25+ patterns)
    - ALTERapi: NumPy/Pandas/SciPy API optimization recommendations
    - Scalene: AI-powered optimization proposals
    - Academic research: Static analysis for numerical computing
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.analyzer_type = "scientific_computing"

    @property
    def supported_patterns(self) -> List[PatternType]:
        """Return only the patterns that are currently implemented with detection methods."""
        return [
            # Currently implemented scientific computing patterns
            PatternType.MISSED_VECTORIZATION,
            PatternType.INEFFICIENT_ARRAY_OPERATIONS,
            PatternType.SUBOPTIMAL_MATRIX_OPERATIONS,
            PatternType.UNNECESSARY_ARRAY_COPY,
            PatternType.INEFFICIENT_BROADCASTING,
            PatternType.SCALAR_ARRAY_OPERATIONS,
            PatternType.WRONG_DTYPE_USAGE,
            PatternType.INEFFICIENT_ARRAY_CONCATENATION,
            PatternType.SUBOPTIMAL_LINEAR_ALGEBRA,
            # Additional patterns available for implementation:
            # PatternType.NON_CONTIGUOUS_MEMORY_ACCESS,
            # PatternType.INEFFICIENT_MEMORY_LAYOUT,
            # PatternType.SUBOPTIMAL_ALGORITHM_CHOICE,
            # PatternType.UNNECESSARY_LIST_CAST,
            # PatternType.INCORRECT_DICTIONARY_ITERATOR,
            # PatternType.LOOP_INVARIANT_STATEMENT,
            # PatternType.GLOBAL_NAME_IN_LOOP,
            # PatternType.TRY_EXCEPT_IN_LOOP,
            # PatternType.INEFFICIENT_BYTES_SLICING,
            # PatternType.DOTTED_IMPORT_IN_LOOP,
            # PatternType.USE_TUPLE_OVER_LIST,
            # PatternType.USE_LIST_COMPREHENSION,
            # PatternType.USE_LIST_COPY,
            # PatternType.USE_DICT_COMPREHENSION,
            # PatternType.PYTHON_LIST_VS_NUMPY_ARRAY,
            # PatternType.INEFFICIENT_STRING_OPERATIONS,
            # PatternType.CACHE_INEFFICIENT_ACCESS,
            # PatternType.WRONG_AXIS_SPECIFICATION,
            # PatternType.INEFFICIENT_INDEXING_PATTERNS,
            # PatternType.SUBOPTIMAL_AGGREGATION,
        ]

    def analyze(self, code: str, file_path: Path, **kwargs) -> List[AnalysisResult]:
        """Analyze code for scientific computing anti-patterns.

        Currently implemented detectors:
        - Missed vectorization
        - Inefficient array operations
        - Suboptimal matrix operations
        - Unnecessary array copies
        - Inefficient broadcasting
        - Scalar array operations
        - Wrong dtype usage
        - Inefficient concatenation
        - Suboptimal linear algebra

        Additional patterns are defined but require implementation:
        - General performance patterns from perflint (12 patterns)
        - Data structure & memory patterns (6 patterns)
        """
        results = []

        tree = ast.parse(code)

        # Detect implemented scientific computing anti-patterns
        results.extend(self._detect_missed_vectorization(tree, file_path))
        results.extend(self._detect_inefficient_array_ops(tree, file_path))
        results.extend(self._detect_suboptimal_matrix_ops(tree, file_path))
        results.extend(self._detect_unnecessary_array_copies(tree, file_path))
        results.extend(self._detect_inefficient_broadcasting(tree, file_path))
        results.extend(self._detect_scalar_array_operations(tree, file_path))
        results.extend(self._detect_wrong_dtype_usage(tree, file_path))
        results.extend(self._detect_inefficient_concatenation(tree, file_path))
        results.extend(self._detect_suboptimal_linear_algebra(tree, file_path))

        # TODO: Implement additional pattern detectors based on external tools:
        # - perflint patterns (unnecessary list cast, loop invariants, etc.)
        # - Data structure optimization patterns
        # - Memory layout patterns

        return results

    def analyze_with_profiling_data(
        self, code: str, file_path: Path, profiling_data: Dict[str, Any], **kwargs
    ) -> List[AnalysisResult]:
        """Analyze code with correlation to profiling data for enhanced pattern detection."""
        # Run the standard analysis first
        results = self.analyze(code, file_path, **kwargs)

        # TODO: Enhance detection with profiling correlation
        # - Prioritize patterns found in performance hotspots
        # - Correlate array operations with memory allocation data
        # - Link vectorization opportunities with execution time data

        return results

    def _detect_missed_vectorization(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect loops that could be vectorized using NumPy operations."""
        results = []

        class VectorizationVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                # Check for explicit loops over arrays that could be vectorized
                if self._is_array_element_loop(node):
                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.MISSED_VECTORIZATION,
                            severity="medium",
                            confidence=0.8,
                            description="Loop over array elements could be vectorized with NumPy operations",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Replace explicit loop with vectorized NumPy operations",
                            metrics={"loop_type": "array_element_access"},
                        )
                    )
                self.generic_visit(node)

            def _is_array_element_loop(self, node: ast.For) -> bool:
                """Check if this is a loop over array elements with element-wise operations."""
                # Look for patterns like: for i in range(len(array)): result[i] = array[i] + value
                if (
                    isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ):

                    # Check if loop body contains element-wise operations
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign) and isinstance(
                            stmt.targets[0], ast.Subscript
                        ):
                            return True
                return False

        VectorizationVisitor().visit(tree)
        return results

    def _detect_inefficient_array_ops(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect inefficient array operations based on ALTERapi patterns."""
        results = []

        class ArrayOpsVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Detect np.array(range(n)) -> should use np.arange(n)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"
                    and node.func.attr == "array"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Call)
                    and isinstance(node.args[0].func, ast.Name)
                    and node.args[0].func.id == "range"
                ):

                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.INEFFICIENT_ARRAY_OPERATIONS,
                            severity="medium",
                            confidence=0.9,
                            description="np.array(range(n)) is inefficient, use np.arange(n) instead",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Replace with np.arange() for better performance",
                            metrics={"optimization_type": "array_creation"},
                        )
                    )

                # Detect (arr > value).sum() -> should use np.count_nonzero()
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sum"
                    and isinstance(node.func.value, ast.Compare)
                ):

                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.INEFFICIENT_ARRAY_OPERATIONS,
                            severity="low",
                            confidence=0.7,
                            description="Boolean array sum could use np.count_nonzero() for better performance",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Use np.count_nonzero() instead of .sum() on boolean arrays",
                            metrics={"optimization_type": "counting_operation"},
                        )
                    )

                self.generic_visit(node)

        ArrayOpsVisitor().visit(tree)
        return results

    def _detect_suboptimal_matrix_ops(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect suboptimal matrix operations."""
        results = []

        class MatrixOpsVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Detect np.tensordot patterns that could use simpler operations
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "np"
                    and node.func.attr == "tensordot"
                ):

                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.SUBOPTIMAL_MATRIX_OPERATIONS,
                            severity="low",
                            confidence=0.6,
                            description="np.tensordot might be replaceable with .dot() for better performance",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Consider using .dot() or @ operator for simple matrix multiplication",
                            metrics={"operation_type": "matrix_multiplication"},
                        )
                    )

                # Detect manual matrix multiplication loops
                elif (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "sum"
                    and len(node.args) == 1
                ):
                    # This is a simplified check - could be expanded
                    pass

                self.generic_visit(node)

        MatrixOpsVisitor().visit(tree)
        return results

    def _detect_unnecessary_array_copies(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect unnecessary array copying operations."""
        results = []

        class ArrayCopyVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Detect .copy() calls that might be unnecessary
                if isinstance(node.func, ast.Attribute) and node.func.attr == "copy":

                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.UNNECESSARY_ARRAY_COPY,
                            severity="low",
                            confidence=0.5,
                            description="Array copy detected - verify if necessary for performance",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Consider if array copy is necessary or if views can be used",
                            metrics={"optimization_type": "array_copy"},
                        )
                    )

                self.generic_visit(node)

        ArrayCopyVisitor().visit(tree)
        return results

    def _detect_inefficient_broadcasting(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect inefficient broadcasting patterns."""
        results = []

        class BroadcastingVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                # Look for manual broadcasting in loops
                if self._is_manual_broadcasting_loop(node):
                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.INEFFICIENT_BROADCASTING,
                            severity="medium",
                            confidence=0.7,
                            description="Manual broadcasting detected - NumPy can handle this automatically",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Use NumPy's automatic broadcasting instead of explicit loops",
                            metrics={"optimization_type": "broadcasting"},
                        )
                    )

            def _is_manual_broadcasting_loop(self, node: ast.For) -> bool:
                """Check if this loop is doing manual broadcasting."""
                # Simplified check - could be more sophisticated
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        return True
                return False

        BroadcastingVisitor().visit(tree)
        return results

    def _detect_scalar_array_operations(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect scalar operations on arrays that could be vectorized."""
        results = []

        class ScalarArrayVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                # Look for scalar operations in loops over arrays
                if self._has_scalar_array_operations(node):
                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.SCALAR_ARRAY_OPERATIONS,
                            severity="medium",
                            confidence=0.7,
                            description="Scalar operations in loop could be vectorized",
                            location={
                                "file": str(file_path),
                                "line": node.lineno,
                                "column": node.col_offset,
                            },
                            suggestion="Use vectorized operations instead of scalar operations in loops",
                            metrics={"optimization_type": "vectorization"},
                        )
                    )

            def _has_scalar_array_operations(self, node: ast.For) -> bool:
                """Check if loop contains scalar operations on array elements."""
                # Simplified implementation
                return len(node.body) > 0

        ScalarArrayVisitor().visit(tree)
        return results

    def _detect_numba_opportunities(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect computational loops that could benefit from Numba JIT compilation."""
        results = []

        class NumbaOpportunityVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                if self._is_computational_intensive_loop(node):
                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.MISSING_NUMBA_OPPORTUNITY,
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            message="Computational loop could benefit from Numba JIT compilation",
                            severity="low",
                            confidence=0.6,
                            suggestion="Consider using @numba.jit decorator for performance improvement",
                            code_context=(
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else str(node)
                            ),
                        )
                    )
                self.generic_visit(node)

            def _is_computational_intensive_loop(self, node: ast.For) -> bool:
                """Check if loop is computationally intensive."""
                # Look for nested loops or complex computations
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.For) and stmt != node:
                        return True  # Nested loop
                    if isinstance(stmt, ast.BinOp):
                        return True  # Mathematical operations
                return False

        NumbaOpportunityVisitor().visit(tree)
        return results

    def _detect_wrong_dtype_usage(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect potentially suboptimal dtype usage."""
        results = []

        class DtypeVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Check for float64 when float32 might suffice
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ["zeros", "ones", "full", "array"]
                    and any(
                        isinstance(kw.value, ast.Constant)
                        and kw.value.value == "float64"
                        for kw in node.keywords
                        if kw.arg == "dtype"
                    )
                ):

                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.WRONG_DTYPE_USAGE,
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            message="Consider using float32 instead of float64 if precision allows",
                            severity="low",
                            confidence=0.4,
                            suggestion="Use float32 for better memory usage and performance if precision is sufficient",
                            code_context=(
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else str(node)
                            ),
                        )
                    )

                self.generic_visit(node)

        DtypeVisitor().visit(tree)
        return results

    def _detect_inefficient_concatenation(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect inefficient array concatenation patterns."""
        results = []

        class ConcatenationVisitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For):
                # Look for repeated np.concatenate or np.append in loops
                for stmt in ast.walk(node):
                    if (
                        isinstance(stmt, ast.Call)
                        and isinstance(stmt.func, ast.Attribute)
                        and stmt.func.attr
                        in ["concatenate", "append", "vstack", "hstack"]
                    ):

                        results.append(
                            AnalysisResult(
                                pattern_type=PatternType.INEFFICIENT_ARRAY_CONCATENATION,
                                severity="medium",
                                confidence=0.8,
                                description="Array concatenation in loop is inefficient",
                                location={
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "column": node.col_offset,
                                },
                                suggestion="Pre-allocate array and use indexing, or collect items and concatenate once",
                                metrics={"optimization_type": "array_concatenation"},
                            )
                        )
                        break

                self.generic_visit(node)

        ConcatenationVisitor().visit(tree)
        return results

    def _detect_suboptimal_linear_algebra(
        self, tree: ast.AST, file_path: Path
    ) -> List[AnalysisResult]:
        """Detect suboptimal linear algebra operations."""
        results = []

        class LinearAlgebraVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call):
                # Detect manual matrix inversion when solve could be used
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "linalg"
                    and node.func.attr == "inv"
                ):

                    results.append(
                        AnalysisResult(
                            pattern_type=PatternType.SUBOPTIMAL_LINEAR_ALGEBRA,
                            file_path=file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            message="Matrix inversion detected - consider using solve() for linear systems",
                            severity="medium",
                            confidence=0.7,
                            suggestion="Use np.linalg.solve() instead of matrix inversion for solving linear systems",
                            code_context=(
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else str(node)
                            ),
                        )
                    )

                self.generic_visit(node)

        LinearAlgebraVisitor().visit(tree)
        return results
