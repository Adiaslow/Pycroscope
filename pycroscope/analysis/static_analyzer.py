"""
Static code analyzer for Pycroscope.

Analyzes code structure, complexity, and static patterns without
execution data to identify potential optimization opportunities.
"""

import ast
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

from .base_analyzer import BaseAnalyzer
from ..core.models import (
    ProfileSession,
    AnalysisResult,
    StaticAnalysisResult,
    DetectedPattern,
    OptimizationRecommendation,
    SourceLocation,
)
from ..core.config import AnalysisConfig


class StaticAnalyzer(BaseAnalyzer):
    """
    Static code analysis engine.

    Performs code analysis without execution data to identify
    structural issues, complexity problems, and optimization opportunities.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the static analyzer.

        Args:
            config: Optional analysis configuration
        """
        super().__init__(config)
        self._complexity_thresholds = {
            "function": 10,  # McCabe complexity
            "class": 20,
            "module": 50,
        }

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        return "static"

    @property
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        return []  # Static analysis doesn't need runtime data

    def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Perform static code analysis.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results with static insights
        """
        # Initialize results
        patterns = []
        recommendations = []
        complexity_metrics = {}

        # Analyze source code from session source mapping
        if profile_data.source_mapping:
            for source_path, source_location in profile_data.source_mapping.items():
                try:
                    # Analyze file if it exists
                    if os.path.exists(source_location.filename):
                        file_patterns, file_recommendations, file_metrics = (
                            self._analyze_source_file(source_location.filename)
                        )
                        patterns.extend(file_patterns)
                        recommendations.extend(file_recommendations)
                        complexity_metrics[source_location.filename] = file_metrics
                except Exception as e:
                    # Skip files that can't be analyzed
                    continue

        # Calculate overall code quality score
        code_quality_score = self._calculate_code_quality_score(
            patterns, complexity_metrics
        )

        # Create static analysis result
        static_result = StaticAnalysisResult(
            complexity_metrics=complexity_metrics,
            detected_patterns=patterns,
            code_quality_score=code_quality_score,
        )

        # Create complete analysis result
        from ..core.models import DynamicAnalysisResult

        return AnalysisResult(
            session_id=profile_data.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=static_result,
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=patterns,
            recommendations=recommendations,
            overall_score=code_quality_score,
            performance_grade=self._grade_from_score(code_quality_score),
        )

    def _analyze_source_file(self, filename: str) -> tuple:
        """
        Analyze a single source file.

        Args:
            filename: Path to source file

        Returns:
            Tuple of (patterns, recommendations, metrics)
        """
        patterns = []
        recommendations = []
        metrics = {}

        try:
            with open(filename, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Parse AST
            tree = ast.parse(source_code, filename=filename)

            # Analyze complexity
            complexity_patterns, complexity_metrics = self._analyze_complexity(
                tree, filename
            )
            patterns.extend(complexity_patterns)
            metrics.update(complexity_metrics)

            # Detect anti-patterns
            antipattern_patterns = self._detect_antipatterns(tree, filename)
            patterns.extend(antipattern_patterns)

            # Generate recommendations
            file_recommendations = self._generate_static_recommendations(
                patterns, filename
            )
            recommendations.extend(file_recommendations)

        except Exception as e:
            # Add error pattern
            patterns.append(
                DetectedPattern(
                    pattern_type="analysis_error",
                    severity="low",
                    source_location=SourceLocation(filename, 1, "unknown"),
                    description=f"Failed to analyze file: {str(e)}",
                    impact_estimate=0.0,
                )
            )

        return patterns, recommendations, metrics

    def _analyze_complexity(self, tree: ast.AST, filename: str) -> tuple:
        """
        Analyze code complexity metrics.

        Args:
            tree: AST of the source code
            filename: Source filename

        Returns:
            Tuple of (patterns, metrics)
        """
        patterns = []
        metrics = {}

        # Analyze functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_mccabe_complexity(node)
                metrics[f"function_{node.name}"] = complexity

                if complexity > self._complexity_thresholds["function"]:
                    patterns.append(
                        DetectedPattern(
                            pattern_type="high_complexity",
                            severity="medium" if complexity < 20 else "high",
                            source_location=SourceLocation(
                                filename, node.lineno, node.name
                            ),
                            description=f"Function has high cyclomatic complexity: {complexity}",
                            impact_estimate=min(0.8, complexity / 20.0),
                            evidence={
                                "complexity": complexity,
                                "threshold": self._complexity_thresholds["function"],
                            },
                        )
                    )

            elif isinstance(node, ast.ClassDef):
                # Analyze class complexity
                class_methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                total_complexity = sum(
                    self._calculate_mccabe_complexity(method)
                    for method in class_methods
                )
                metrics[f"class_{node.name}"] = total_complexity

                if total_complexity > self._complexity_thresholds["class"]:
                    patterns.append(
                        DetectedPattern(
                            pattern_type="complex_class",
                            severity="medium",
                            source_location=SourceLocation(
                                filename, node.lineno, node.name
                            ),
                            description=f"Class has high total complexity: {total_complexity}",
                            impact_estimate=min(0.6, total_complexity / 50.0),
                            evidence={
                                "complexity": total_complexity,
                                "method_count": len(class_methods),
                            },
                        )
                    )

        return patterns, metrics

    def _calculate_mccabe_complexity(self, node: ast.FunctionDef) -> int:
        """
        Calculate McCabe cyclomatic complexity for a function.

        Args:
            node: Function AST node

        Returns:
            Complexity score
        """
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # AND/OR operations add complexity
                complexity += len(child.values) - 1
            elif isinstance(
                child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
            ):
                complexity += 1

        return complexity

    def _detect_antipatterns(
        self, tree: ast.AST, filename: str
    ) -> List[DetectedPattern]:
        """
        Detect common anti-patterns in code.

        Args:
            tree: AST of the source code
            filename: Source filename

        Returns:
            List of detected anti-pattern issues
        """
        patterns = []

        for node in ast.walk(tree):
            # Detect deeply nested loops
            if isinstance(node, (ast.For, ast.While)):
                nesting_depth = self._calculate_nesting_depth(node)
                if nesting_depth >= 3:
                    patterns.append(
                        DetectedPattern(
                            pattern_type="deep_nesting",
                            severity="medium" if nesting_depth == 3 else "high",
                            source_location=SourceLocation(
                                filename, node.lineno, "nested_loop"
                            ),
                            description=f"Deeply nested loop (depth: {nesting_depth})",
                            impact_estimate=min(0.7, nesting_depth / 5.0),
                            evidence={"nesting_depth": nesting_depth},
                        )
                    )

            # Detect long parameter lists
            elif isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 7:
                    patterns.append(
                        DetectedPattern(
                            pattern_type="long_parameter_list",
                            severity="low" if param_count <= 10 else "medium",
                            source_location=SourceLocation(
                                filename, node.lineno, node.name
                            ),
                            description=f"Function has too many parameters: {param_count}",
                            impact_estimate=min(0.4, param_count / 15.0),
                            evidence={"parameter_count": param_count},
                        )
                    )

            # Detect large classes
            elif isinstance(node, ast.ClassDef):
                method_count = len(
                    [n for n in node.body if isinstance(n, ast.FunctionDef)]
                )
                if method_count > 20:
                    patterns.append(
                        DetectedPattern(
                            pattern_type="large_class",
                            severity="medium",
                            source_location=SourceLocation(
                                filename, node.lineno, node.name
                            ),
                            description=f"Class has too many methods: {method_count}",
                            impact_estimate=min(0.6, method_count / 30.0),
                            evidence={"method_count": method_count},
                        )
                    )

        return patterns

    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """
        Calculate the maximum nesting depth of loops/conditionals.

        Args:
            node: AST node to analyze

        Returns:
            Maximum nesting depth
        """
        max_depth = 0

        def calculate_depth(current_node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)

            for child in ast.iter_child_nodes(current_node):
                if isinstance(child, (ast.For, ast.While, ast.If)):
                    calculate_depth(child, current_depth + 1)
                else:
                    calculate_depth(child, current_depth)

        calculate_depth(node, 1)
        return max_depth

    def _generate_static_recommendations(
        self, patterns: List[DetectedPattern], filename: str
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on detected patterns.

        Args:
            patterns: Detected patterns
            filename: Source filename

        Returns:
            List of recommendations
        """
        recommendations = []

        for pattern in patterns:
            if pattern.pattern_type == "high_complexity":
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"complexity_{pattern.source_location.function_name}",
                        title="Reduce Function Complexity",
                        description="Break down complex function into smaller, more focused functions",
                        target_location=pattern.source_location,
                        estimated_improvement=1.5,
                        confidence=0.8,
                        complexity="medium",
                        suggested_approach="Extract method refactoring to split complex logic",
                        code_example="""
# Instead of one complex function:
def complex_function(data):
    # ... 50+ lines of complex logic ...

# Split into focused functions:
def process_data(data):
    validated = validate_data(data)
    transformed = transform_data(validated)
    return finalize_data(transformed)

def validate_data(data): ...
def transform_data(data): ...
def finalize_data(data): ...
                        """.strip(),
                        addresses_patterns=[pattern.pattern_type],
                    )
                )

            elif pattern.pattern_type == "deep_nesting":
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"nesting_{pattern.source_location.line_number}",
                        title="Reduce Nesting Depth",
                        description="Use early returns or extract nested logic to reduce complexity",
                        target_location=pattern.source_location,
                        estimated_improvement=1.3,
                        confidence=0.7,
                        complexity="low",
                        suggested_approach="Use guard clauses and early returns to flatten structure",
                        addresses_patterns=[pattern.pattern_type],
                    )
                )

        return recommendations

    def _calculate_code_quality_score(
        self, patterns: List[DetectedPattern], metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall code quality score.

        Args:
            patterns: Detected patterns
            metrics: Complexity metrics

        Returns:
            Quality score (0-100)
        """
        base_score = 100.0

        # Deduct for patterns
        for pattern in patterns:
            if pattern.severity == "critical":
                base_score -= 20
            elif pattern.severity == "high":
                base_score -= 10
            elif pattern.severity == "medium":
                base_score -= 5
            elif pattern.severity == "low":
                base_score -= 2

        # Deduct for high complexity
        avg_complexity = (
            sum(v for v in metrics.values() if isinstance(v, (int, float)))
            / len(metrics)
            if metrics
            else 0
        )

        if avg_complexity > 15:
            base_score -= min(20, avg_complexity - 15)

        return max(0.0, base_score)

    def _grade_from_score(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
