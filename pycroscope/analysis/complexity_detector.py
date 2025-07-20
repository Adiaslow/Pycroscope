"""
Algorithm complexity detector for Pycroscope.

Empirically analyzes execution patterns to determine algorithmic complexity
of functions and detect potential performance scaling issues.
"""

import math
import statistics
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from ..core.models import (
    AnalysisResult,
    DetectedPattern,
    DynamicAnalysisResult,
    OptimizationRecommendation,
    ProfileSession,
    SourceLocation,
    StaticAnalysisResult,
)
from .base_analyzer import BaseAnalyzer


class ComplexityClass(Enum):
    """Common algorithmic complexity classes."""

    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"
    UNKNOWN = "O(?)"


@dataclass
class ComplexityMeasurement:
    """Single complexity measurement for a function."""

    input_size: int
    execution_time: float  # nanoseconds
    memory_usage: Optional[int]  # bytes
    call_count: int


@dataclass
class ComplexityAnalysis:
    """Complete complexity analysis for a function."""

    function_name: str
    source_location: SourceLocation
    detected_complexity: ComplexityClass
    confidence: float
    measurements: List[ComplexityMeasurement]
    fit_quality: float  # R² value
    scaling_factor: float
    evidence: Dict[str, Any]


ComplexityFunction = namedtuple(
    "ComplexityFunction", ["func", "name", "complexity_class"]
)


class AlgorithmComplexityDetector(BaseAnalyzer):
    """
    Algorithm complexity detection engine.

    Analyzes execution patterns to empirically determine the algorithmic
    complexity of functions by fitting mathematical models to execution data.
    """

    def __init__(self, config=None):
        """
        Initialize the complexity detector.

        Args:
            config: Optional analysis configuration
        """
        super().__init__(config)

        # Minimum measurements needed for complexity analysis
        self.min_measurements = 5

        # R² threshold for accepting complexity fits
        self.fit_threshold = 0.7

        # Define complexity function models
        self.complexity_models = [
            ComplexityFunction(
                lambda n, a: np.full_like(n, a), "O(1)", ComplexityClass.CONSTANT
            ),
            ComplexityFunction(
                lambda n, a, b: a * np.log(n) + b,
                "O(log n)",
                ComplexityClass.LOGARITHMIC,
            ),
            ComplexityFunction(
                lambda n, a, b: a * n + b, "O(n)", ComplexityClass.LINEAR
            ),
            ComplexityFunction(
                lambda n, a, b: a * n * np.log(n) + b,
                "O(n log n)",
                ComplexityClass.LINEARITHMIC,
            ),
            ComplexityFunction(
                lambda n, a, b: a * n**2 + b, "O(n²)", ComplexityClass.QUADRATIC
            ),
            ComplexityFunction(
                lambda n, a, b: a * n**3 + b, "O(n³)", ComplexityClass.CUBIC
            ),
            ComplexityFunction(
                lambda n, a, b: a * 2**n + b, "O(2^n)", ComplexityClass.EXPONENTIAL
            ),
        ]

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        return "complexity"

    @property
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        return ["line", "call", "memory"]

    def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Perform algorithm complexity analysis.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results with complexity insights
        """
        patterns = []
        recommendations = []

        # Extract execution measurements by function
        function_measurements = self._extract_function_measurements(
            profile_data.execution_events
        )

        # Analyze complexity for each function
        complexity_analyses = []
        for func_key, measurements in function_measurements.items():
            if len(measurements) >= self.min_measurements:
                analysis = self._analyze_function_complexity(func_key, measurements)
                if analysis:
                    complexity_analyses.append(analysis)

        # Generate patterns from complexity analyses
        for analysis in complexity_analyses:
            pattern = self._create_complexity_pattern(analysis)
            if pattern:
                patterns.append(pattern)

        # Generate recommendations
        complexity_recommendations = self._generate_complexity_recommendations(
            complexity_analyses
        )
        recommendations.extend(complexity_recommendations)

        # Calculate overall complexity score
        overall_score = self._calculate_complexity_score(complexity_analyses)

        return AnalysisResult(
            session_id=profile_data.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=patterns,
            recommendations=recommendations,
            overall_score=overall_score,
            performance_grade=self._grade_from_score(overall_score),
        )

    def _extract_function_measurements(
        self, events
    ) -> Dict[str, List[ComplexityMeasurement]]:
        """Extract execution measurements organized by function."""
        function_data = defaultdict(list)

        # Group events by function
        function_events = defaultdict(list)
        for event in events:
            if hasattr(event, "frame_info") and event.frame_info.source_location:
                func_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.function_name}"
                function_events[func_key].append(event)

        # Analyze each function's execution patterns
        for func_key, func_events in function_events.items():
            measurements = self._extract_measurements_from_events(func_events)
            if measurements:
                function_data[func_key] = measurements

        return dict(function_data)

    def _extract_measurements_from_events(self, events) -> List[ComplexityMeasurement]:
        """Extract complexity measurements from function events."""
        measurements = []

        # Group events by execution instance (assuming similar timestamps = same call)
        execution_groups = self._group_events_by_execution(events)

        for group in execution_groups:
            # Estimate input size from various metrics
            input_size = self._estimate_input_size(group)

            if input_size > 0:
                # Calculate execution metrics
                total_time = sum(e.execution_time or 0 for e in group)
                memory_usage = max(
                    (e.memory_delta for e in group if e.memory_delta), default=None
                )
                call_count = len(group)

                measurement = ComplexityMeasurement(
                    input_size=input_size,
                    execution_time=total_time,
                    memory_usage=memory_usage,
                    call_count=call_count,
                )
                measurements.append(measurement)

        return measurements

    def _group_events_by_execution(self, events) -> List[List]:
        """Group events that belong to the same function execution."""
        if not events:
            return []

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        groups = []
        current_group = [sorted_events[0]]

        for event in sorted_events[1:]:
            # If events are close in time (within 10ms), consider them same execution
            time_diff = event.timestamp - current_group[-1].timestamp
            if time_diff < 10_000_000:  # 10ms in nanoseconds
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]

        if current_group:
            groups.append(current_group)

        return groups

    def _estimate_input_size(self, events) -> int:
        """Estimate the input size for a function execution."""

        # Strategy 1: Look for line execution counts (indicates iterations)
        line_events = [
            e for e in events if hasattr(e, "event_type") and e.event_type == "line"
        ]
        if line_events:
            # Group by line number to find loops
            line_counts = defaultdict(int)
            for event in line_events:
                if hasattr(event, "frame_info") and event.frame_info.source_location:
                    line_counts[event.frame_info.source_location.line_number] += 1

            # Find the line with most executions (likely loop body)
            if line_counts:
                max_line_count = max(line_counts.values())
                if max_line_count > 1:
                    return max_line_count

        # Strategy 2: Look for memory allocations (indicates data size)
        memory_events = [e for e in events if hasattr(e, "event_data") and e.event_data]
        total_allocated = 0
        for event in memory_events:
            if event.event_data and "bytes_allocated" in event.event_data:
                total_allocated += event.event_data["bytes_allocated"]

        if total_allocated > 0:
            # Estimate size based on allocation (assuming 8 bytes per element on average)
            estimated_elements = total_allocated // 8
            return max(1, estimated_elements)

        # Strategy 3: Use call count as proxy for input size
        call_events = [
            e for e in events if hasattr(e, "event_type") and e.event_type == "call"
        ]
        if len(call_events) > 1:
            return len(call_events)

        # Default: return 1 for constant-time operations
        return 1

    def _analyze_function_complexity(
        self, func_key: str, measurements: List[ComplexityMeasurement]
    ) -> Optional[ComplexityAnalysis]:
        """Analyze the algorithmic complexity of a function."""

        if len(measurements) < self.min_measurements:
            return None

        # Sort measurements by input size
        sorted_measurements = sorted(measurements, key=lambda m: m.input_size)

        # Extract data for curve fitting
        input_sizes = np.array([m.input_size for m in sorted_measurements], dtype=float)
        execution_times = np.array(
            [m.execution_time for m in sorted_measurements], dtype=float
        )

        # Filter out zero or negative values
        valid_indices = (input_sizes > 0) & (execution_times > 0)
        input_sizes = input_sizes[valid_indices]
        execution_times = execution_times[valid_indices]

        if len(input_sizes) < self.min_measurements:
            return None

        # Try to fit each complexity model
        best_fit = None
        best_r_squared = -1

        for model in self.complexity_models:
            try:
                fit_result = self._fit_complexity_model(
                    input_sizes, execution_times, model
                )
                if fit_result and fit_result["r_squared"] > best_r_squared:
                    best_r_squared = fit_result["r_squared"]
                    best_fit = {
                        "model": model,
                        "r_squared": fit_result["r_squared"],
                        "parameters": fit_result["parameters"],
                        "scaling_factor": (
                            fit_result["parameters"][0]
                            if fit_result["parameters"]
                            else 0
                        ),
                    }
            except Exception:
                # Skip models that fail to fit
                continue

        if not best_fit or best_fit["r_squared"] < self.fit_threshold:
            # No good fit found
            detected_complexity = ComplexityClass.UNKNOWN
            confidence = 0.0
            fit_quality = 0.0
            scaling_factor = 0.0
        else:
            detected_complexity = best_fit["model"].complexity_class
            confidence = min(1.0, best_fit["r_squared"])
            fit_quality = best_fit["r_squared"]
            scaling_factor = best_fit["scaling_factor"]

        # Extract source location from func_key
        filename, function_name = func_key.split(":", 1)
        source_location = SourceLocation(
            filename, 1, function_name
        )  # Line number unknown

        return ComplexityAnalysis(
            function_name=function_name,
            source_location=source_location,
            detected_complexity=detected_complexity,
            confidence=confidence,
            measurements=sorted_measurements,
            fit_quality=fit_quality,
            scaling_factor=scaling_factor,
            evidence={
                "measurement_count": len(sorted_measurements),
                "input_size_range": (min(input_sizes), max(input_sizes)),
                "execution_time_range": (min(execution_times), max(execution_times)),
                "best_fit_model": best_fit["model"].name if best_fit else None,
            },
        )

    def _fit_complexity_model(
        self,
        input_sizes: np.ndarray,
        execution_times: np.ndarray,
        model: ComplexityFunction,
    ) -> Optional[Dict]:
        """Fit a specific complexity model to the data."""

        try:
            # Handle special cases for different models
            if model.complexity_class == ComplexityClass.CONSTANT:
                # For constant time, just use mean
                predicted = np.full_like(input_sizes, np.mean(execution_times))
                parameters = [np.mean(execution_times)]
            elif model.complexity_class == ComplexityClass.EXPONENTIAL:
                # For exponential, limit input sizes to avoid overflow
                if np.max(input_sizes) > 20:
                    return None  # Skip exponential fit for large inputs
                parameters, _ = curve_fit(
                    model.func, input_sizes, execution_times, maxfev=1000
                )
                predicted = model.func(input_sizes, *parameters)
            else:
                # Standard curve fitting
                parameters, _ = curve_fit(
                    model.func, input_sizes, execution_times, maxfev=1000
                )
                predicted = model.func(input_sizes, *parameters)

            # Calculate R-squared
            ss_res = np.sum((execution_times - predicted) ** 2)
            ss_tot = np.sum((execution_times - np.mean(execution_times)) ** 2)

            if ss_tot == 0:
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)

            return {
                "parameters": parameters,
                "r_squared": max(0.0, r_squared),  # Ensure non-negative
                "predicted": predicted,
            }

        except Exception:
            return None

    def _create_complexity_pattern(
        self, analysis: ComplexityAnalysis
    ) -> Optional[DetectedPattern]:
        """Create a detected pattern from complexity analysis."""

        if analysis.confidence < 0.5:
            return None

        # Determine severity based on complexity class
        severity_map = {
            ComplexityClass.CONSTANT: "low",
            ComplexityClass.LOGARITHMIC: "low",
            ComplexityClass.LINEAR: "low",
            ComplexityClass.LINEARITHMIC: "medium",
            ComplexityClass.QUADRATIC: "high",
            ComplexityClass.CUBIC: "high",
            ComplexityClass.EXPONENTIAL: "critical",
            ComplexityClass.FACTORIAL: "critical",
            ComplexityClass.UNKNOWN: "medium",
        }

        severity = severity_map.get(analysis.detected_complexity, "medium")

        # Calculate impact estimate
        impact_estimate = self._calculate_complexity_impact(analysis)

        # Create description
        if analysis.detected_complexity in [
            ComplexityClass.QUADRATIC,
            ComplexityClass.CUBIC,
            ComplexityClass.EXPONENTIAL,
        ]:
            description = f"Inefficient algorithm detected - {analysis.detected_complexity.value} complexity"
        elif analysis.detected_complexity == ComplexityClass.UNKNOWN:
            description = f"Unknown complexity pattern - unable to determine algorithmic complexity"
        else:
            description = f"Algorithm complexity: {analysis.detected_complexity.value}"

        return DetectedPattern(
            pattern_type="algorithm_complexity",
            severity=severity,
            source_location=analysis.source_location,
            description=description,
            impact_estimate=impact_estimate,
            evidence={
                "detected_complexity": analysis.detected_complexity.value,
                "confidence": analysis.confidence,
                "fit_quality": analysis.fit_quality,
                "scaling_factor": analysis.scaling_factor,
                "measurement_count": len(analysis.measurements),
                "input_size_range": analysis.evidence["input_size_range"],
                "function_name": analysis.function_name,
            },
        )

    def _calculate_complexity_impact(self, analysis: ComplexityAnalysis) -> float:
        """Calculate the performance impact of detected complexity."""

        # Base impact on complexity class
        complexity_impact = {
            ComplexityClass.CONSTANT: 0.1,
            ComplexityClass.LOGARITHMIC: 0.2,
            ComplexityClass.LINEAR: 0.3,
            ComplexityClass.LINEARITHMIC: 0.5,
            ComplexityClass.QUADRATIC: 0.8,
            ComplexityClass.CUBIC: 0.9,
            ComplexityClass.EXPONENTIAL: 1.0,
            ComplexityClass.FACTORIAL: 1.0,
            ComplexityClass.UNKNOWN: 0.4,
        }

        base_impact = complexity_impact.get(analysis.detected_complexity, 0.5)

        # Adjust based on confidence
        confidence_adjusted = base_impact * analysis.confidence

        # Adjust based on scaling factor (higher scaling = higher impact)
        if analysis.scaling_factor > 0:
            # Normalize scaling factor impact
            scaling_impact = min(0.3, math.log10(analysis.scaling_factor + 1) / 10)
            confidence_adjusted += scaling_impact

        return min(1.0, confidence_adjusted)

    def _generate_complexity_recommendations(
        self, analyses: List[ComplexityAnalysis]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on complexity analysis."""
        recommendations = []

        for analysis in analyses:
            if analysis.detected_complexity in [
                ComplexityClass.QUADRATIC,
                ComplexityClass.CUBIC,
                ComplexityClass.EXPONENTIAL,
            ]:
                recommendations.append(self._create_complexity_recommendation(analysis))

        return recommendations

    def _create_complexity_recommendation(
        self, analysis: ComplexityAnalysis
    ) -> OptimizationRecommendation:
        """Create a specific recommendation for high complexity functions."""

        if analysis.detected_complexity == ComplexityClass.QUADRATIC:
            title = "Optimize Quadratic Algorithm"
            description = f"Function {analysis.function_name} shows O(n²) complexity"
            actions = [
                "Consider using hash tables or sets for lookups instead of nested loops",
                "Implement more efficient sorting algorithms if applicable",
                "Cache intermediate results to avoid repeated computations",
                "Consider algorithmic alternatives with better complexity",
            ]
            examples = [
                "# Instead of nested loops for search:\n# for x in list1:\n#     for y in list2:\n#         if x == y: ...\n\n# Use set intersection:\ncommon = set(list1) & set(list2)"
            ]
            effort = "medium"

        elif analysis.detected_complexity == ComplexityClass.CUBIC:
            title = "Optimize Cubic Algorithm"
            description = f"Function {analysis.function_name} shows O(n³) complexity"
            actions = [
                "Break down the algorithm into smaller, more efficient parts",
                "Consider dynamic programming if the problem has overlapping subproblems",
                "Use memoization to cache expensive computations",
                "Look for mathematical optimizations to reduce complexity",
            ]
            examples = [
                "# Use memoization for expensive recursive functions:\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\ndef expensive_function(n):\n    # Your computation here"
            ]
            effort = "high"

        elif analysis.detected_complexity == ComplexityClass.EXPONENTIAL:
            title = "Critical: Exponential Algorithm"
            description = f"Function {analysis.function_name} shows O(2^n) complexity - critical performance issue"
            actions = [
                "URGENT: Replace with polynomial-time algorithm",
                "Implement dynamic programming solution",
                "Use approximation algorithms if exact solution not required",
                "Consider heuristic approaches for large inputs",
            ]
            examples = [
                "# Replace exponential recursion with dynamic programming:\n# memo = {}\n# def optimized_func(n):\n#     if n in memo:\n#         return memo[n]\n#     result = compute_result(n)\n#     memo[n] = result\n#     return result"
            ]
            effort = "critical"

        else:
            title = "Optimize Algorithm Complexity"
            description = f"Function {analysis.function_name} may benefit from algorithmic optimization"
            actions = [
                "Review algorithm for potential optimizations",
                "Consider alternative data structures",
                "Profile specific bottlenecks within the function",
            ]
            examples = []
            effort = "medium"

        return OptimizationRecommendation(
            category="algorithm",
            title=title,
            description=description,
            estimated_improvement=self._calculate_complexity_impact(analysis),
            confidence=analysis.confidence,
            implementation_effort=effort,
            suggested_actions=actions,
            code_examples=examples,
        )

    def _calculate_complexity_score(self, analyses: List[ComplexityAnalysis]) -> float:
        """Calculate overall complexity health score."""

        if not analyses:
            return 0.8  # Neutral score if no data

        # Score functions based on their complexity
        complexity_scores = {
            ComplexityClass.CONSTANT: 1.0,
            ComplexityClass.LOGARITHMIC: 0.9,
            ComplexityClass.LINEAR: 0.8,
            ComplexityClass.LINEARITHMIC: 0.7,
            ComplexityClass.QUADRATIC: 0.4,
            ComplexityClass.CUBIC: 0.2,
            ComplexityClass.EXPONENTIAL: 0.0,
            ComplexityClass.FACTORIAL: 0.0,
            ComplexityClass.UNKNOWN: 0.6,
        }

        # Calculate weighted average score
        total_weighted_score = 0
        total_weight = 0

        for analysis in analyses:
            base_score = complexity_scores.get(analysis.detected_complexity, 0.5)
            weight = analysis.confidence

            total_weighted_score += base_score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.8

        return total_weighted_score / total_weight

    def detect_complexity_regressions(
        self,
        current_analysis: List[ComplexityAnalysis],
        previous_analysis: Optional[List[ComplexityAnalysis]],
    ) -> List[DetectedPattern]:
        """Detect complexity regressions between analysis runs."""

        if not previous_analysis:
            return []

        regressions = []

        # Create lookup for previous analyses
        prev_by_function = {
            analysis.function_name: analysis for analysis in previous_analysis
        }

        for current in current_analysis:
            if current.function_name in prev_by_function:
                previous = prev_by_function[current.function_name]

                # Check for complexity regression
                regression = self._detect_single_complexity_regression(
                    current, previous
                )
                if regression:
                    regressions.append(regression)

        return regressions

    def _detect_single_complexity_regression(
        self, current: ComplexityAnalysis, previous: ComplexityAnalysis
    ) -> Optional[DetectedPattern]:
        """Detect regression between two complexity analyses of the same function."""

        # Define complexity ordering (lower is better)
        complexity_order = {
            ComplexityClass.CONSTANT: 0,
            ComplexityClass.LOGARITHMIC: 1,
            ComplexityClass.LINEAR: 2,
            ComplexityClass.LINEARITHMIC: 3,
            ComplexityClass.QUADRATIC: 4,
            ComplexityClass.CUBIC: 5,
            ComplexityClass.EXPONENTIAL: 6,
            ComplexityClass.FACTORIAL: 7,
            ComplexityClass.UNKNOWN: 3,  # Treat as moderate
        }

        current_order = complexity_order.get(current.detected_complexity, 3)
        previous_order = complexity_order.get(previous.detected_complexity, 3)

        # Check for regression (higher complexity)
        if current_order > previous_order:
            severity = "critical" if current_order - previous_order > 2 else "high"

            return DetectedPattern(
                pattern_type="complexity_regression",
                severity=severity,
                source_location=current.source_location,
                description=f"Complexity regression detected: {previous.detected_complexity.value} → {current.detected_complexity.value}",
                impact_estimate=min(1.0, (current_order - previous_order) / 4),
                evidence={
                    "previous_complexity": previous.detected_complexity.value,
                    "current_complexity": current.detected_complexity.value,
                    "regression_magnitude": current_order - previous_order,
                    "previous_confidence": previous.confidence,
                    "current_confidence": current.confidence,
                },
            )

        return None
