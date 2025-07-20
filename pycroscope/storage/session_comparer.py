"""
Session comparison utilities for Pycroscope.

Provides tools for comparing profiling sessions to analyze optimization
results and track performance changes over time.
"""

import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import DetectedPattern, OptimizationRecommendation, ProfileSession


class ComparisonType(Enum):
    """Types of comparisons that can be performed."""

    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NEUTRAL = "neutral"
    INCONCLUSIVE = "inconclusive"


class MetricType(Enum):
    """Types of metrics that can be compared."""

    TOTAL_TIME = "total_time"
    PEAK_MEMORY = "peak_memory"
    TOTAL_EVENTS = "total_events"
    FUNCTION_CALLS = "function_calls"
    MEMORY_ALLOCATIONS = "memory_allocations"
    IO_OPERATIONS = "io_operations"
    EXCEPTION_COUNT = "exception_count"
    GC_COLLECTIONS = "gc_collections"


@dataclass
class MetricComparison:
    """Comparison result for a specific metric."""

    metric_type: MetricType
    baseline_value: float
    comparison_value: float
    percentage_change: float
    comparison_type: ComparisonType
    significance: float  # Statistical significance (0.0 to 1.0)

    @property
    def improvement_factor(self) -> float:
        """Calculate improvement factor (higher is better)."""
        if self.baseline_value == 0:
            return 1.0
        return (
            self.baseline_value / self.comparison_value
            if self.comparison_value > 0
            else 1.0
        )


@dataclass
class SessionComparison:
    """Complete comparison between two profiling sessions."""

    baseline_session_id: str
    comparison_session_id: str
    comparison_timestamp: datetime
    overall_assessment: ComparisonType
    confidence_level: float

    # Detailed comparisons
    metric_comparisons: List[MetricComparison]
    pattern_changes: Dict[str, Any]
    hotspot_changes: List[Dict[str, Any]]

    # Summary insights
    improvement_highlights: List[str]
    regression_risks: List[str]
    optimization_opportunities: List[str]

    @property
    def total_improvement(self) -> float:
        """Calculate overall improvement percentage."""
        improvements = [
            mc.percentage_change
            for mc in self.metric_comparisons
            if mc.comparison_type == ComparisonType.IMPROVEMENT
        ]
        return statistics.mean(improvements) if improvements else 0.0

    @property
    def major_improvements(self) -> List[MetricComparison]:
        """Get metrics with significant improvements (>20%)."""
        return [
            mc
            for mc in self.metric_comparisons
            if mc.comparison_type == ComparisonType.IMPROVEMENT
            and abs(mc.percentage_change) > 20
        ]

    @property
    def regressions(self) -> List[MetricComparison]:
        """Get metrics that regressed."""
        return [
            mc
            for mc in self.metric_comparisons
            if mc.comparison_type == ComparisonType.REGRESSION
        ]


class SessionComparer:
    """
    Advanced session comparison engine.

    Provides comprehensive comparison capabilities for profiling sessions
    including statistical analysis and trend detection.
    """

    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize the session comparer.

        Args:
            significance_threshold: Threshold for statistical significance
        """
        self.significance_threshold = significance_threshold

    def compare_sessions(
        self, baseline: ProfileSession, comparison: ProfileSession
    ) -> SessionComparison:
        """
        Perform comprehensive comparison between two sessions.

        Args:
            baseline: Baseline profiling session
            comparison: Comparison profiling session

        Returns:
            Detailed comparison results
        """
        # Extract and compare metrics
        metric_comparisons = self._compare_all_metrics(baseline, comparison)

        # Analyze pattern changes
        pattern_changes = self._analyze_pattern_changes(baseline, comparison)

        # Compare hotspots
        hotspot_changes = self._compare_hotspots(baseline, comparison)

        # Generate overall assessment
        overall_assessment = self._determine_overall_assessment(metric_comparisons)

        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(metric_comparisons)

        # Generate insights
        improvement_highlights = self._generate_improvement_highlights(
            metric_comparisons
        )
        regression_risks = self._generate_regression_risks(metric_comparisons)
        optimization_opportunities = self._generate_optimization_opportunities(
            pattern_changes, hotspot_changes
        )

        return SessionComparison(
            baseline_session_id=baseline.session_id,
            comparison_session_id=comparison.session_id,
            comparison_timestamp=datetime.now(),
            overall_assessment=overall_assessment,
            confidence_level=confidence_level,
            metric_comparisons=metric_comparisons,
            pattern_changes=pattern_changes,
            hotspot_changes=hotspot_changes,
            improvement_highlights=improvement_highlights,
            regression_risks=regression_risks,
            optimization_opportunities=optimization_opportunities,
        )

    def _compare_all_metrics(
        self, baseline: ProfileSession, comparison: ProfileSession
    ) -> List[MetricComparison]:
        """Compare all available metrics between sessions."""
        comparisons = []

        # Total execution time
        if baseline.total_time > 0 and comparison.total_time > 0:
            comparisons.append(
                self._compare_metric(
                    MetricType.TOTAL_TIME, baseline.total_time, comparison.total_time
                )
            )

        # Peak memory usage
        if baseline.peak_memory > 0 and comparison.peak_memory > 0:
            comparisons.append(
                self._compare_metric(
                    MetricType.PEAK_MEMORY, baseline.peak_memory, comparison.peak_memory
                )
            )

        # Total events
        comparisons.append(
            self._compare_metric(
                MetricType.TOTAL_EVENTS, baseline.total_events, comparison.total_events
            )
        )

        # Function calls
        baseline_calls = self._count_function_calls(baseline)
        comparison_calls = self._count_function_calls(comparison)
        if baseline_calls > 0 and comparison_calls > 0:
            comparisons.append(
                self._compare_metric(
                    MetricType.FUNCTION_CALLS, baseline_calls, comparison_calls
                )
            )

        # Memory allocations
        baseline_allocs = self._count_memory_allocations(baseline)
        comparison_allocs = self._count_memory_allocations(comparison)
        if baseline_allocs > 0 and comparison_allocs > 0:
            comparisons.append(
                self._compare_metric(
                    MetricType.MEMORY_ALLOCATIONS, baseline_allocs, comparison_allocs
                )
            )

        # I/O operations
        baseline_io = self._count_io_operations(baseline)
        comparison_io = self._count_io_operations(comparison)
        if baseline_io > 0 and comparison_io > 0:
            comparisons.append(
                self._compare_metric(
                    MetricType.IO_OPERATIONS, baseline_io, comparison_io
                )
            )

        # Exception count
        baseline_exceptions = self._count_exceptions(baseline)
        comparison_exceptions = self._count_exceptions(comparison)
        comparisons.append(
            self._compare_metric(
                MetricType.EXCEPTION_COUNT, baseline_exceptions, comparison_exceptions
            )
        )

        # GC collections
        baseline_gc = self._count_gc_collections(baseline)
        comparison_gc = self._count_gc_collections(comparison)
        if baseline_gc > 0 and comparison_gc > 0:
            comparisons.append(
                self._compare_metric(
                    MetricType.GC_COLLECTIONS, baseline_gc, comparison_gc
                )
            )

        return comparisons

    def _compare_metric(
        self, metric_type: MetricType, baseline_value: float, comparison_value: float
    ) -> MetricComparison:
        """Compare a specific metric between sessions."""

        # Calculate percentage change
        if baseline_value == 0:
            percentage_change = 100.0 if comparison_value > 0 else 0.0
        else:
            percentage_change = (
                (comparison_value - baseline_value) / baseline_value
            ) * 100

        # Determine comparison type
        if abs(percentage_change) < 5:  # Less than 5% change
            comparison_type = ComparisonType.NEUTRAL
        elif percentage_change < 0:  # Decrease in value
            # For time, memory, events - decrease is improvement
            # For exceptions - decrease is also improvement
            comparison_type = ComparisonType.IMPROVEMENT
        else:  # Increase in value
            # For time, memory, events - increase is regression
            comparison_type = ComparisonType.REGRESSION

        # Calculate statistical significance (simplified)
        significance = min(1.0, abs(percentage_change) / 100)

        return MetricComparison(
            metric_type=metric_type,
            baseline_value=baseline_value,
            comparison_value=comparison_value,
            percentage_change=percentage_change,
            comparison_type=comparison_type,
            significance=significance,
        )

    def _analyze_pattern_changes(
        self, baseline: ProfileSession, comparison: ProfileSession
    ) -> Dict[str, Any]:
        """Analyze changes in detected patterns between sessions."""

        baseline_patterns = self._extract_patterns(baseline)
        comparison_patterns = self._extract_patterns(comparison)

        # Count patterns by type
        baseline_counts = {}
        comparison_counts = {}

        for pattern in baseline_patterns:
            pattern_type = pattern.pattern_type
            baseline_counts[pattern_type] = baseline_counts.get(pattern_type, 0) + 1

        for pattern in comparison_patterns:
            pattern_type = pattern.pattern_type
            comparison_counts[pattern_type] = comparison_counts.get(pattern_type, 0) + 1

        # Analyze changes
        pattern_changes = {
            "baseline_total": len(baseline_patterns),
            "comparison_total": len(comparison_patterns),
            "new_patterns": [],
            "resolved_patterns": [],
            "pattern_type_changes": {},
        }

        # Find new and resolved pattern types
        all_pattern_types = set(baseline_counts.keys()) | set(comparison_counts.keys())

        for pattern_type in all_pattern_types:
            baseline_count = baseline_counts.get(pattern_type, 0)
            comparison_count = comparison_counts.get(pattern_type, 0)

            change = comparison_count - baseline_count
            pattern_changes["pattern_type_changes"][pattern_type] = {
                "baseline_count": baseline_count,
                "comparison_count": comparison_count,
                "change": change,
            }

            if baseline_count == 0 and comparison_count > 0:
                pattern_changes["new_patterns"].append(pattern_type)
            elif baseline_count > 0 and comparison_count == 0:
                pattern_changes["resolved_patterns"].append(pattern_type)

        return pattern_changes

    def _compare_hotspots(
        self, baseline: ProfileSession, comparison: ProfileSession
    ) -> List[Dict[str, Any]]:
        """Compare performance hotspots between sessions."""

        baseline_hotspots = self._extract_hotspots(baseline)
        comparison_hotspots = self._extract_hotspots(comparison)

        hotspot_changes = []

        # Compare hotspots by function name
        baseline_by_function = {
            h.get("function_name", ""): h for h in baseline_hotspots
        }
        comparison_by_function = {
            h.get("function_name", ""): h for h in comparison_hotspots
        }

        all_functions = set(baseline_by_function.keys()) | set(
            comparison_by_function.keys()
        )

        for function_name in all_functions:
            baseline_hotspot = baseline_by_function.get(function_name)
            comparison_hotspot = comparison_by_function.get(function_name)

            if baseline_hotspot and comparison_hotspot:
                # Both sessions have this hotspot - compare performance
                baseline_time = baseline_hotspot.get("total_time", 0)
                comparison_time = comparison_hotspot.get("total_time", 0)

                if baseline_time > 0:
                    percentage_change = (
                        (comparison_time - baseline_time) / baseline_time
                    ) * 100

                    hotspot_changes.append(
                        {
                            "function_name": function_name,
                            "change_type": "modified",
                            "baseline_time": baseline_time,
                            "comparison_time": comparison_time,
                            "percentage_change": percentage_change,
                        }
                    )

            elif baseline_hotspot and not comparison_hotspot:
                # Hotspot resolved
                hotspot_changes.append(
                    {
                        "function_name": function_name,
                        "change_type": "resolved",
                        "baseline_time": baseline_hotspot.get("total_time", 0),
                        "comparison_time": 0,
                        "percentage_change": -100.0,
                    }
                )

            elif not baseline_hotspot and comparison_hotspot:
                # New hotspot appeared
                hotspot_changes.append(
                    {
                        "function_name": function_name,
                        "change_type": "new",
                        "baseline_time": 0,
                        "comparison_time": comparison_hotspot.get("total_time", 0),
                        "percentage_change": 100.0,
                    }
                )

        return hotspot_changes

    def _determine_overall_assessment(
        self, metric_comparisons: List[MetricComparison]
    ) -> ComparisonType:
        """Determine overall assessment based on metric comparisons."""

        if not metric_comparisons:
            return ComparisonType.INCONCLUSIVE

        improvement_count = sum(
            1
            for mc in metric_comparisons
            if mc.comparison_type == ComparisonType.IMPROVEMENT
        )
        regression_count = sum(
            1
            for mc in metric_comparisons
            if mc.comparison_type == ComparisonType.REGRESSION
        )
        neutral_count = sum(
            1
            for mc in metric_comparisons
            if mc.comparison_type == ComparisonType.NEUTRAL
        )

        total_count = len(metric_comparisons)

        # Determine overall assessment
        if (
            improvement_count > regression_count
            and improvement_count > total_count * 0.4
        ):
            return ComparisonType.IMPROVEMENT
        elif (
            regression_count > improvement_count
            and regression_count > total_count * 0.4
        ):
            return ComparisonType.REGRESSION
        elif neutral_count > total_count * 0.6:
            return ComparisonType.NEUTRAL
        else:
            return ComparisonType.INCONCLUSIVE

    def _calculate_confidence_level(
        self, metric_comparisons: List[MetricComparison]
    ) -> float:
        """Calculate confidence level for the comparison."""

        if not metric_comparisons:
            return 0.0

        # Base confidence on number of metrics and their significance
        significance_scores = [mc.significance for mc in metric_comparisons]
        avg_significance = statistics.mean(significance_scores)

        # More metrics = higher confidence
        metric_count_factor = min(1.0, len(metric_comparisons) / 5)

        # Combine factors
        confidence = (avg_significance + metric_count_factor) / 2

        return min(1.0, confidence)

    def _generate_improvement_highlights(
        self, metric_comparisons: List[MetricComparison]
    ) -> List[str]:
        """Generate improvement highlights from metric comparisons."""

        highlights = []

        for mc in metric_comparisons:
            if (
                mc.comparison_type == ComparisonType.IMPROVEMENT
                and abs(mc.percentage_change) > 10
            ):
                metric_name = mc.metric_type.value.replace("_", " ").title()
                highlights.append(
                    f"{metric_name} improved by {abs(mc.percentage_change):.1f}%"
                )

        return highlights

    def _generate_regression_risks(
        self, metric_comparisons: List[MetricComparison]
    ) -> List[str]:
        """Generate regression risk warnings from metric comparisons."""

        risks = []

        for mc in metric_comparisons:
            if (
                mc.comparison_type == ComparisonType.REGRESSION
                and abs(mc.percentage_change) > 15
            ):
                metric_name = mc.metric_type.value.replace("_", " ").title()
                risks.append(
                    f"{metric_name} regressed by {abs(mc.percentage_change):.1f}%"
                )

        return risks

    def _generate_optimization_opportunities(
        self, pattern_changes: Dict[str, Any], hotspot_changes: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate optimization opportunity suggestions."""

        opportunities = []

        # Check for new patterns
        for pattern_type in pattern_changes.get("new_patterns", []):
            opportunities.append(
                f"Address new {pattern_type.replace('_', ' ')} pattern"
            )

        # Check for new hotspots
        new_hotspots = [h for h in hotspot_changes if h["change_type"] == "new"]
        if new_hotspots:
            opportunities.append(
                f"Optimize {len(new_hotspots)} new performance hotspots"
            )

        # Check for increased pattern counts
        for pattern_type, changes in pattern_changes.get(
            "pattern_type_changes", {}
        ).items():
            if changes["change"] > 0:
                opportunities.append(
                    f"Reduce {pattern_type.replace('_', ' ')} occurrences"
                )

        return opportunities

    # Helper methods for extracting data from sessions

    def _extract_patterns(self, session: ProfileSession) -> List[DetectedPattern]:
        """Extract detected patterns from session."""
        if session.analysis_result:
            return session.analysis_result.detected_patterns
        return []

    def _extract_hotspots(self, session: ProfileSession) -> List[Dict[str, Any]]:
        """Extract performance hotspots from session."""
        hotspots = []

        if session.call_tree:
            # Extract hotspots from call tree
            tree_hotspots = session.call_tree.find_hotspots()
            for hotspot in tree_hotspots:
                hotspots.append(
                    {
                        "function_name": hotspot.source_location.function_name,
                        "total_time": hotspot.total_time,
                        "self_time": hotspot.self_time,
                        "call_count": hotspot.call_count,
                    }
                )

        return hotspots

    def _count_function_calls(self, session: ProfileSession) -> int:
        """Count function call events in session."""
        from ..core.models import EventType

        return sum(
            1
            for event in session.execution_events
            if event.event_type == EventType.CALL
        )

    def _count_memory_allocations(self, session: ProfileSession) -> int:
        """Count memory allocation events in session."""
        from ..core.models import EventType

        return sum(
            1
            for event in session.execution_events
            if event.event_type == EventType.MEMORY_ALLOC
        )

    def _count_io_operations(self, session: ProfileSession) -> int:
        """Count I/O operation events in session."""
        from ..core.models import EventType

        return sum(
            1
            for event in session.execution_events
            if event.event_type in [EventType.IO_READ, EventType.IO_WRITE]
        )

    def _count_exceptions(self, session: ProfileSession) -> int:
        """Count exception events in session."""
        from ..core.models import EventType

        return sum(
            1
            for event in session.execution_events
            if event.event_type == EventType.EXCEPTION
        )

    def _count_gc_collections(self, session: ProfileSession) -> int:
        """Count garbage collection events in session."""
        if session.memory_snapshots:
            return max(snapshot.gc_collections for snapshot in session.memory_snapshots)
        return 0


def quick_compare_sessions(
    baseline: ProfileSession, comparison: ProfileSession
) -> SessionComparison:
    """
    Quick comparison between two sessions using default settings.

    Args:
        baseline: Baseline profiling session
        comparison: Comparison profiling session

    Returns:
        Session comparison results
    """
    comparer = SessionComparer()
    return comparer.compare_sessions(baseline, comparison)


def compare_session_metrics(
    baseline: ProfileSession, comparison: ProfileSession
) -> Dict[str, float]:
    """
    Simple metric comparison returning percentage changes.

    Args:
        baseline: Baseline profiling session
        comparison: Comparison profiling session

    Returns:
        Dictionary of metric names to percentage changes
    """
    comparer = SessionComparer()
    metric_comparisons = comparer._compare_all_metrics(baseline, comparison)

    return {mc.metric_type.value: mc.percentage_change for mc in metric_comparisons}
