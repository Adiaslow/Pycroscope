"""
Dynamic execution analyzer for Pycroscope.

Analyzes runtime execution data to identify performance hotspots,
memory issues, and dynamic optimization opportunities.
"""

import statistics
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple

from .base_analyzer import BaseAnalyzer
from ..core.models import (
    ProfileSession,
    AnalysisResult,
    DynamicAnalysisResult,
    CallNode,
    DetectedPattern,
    OptimizationRecommendation,
    SourceLocation,
    MetricType,
)
from ..core.config import AnalysisConfig


class DynamicAnalyzer(BaseAnalyzer):
    """
    Dynamic execution analysis engine.

    Processes runtime profiling data to identify performance bottlenecks,
    memory usage patterns, and execution hotspots.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the dynamic analyzer.

        Args:
            config: Optional analysis configuration
        """
        super().__init__(config)
        self._hotspot_threshold = 0.05  # 5% of total time
        self._memory_growth_threshold = 1024 * 1024  # 1MB

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        return "dynamic"

    @property
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        return ["line", "memory", "call"]  # Needs runtime data

    def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Perform dynamic execution analysis.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results with dynamic insights
        """
        # Initialize results
        patterns = []
        recommendations = []
        performance_metrics = {}
        hotspots = []
        memory_leaks = []

        # Analyze execution events
        if profile_data.execution_events:
            event_patterns, event_metrics = self._analyze_execution_events(
                profile_data.execution_events
            )
            patterns.extend(event_patterns)
            performance_metrics.update(event_metrics)

        # Analyze memory snapshots
        if profile_data.memory_snapshots:
            memory_patterns = self._analyze_memory_patterns(
                profile_data.memory_snapshots
            )
            patterns.extend(memory_patterns)
            memory_leaks.extend(memory_patterns)

        # Analyze call tree
        if profile_data.call_tree:
            call_patterns, call_hotspots = self._analyze_call_tree(
                profile_data.call_tree
            )
            patterns.extend(call_patterns)
            hotspots.extend(call_hotspots)

        # Generate dynamic recommendations
        dynamic_recommendations = self._generate_dynamic_recommendations(
            patterns, performance_metrics
        )
        recommendations.extend(dynamic_recommendations)

        # Create dynamic analysis result
        dynamic_result = DynamicAnalysisResult(
            hotspots=hotspots,
            memory_leaks=memory_leaks,
            performance_metrics=performance_metrics,
        )

        # Calculate overall performance score
        overall_score = self._calculate_performance_score(patterns, performance_metrics)

        # Create complete analysis result
        from ..core.models import StaticAnalysisResult

        return AnalysisResult(
            session_id=profile_data.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=dynamic_result,
            detected_patterns=patterns,
            recommendations=recommendations,
            overall_score=overall_score,
            performance_grade=self._grade_from_score(overall_score),
        )

    def _analyze_execution_events(
        self, events
    ) -> Tuple[List[DetectedPattern], Dict[MetricType, float]]:
        """
        Analyze execution events for performance patterns.

        Args:
            events: List of execution events

        Returns:
            Tuple of (patterns, metrics)
        """
        patterns = []
        metrics = {}

        if not events:
            return patterns, metrics

        # Group events by type and location
        events_by_location = defaultdict(list)
        events_by_type = defaultdict(list)

        for event in events:
            location_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.line_number}"
            events_by_location[location_key].append(event)
            events_by_type[event.event_type].append(event)

        # Calculate total execution time
        total_time = sum(
            event.execution_time for event in events if event.execution_time is not None
        )

        metrics[MetricType.TIME] = total_time

        # Detect hotspots (lines consuming significant time)
        hotspot_threshold = total_time * self._hotspot_threshold

        for location, location_events in events_by_location.items():
            location_time = sum(
                event.execution_time
                for event in location_events
                if event.execution_time is not None
            )

            if location_time > hotspot_threshold:
                # Extract location info
                filename = location_events[0].frame_info.source_location.filename
                line_number = location_events[0].frame_info.source_location.line_number
                function_name = location_events[
                    0
                ].frame_info.source_location.function_name

                patterns.append(
                    DetectedPattern(
                        pattern_type="execution_hotspot",
                        severity=(
                            "high"
                            if location_time > hotspot_threshold * 2
                            else "medium"
                        ),
                        source_location=SourceLocation(
                            filename, line_number, function_name
                        ),
                        description=f"Execution hotspot consuming {location_time/1_000_000:.2f}ms ({location_time/total_time*100:.1f}% of total time)",
                        impact_estimate=location_time / total_time,
                        evidence={
                            "location_time": location_time,
                            "total_time": total_time,
                            "percentage": location_time / total_time * 100,
                            "event_count": len(location_events),
                        },
                    )
                )

        # Detect exception hotspots
        exception_events = events_by_type.get("exception", [])
        if len(exception_events) > 10:  # Threshold for too many exceptions
            exception_locations = Counter(
                f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.line_number}"
                for event in exception_events
            )

            for location, count in exception_locations.most_common(5):
                if count > 5:  # More than 5 exceptions at same location
                    event = next(
                        e
                        for e in exception_events
                        if f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.line_number}"
                        == location
                    )

                    patterns.append(
                        DetectedPattern(
                            pattern_type="exception_hotspot",
                            severity="high",
                            source_location=event.frame_info.source_location,
                            description=f"Frequent exceptions at this location: {count} exceptions",
                            impact_estimate=min(0.8, count / 20),
                            evidence={"exception_count": count},
                        )
                    )

        return patterns, metrics

    def _analyze_memory_patterns(self, memory_snapshots) -> List[DetectedPattern]:
        """
        Analyze memory snapshots for memory leaks and usage patterns.

        Args:
            memory_snapshots: List of memory snapshots

        Returns:
            List of detected memory patterns
        """
        patterns = []

        if len(memory_snapshots) < 2:
            return patterns

        # Sort snapshots by timestamp
        sorted_snapshots = sorted(memory_snapshots, key=lambda s: s.timestamp)

        # Detect memory growth trends
        memory_values = [snapshot.total_memory for snapshot in sorted_snapshots]

        # Calculate growth rate
        initial_memory = memory_values[0]
        final_memory = memory_values[-1]
        memory_growth = final_memory - initial_memory

        if memory_growth > self._memory_growth_threshold:
            # Check if growth is consistent (potential leak)
            growth_consistency = self._calculate_growth_consistency(memory_values)

            if growth_consistency > 0.7:  # 70% consistency indicates leak
                patterns.append(
                    DetectedPattern(
                        pattern_type="memory_leak",
                        severity=(
                            "critical" if memory_growth > 10 * 1024 * 1024 else "high"
                        ),
                        source_location=SourceLocation(
                            "memory_analysis", 1, "memory_leak"
                        ),
                        description=f"Potential memory leak detected: {memory_growth / (1024*1024):.2f}MB growth",
                        impact_estimate=min(1.0, memory_growth / (50 * 1024 * 1024)),
                        evidence={
                            "memory_growth_bytes": memory_growth,
                            "growth_consistency": growth_consistency,
                            "snapshots_analyzed": len(sorted_snapshots),
                        },
                    )
                )

        # Detect memory spikes
        if len(memory_values) > 5:
            memory_mean = statistics.mean(memory_values)
            memory_std = (
                statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            )

            for i, memory_value in enumerate(memory_values):
                if memory_value > memory_mean + 2 * memory_std:  # 2 standard deviations
                    patterns.append(
                        DetectedPattern(
                            pattern_type="memory_spike",
                            severity="medium",
                            source_location=SourceLocation(
                                "memory_analysis", i + 1, "memory_spike"
                            ),
                            description=f"Memory spike detected: {memory_value/(1024*1024):.2f}MB (2σ above mean)",
                            impact_estimate=min(
                                0.6, (memory_value - memory_mean) / (10 * 1024 * 1024)
                            ),
                            evidence={
                                "spike_memory": memory_value,
                                "mean_memory": memory_mean,
                                "std_dev": memory_std,
                            },
                        )
                    )

        return patterns

    def _analyze_call_tree(
        self, call_tree
    ) -> Tuple[List[DetectedPattern], List[CallNode]]:
        """
        Analyze call tree for performance patterns.

        Args:
            call_tree: Call tree data

        Returns:
            Tuple of (patterns, hotspots)
        """
        patterns = []
        hotspots = []

        if not call_tree:
            return patterns, hotspots

        # Find hotspots using call tree analysis
        call_hotspots = call_tree.find_hotspots(threshold_percent=5.0)
        hotspots.extend(call_hotspots)

        # Detect recursive patterns
        recursive_patterns = self._detect_recursive_calls(call_tree)
        patterns.extend(recursive_patterns)

        # Detect deep call stacks
        if call_tree.max_depth > 50:
            patterns.append(
                DetectedPattern(
                    pattern_type="deep_call_stack",
                    severity="medium",
                    source_location=SourceLocation("call_analysis", 1, "deep_stack"),
                    description=f"Very deep call stack detected: {call_tree.max_depth} levels",
                    impact_estimate=min(0.5, call_tree.max_depth / 100),
                    evidence={"max_depth": call_tree.max_depth},
                )
            )

        return patterns, hotspots

    def _detect_recursive_calls(self, call_tree) -> List[DetectedPattern]:
        """
        Detect potentially problematic recursive call patterns.

        Args:
            call_tree: Call tree to analyze

        Returns:
            List of recursive call patterns
        """
        patterns = []

        # This would require more sophisticated call tree traversal
        # For now, return placeholder analysis

        if call_tree.total_calls > 10000:  # High call count might indicate recursion
            patterns.append(
                DetectedPattern(
                    pattern_type="excessive_calls",
                    severity="medium",
                    source_location=SourceLocation(
                        "call_analysis", 1, "excessive_calls"
                    ),
                    description=f"High call count detected: {call_tree.total_calls} total calls",
                    impact_estimate=min(0.6, call_tree.total_calls / 50000),
                    evidence={"total_calls": call_tree.total_calls},
                )
            )

        return patterns

    def _calculate_growth_consistency(self, values: List[float]) -> float:
        """
        Calculate how consistently values are growing.

        Args:
            values: List of values to analyze

        Returns:
            Consistency score (0.0 to 1.0)
        """
        if len(values) < 3:
            return 0.0

        growth_directions = []
        for i in range(1, len(values)):
            if values[i] > values[i - 1]:
                growth_directions.append(1)
            elif values[i] < values[i - 1]:
                growth_directions.append(-1)
            else:
                growth_directions.append(0)

        # Calculate percentage of positive growth
        positive_growth = sum(1 for direction in growth_directions if direction > 0)
        return positive_growth / len(growth_directions)

    def _generate_dynamic_recommendations(
        self, patterns: List[DetectedPattern], metrics: Dict[MetricType, float]
    ) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations based on dynamic analysis.

        Args:
            patterns: Detected patterns
            metrics: Performance metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        for pattern in patterns:
            if pattern.pattern_type == "execution_hotspot":
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"hotspot_{pattern.source_location.line_number}",
                        title="Optimize Execution Hotspot",
                        description="This code location consumes significant execution time",
                        target_location=pattern.source_location,
                        estimated_improvement=2.0,
                        confidence=0.9,
                        complexity="medium",
                        suggested_approach="Profile the specific operations at this location and consider algorithmic improvements",
                        code_example="""
# Consider these optimization strategies:
# 1. Algorithmic improvements (O(n²) → O(n log n))
# 2. Caching frequently computed values
# 3. Moving invariant calculations outside loops
# 4. Using more efficient data structures
                        """.strip(),
                        addresses_patterns=[pattern.pattern_type],
                    )
                )

            elif pattern.pattern_type == "memory_leak":
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"memory_leak_{pattern.source_location.line_number}",
                        title="Fix Memory Leak",
                        description="Memory usage is growing consistently, indicating a potential leak",
                        target_location=pattern.source_location,
                        estimated_improvement=3.0,
                        confidence=0.8,
                        complexity="high",
                        suggested_approach="Identify objects that are not being properly released and fix reference cycles",
                        code_example="""
# Common memory leak fixes:
# 1. Use weak references for callbacks/observers
# 2. Explicitly close files, connections, etc.
# 3. Clear large data structures when done
# 4. Use context managers for resource management

import weakref
from contextlib import contextmanager

# Use weak references for callbacks
self.callbacks = weakref.WeakSet()

# Use context managers
@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        resource.close()
                        """.strip(),
                        addresses_patterns=[pattern.pattern_type],
                    )
                )

            elif pattern.pattern_type == "exception_hotspot":
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"exception_{pattern.source_location.line_number}",
                        title="Reduce Exception Frequency",
                        description="This location generates many exceptions, impacting performance",
                        target_location=pattern.source_location,
                        estimated_improvement=1.5,
                        confidence=0.7,
                        complexity="low",
                        suggested_approach="Add validation checks to prevent exceptions or use alternative control flow",
                        addresses_patterns=[pattern.pattern_type],
                    )
                )

        return recommendations

    def _calculate_performance_score(
        self, patterns: List[DetectedPattern], metrics: Dict[MetricType, float]
    ) -> float:
        """
        Calculate overall performance score based on dynamic analysis.

        Args:
            patterns: Detected patterns
            metrics: Performance metrics

        Returns:
            Performance score (0-100)
        """
        base_score = 80.0  # Start higher for dynamic analysis

        # Deduct for patterns
        for pattern in patterns:
            if pattern.pattern_type == "memory_leak":
                base_score -= 25
            elif pattern.pattern_type == "execution_hotspot":
                base_score -= 15
            elif pattern.pattern_type == "exception_hotspot":
                base_score -= 10
            elif pattern.severity == "high":
                base_score -= 8
            elif pattern.severity == "medium":
                base_score -= 4
            elif pattern.severity == "low":
                base_score -= 2

        return max(0.0, base_score)

    def _grade_from_score(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 60:
            return "poor"
        else:
            return "critical"
