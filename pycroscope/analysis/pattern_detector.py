"""
Advanced pattern detection engine for Pycroscope.

Identifies sophisticated performance patterns and anti-patterns across
multi-dimensional profiling data from all collectors.
"""

import math
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

from ..core.models import (
    DetectedPattern,
    EventType,
    ExecutionEvent,
    OptimizationRecommendation,
    ProfileSession,
    SourceLocation,
)


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternCategory(Enum):
    """Categories of performance patterns."""

    ALGORITHMIC = "algorithmic"
    MEMORY = "memory"
    IO = "io"
    CPU = "cpu"
    EXCEPTION = "exception"
    IMPORT = "import"
    GC = "gc"
    CONCURRENCY = "concurrency"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class PatternEvidence:
    """Evidence supporting a detected pattern."""

    metrics: Dict[str, Any]
    sample_data: List[Any]
    confidence: float
    statistical_significance: float


from .base_analyzer import BaseAnalyzer


class AdvancedPatternDetector(BaseAnalyzer):
    """
    Advanced pattern detection engine.

    Analyzes multi-dimensional profiling data to identify sophisticated
    performance patterns, anti-patterns, and optimization opportunities.
    """

    def __init__(self, config=None, confidence_threshold: float = 0.7):
        """
        Initialize the pattern detector.

        Args:
            config: Optional analysis configuration
            confidence_threshold: Minimum confidence for pattern detection
        """
        super().__init__(config)
        self.confidence_threshold = confidence_threshold

        # Pattern detection thresholds
        self.thresholds = {
            "loop_iteration_threshold": 1000,
            "recursive_depth_threshold": 100,
            "memory_leak_growth_mb": 10,
            "io_bottleneck_threshold_ms": 100,
            "exception_frequency_threshold": 10,
            "import_time_threshold_ms": 500,
            "gc_frequency_threshold": 5,
            "nested_loop_threshold": 3,
        }

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        return "pattern"

    @property
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        return ["line", "memory", "call", "io", "cpu", "gc", "exception", "import"]

    def _perform_analysis(self, profile_data):
        """
        Perform pattern detection analysis.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results with detected patterns
        """
        # Import here to avoid circular imports
        from datetime import datetime

        from ..core.models import (
            AnalysisResult,
            DynamicAnalysisResult,
            StaticAnalysisResult,
        )

        patterns = self.detect_patterns(profile_data)

        return AnalysisResult(
            session_id=profile_data.session_id,
            analysis_timestamp=datetime.now(),
            static_analysis=StaticAnalysisResult(),
            dynamic_analysis=DynamicAnalysisResult(),
            detected_patterns=patterns,
            recommendations=[],
            overall_score=self._calculate_pattern_confidence_average(patterns),
            performance_grade=(
                "good"
                if self._calculate_pattern_confidence_average(patterns) > 0.7
                else "needs_improvement"
            ),
        )

    def _calculate_pattern_confidence_average(self, patterns) -> float:
        """Calculate average confidence of detected patterns."""
        if not patterns:
            return 0.8

        confidences = [self._calculate_pattern_confidence(p) for p in patterns]
        return sum(confidences) / len(confidences) if confidences else 0.8

    def detect_patterns(self, profile_session: ProfileSession) -> List[DetectedPattern]:
        """
        Detect all patterns in the profiling session.

        Args:
            profile_session: Complete profiling session data

        Returns:
            List of detected patterns with evidence
        """
        patterns = []

        # Algorithmic patterns
        patterns.extend(self._detect_algorithmic_patterns(profile_session))

        # Memory patterns
        patterns.extend(self._detect_memory_patterns(profile_session))

        # I/O patterns
        patterns.extend(self._detect_io_patterns(profile_session))

        # CPU patterns
        patterns.extend(self._detect_cpu_patterns(profile_session))

        # Exception patterns
        patterns.extend(self._detect_exception_patterns(profile_session))

        # Import patterns
        patterns.extend(self._detect_import_patterns(profile_session))

        # GC patterns
        patterns.extend(self._detect_gc_patterns(profile_session))

        # Cross-collector correlation patterns
        patterns.extend(self._detect_correlation_patterns(profile_session))

        # Filter by confidence threshold
        return [
            p
            for p in patterns
            if self._calculate_pattern_confidence(p) >= self.confidence_threshold
        ]

    def _detect_algorithmic_patterns(
        self, session: ProfileSession
    ) -> List[DetectedPattern]:
        """Detect algorithmic complexity and efficiency patterns."""
        patterns = []

        if not session.execution_events:
            return patterns

        # Group events by function for complexity analysis
        function_events = defaultdict(list)
        for event in session.execution_events:
            if event.event_type == EventType.CALL:
                func_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.function_name}"
                function_events[func_key].append(event)

        for func_key, events in function_events.items():
            if len(events) < 10:  # Need sufficient data
                continue

            # Detect nested loops (O(n²) or worse)
            nested_loop_pattern = self._detect_nested_loops(events)
            if nested_loop_pattern:
                patterns.append(nested_loop_pattern)

            # Detect inefficient search patterns
            search_pattern = self._detect_inefficient_search(events)
            if search_pattern:
                patterns.append(search_pattern)

            # Detect redundant computations
            redundant_pattern = self._detect_redundant_computation(events)
            if redundant_pattern:
                patterns.append(redundant_pattern)

        return patterns

    def _detect_nested_loops(
        self, events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect nested loop patterns indicating O(n²) complexity."""

        # Analyze timing patterns that suggest nested loops
        if len(events) < 20:
            return None

        # Sort events by timestamp to find execution patterns
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Look for quadratic growth in execution time
        execution_times = [e.execution_time or 0 for e in sorted_events]

        # Calculate if execution time grows quadratically with iterations
        if len(execution_times) >= 10:
            # Simple quadratic detection: check if later executions are much slower
            first_quarter = execution_times[: len(execution_times) // 4]
            last_quarter = execution_times[-len(execution_times) // 4 :]

            if first_quarter and last_quarter:
                avg_early = statistics.mean(first_quarter)
                avg_late = statistics.mean(last_quarter)

                # If later executions are significantly slower, possible nested loop
                if avg_late > avg_early * 4 and avg_early > 0:
                    location = sorted_events[0].frame_info.source_location

                    return DetectedPattern(
                        pattern_type="nested_loops",
                        severity=PatternSeverity.HIGH.value,
                        source_location=location,
                        description=f"Potential nested loops detected - execution time increases quadratically",
                        impact_estimate=min(0.9, avg_late / avg_early / 10),
                        evidence={
                            "early_avg_time": avg_early,
                            "late_avg_time": avg_late,
                            "slowdown_factor": avg_late / avg_early,
                            "event_count": len(events),
                            "pattern_category": PatternCategory.ALGORITHMIC.value,
                        },
                    )

        return None

    def _detect_inefficient_search(
        self, events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect inefficient linear search patterns."""

        # Look for patterns that suggest linear search in collections
        call_patterns = []
        for event in events:
            if event.event_data and "operation" in event.event_data:
                operation = event.event_data["operation"]
                if any(
                    op in operation.lower() for op in ["find", "search", "in", "index"]
                ):
                    call_patterns.append(event)

        if len(call_patterns) > 50:  # Many search operations
            # Calculate average search time
            search_times = [e.execution_time or 0 for e in call_patterns]
            if search_times:
                avg_search_time = statistics.mean(search_times)

                # If searches are taking a long time, might be inefficient
                if avg_search_time > 1_000_000:  # 1ms
                    location = call_patterns[0].frame_info.source_location

                    return DetectedPattern(
                        pattern_type="inefficient_search",
                        severity=PatternSeverity.MEDIUM.value,
                        source_location=location,
                        description=f"Inefficient search pattern detected - consider using hash tables or binary search",
                        impact_estimate=min(0.7, avg_search_time / 10_000_000),
                        evidence={
                            "search_count": len(call_patterns),
                            "avg_search_time_ms": avg_search_time / 1_000_000,
                            "total_search_time_ms": sum(search_times) / 1_000_000,
                            "pattern_category": PatternCategory.ALGORITHMIC.value,
                        },
                    )

        return None

    def _detect_redundant_computation(
        self, events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect redundant computation patterns."""

        # Group events by similar operations
        operation_groups = defaultdict(list)

        for event in events:
            if event.event_data and event.frame_info.source_location.line_number:
                # Group by line number and function - likely same computation
                key = f"{event.frame_info.source_location.function_name}:{event.frame_info.source_location.line_number}"
                operation_groups[key].append(event)

        for key, group_events in operation_groups.items():
            if len(group_events) > 20:  # Same line executed many times
                # Check if results are likely the same (same inputs)
                execution_times = [e.execution_time or 0 for e in group_events]

                if execution_times:
                    total_time = sum(execution_times)
                    avg_time = statistics.mean(execution_times)

                    # If significant time spent on repeated operations
                    if total_time > 50_000_000:  # 50ms total
                        location = group_events[0].frame_info.source_location

                        return DetectedPattern(
                            pattern_type="redundant_computation",
                            severity=PatternSeverity.MEDIUM.value,
                            source_location=location,
                            description=f"Redundant computation detected - consider caching results",
                            impact_estimate=min(0.8, total_time / 500_000_000),
                            evidence={
                                "repetition_count": len(group_events),
                                "total_time_ms": total_time / 1_000_000,
                                "avg_time_ms": avg_time / 1_000_000,
                                "pattern_category": PatternCategory.ALGORITHMIC.value,
                            },
                        )

        return None

    def _detect_memory_patterns(self, session: ProfileSession) -> List[DetectedPattern]:
        """Detect memory-related performance patterns."""
        patterns = []

        if not session.memory_snapshots:
            return patterns

        # Memory leak detection
        leak_pattern = self._detect_memory_leak(session.memory_snapshots)
        if leak_pattern:
            patterns.append(leak_pattern)

        # Memory fragmentation detection
        fragmentation_pattern = self._detect_memory_fragmentation(
            session.memory_snapshots
        )
        if fragmentation_pattern:
            patterns.append(fragmentation_pattern)

        # Excessive allocation patterns
        allocation_pattern = self._detect_excessive_allocations(session)
        if allocation_pattern:
            patterns.append(allocation_pattern)

        return patterns

    def _detect_memory_leak(self, snapshots) -> Optional[DetectedPattern]:
        """Detect memory leak patterns."""
        if len(snapshots) < 5:
            return None

        # Sort by timestamp
        sorted_snapshots = sorted(snapshots, key=lambda s: s.timestamp)
        memory_values = [s.total_memory for s in sorted_snapshots]

        # Calculate trend
        trend_slope = self._calculate_trend_slope(memory_values)

        # If memory is consistently growing
        if trend_slope > 1024 * 1024:  # 1MB per measurement
            total_growth = memory_values[-1] - memory_values[0]

            return DetectedPattern(
                pattern_type="memory_leak",
                severity=(
                    PatternSeverity.CRITICAL.value
                    if total_growth > 50 * 1024 * 1024
                    else PatternSeverity.HIGH.value
                ),
                source_location=SourceLocation("memory_analysis", 1, "memory_leak"),
                description=f"Memory leak detected - {total_growth/(1024*1024):.1f}MB growth",
                impact_estimate=min(1.0, total_growth / (100 * 1024 * 1024)),
                evidence={
                    "total_growth_mb": total_growth / (1024 * 1024),
                    "growth_rate_mb_per_sample": trend_slope / (1024 * 1024),
                    "snapshots_analyzed": len(sorted_snapshots),
                    "pattern_category": PatternCategory.MEMORY.value,
                },
            )

        return None

    def _detect_memory_fragmentation(self, snapshots) -> Optional[DetectedPattern]:
        """Detect memory fragmentation patterns."""
        if len(snapshots) < 3:
            return None

        # Look for high peak memory vs average memory
        peak_memories = [s.peak_memory for s in snapshots]
        total_memories = [s.total_memory for s in snapshots]

        if peak_memories and total_memories:
            avg_peak = statistics.mean(peak_memories)
            avg_total = statistics.mean(total_memories)

            # If peak is much higher than average, suggests fragmentation
            fragmentation_ratio = avg_peak / avg_total if avg_total > 0 else 1

            if fragmentation_ratio > 2.0:  # Peak is 2x average
                return DetectedPattern(
                    pattern_type="memory_fragmentation",
                    severity=PatternSeverity.MEDIUM.value,
                    source_location=SourceLocation(
                        "memory_analysis", 1, "fragmentation"
                    ),
                    description=f"Memory fragmentation detected - peak memory {fragmentation_ratio:.1f}x average",
                    impact_estimate=min(0.6, (fragmentation_ratio - 1) / 3),
                    evidence={
                        "fragmentation_ratio": fragmentation_ratio,
                        "avg_peak_mb": avg_peak / (1024 * 1024),
                        "avg_total_mb": avg_total / (1024 * 1024),
                        "pattern_category": PatternCategory.MEMORY.value,
                    },
                )

        return None

    def _detect_excessive_allocations(
        self, session: ProfileSession
    ) -> Optional[DetectedPattern]:
        """Detect excessive memory allocation patterns."""

        # Look for memory allocation events
        alloc_events = [
            e
            for e in session.execution_events
            if e.event_type == EventType.MEMORY_ALLOC
        ]

        if len(alloc_events) > 1000:  # High allocation count
            # Group by location
            alloc_by_location = defaultdict(list)
            for event in alloc_events:
                location_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.line_number}"
                alloc_by_location[location_key].append(event)

            # Find hotspot locations
            for location, events in alloc_by_location.items():
                if len(events) > 100:  # Many allocations at same location
                    total_allocated = sum(
                        e.event_data.get("bytes_allocated", 0)
                        for e in events
                        if e.event_data
                    )

                    if total_allocated > 10 * 1024 * 1024:  # 10MB total
                        location_obj = events[0].frame_info.source_location

                        return DetectedPattern(
                            pattern_type="excessive_allocations",
                            severity=PatternSeverity.HIGH.value,
                            source_location=location_obj,
                            description=f"Excessive memory allocations - {len(events)} allocations totaling {total_allocated/(1024*1024):.1f}MB",
                            impact_estimate=min(
                                0.8, total_allocated / (50 * 1024 * 1024)
                            ),
                            evidence={
                                "allocation_count": len(events),
                                "total_allocated_mb": total_allocated / (1024 * 1024),
                                "avg_allocation_size": total_allocated / len(events),
                                "pattern_category": PatternCategory.MEMORY.value,
                            },
                        )

        return None

    def _detect_io_patterns(self, session: ProfileSession) -> List[DetectedPattern]:
        """Detect I/O-related performance patterns."""
        patterns = []

        # Find I/O events
        io_events = [
            e
            for e in session.execution_events
            if e.event_type in [EventType.IO_READ, EventType.IO_WRITE]
        ]

        if not io_events:
            return patterns

        # Detect I/O bottlenecks
        bottleneck_pattern = self._detect_io_bottlenecks(io_events)
        if bottleneck_pattern:
            patterns.append(bottleneck_pattern)

        # Detect small I/O operations
        small_io_pattern = self._detect_small_io_operations(io_events)
        if small_io_pattern:
            patterns.append(small_io_pattern)

        # Detect synchronous I/O in loops
        sync_io_pattern = self._detect_synchronous_io_in_loops(io_events)
        if sync_io_pattern:
            patterns.append(sync_io_pattern)

        return patterns

    def _detect_io_bottlenecks(
        self, io_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect I/O bottleneck patterns."""

        slow_io_events = [
            e
            for e in io_events
            if e.execution_time and e.execution_time > 100_000_000  # 100ms
        ]

        if len(slow_io_events) > 5:
            total_slow_time = sum(e.execution_time or 0 for e in slow_io_events)
            avg_slow_time = total_slow_time / len(slow_io_events)

            # Find the slowest location
            location_times = defaultdict(list)
            for event in slow_io_events:
                location_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.line_number}"
                location_times[location_key].append(event.execution_time)

            slowest_location = max(location_times.items(), key=lambda x: sum(x[1]))
            location_key, times = slowest_location

            # Get source location
            location_event = next(
                e
                for e in slow_io_events
                if f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.line_number}"
                == location_key
            )

            return DetectedPattern(
                pattern_type="io_bottleneck",
                severity=PatternSeverity.HIGH.value,
                source_location=location_event.frame_info.source_location,
                description=f"I/O bottleneck detected - {len(slow_io_events)} slow operations averaging {avg_slow_time/1_000_000:.1f}ms",
                impact_estimate=min(0.9, total_slow_time / 5_000_000_000),  # 5 seconds
                evidence={
                    "slow_operation_count": len(slow_io_events),
                    "avg_time_ms": avg_slow_time / 1_000_000,
                    "total_time_ms": total_slow_time / 1_000_000,
                    "pattern_category": PatternCategory.IO.value,
                },
            )

        return None

    def _detect_small_io_operations(
        self, io_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect inefficient small I/O operations."""

        small_io_events = []
        for event in io_events:
            if event.event_data and "bytes_transferred" in event.event_data:
                bytes_transferred = event.event_data["bytes_transferred"]
                if bytes_transferred < 1024:  # Less than 1KB
                    small_io_events.append(event)

        if len(small_io_events) > 50:  # Many small I/O operations
            total_operations = len(small_io_events)
            total_bytes = sum(
                e.event_data.get("bytes_transferred", 0)
                for e in small_io_events
                if e.event_data
            )

            # Find most common location
            location_counts = Counter(
                f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.line_number}"
                for e in small_io_events
            )

            most_common_location, count = location_counts.most_common(1)[0]
            location_event = next(
                e
                for e in small_io_events
                if f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.line_number}"
                == most_common_location
            )

            return DetectedPattern(
                pattern_type="small_io_operations",
                severity=PatternSeverity.MEDIUM.value,
                source_location=location_event.frame_info.source_location,
                description=f"Inefficient small I/O operations - {total_operations} operations averaging {total_bytes/total_operations:.0f} bytes",
                impact_estimate=min(0.6, total_operations / 200),
                evidence={
                    "small_io_count": total_operations,
                    "total_bytes": total_bytes,
                    "avg_bytes_per_operation": total_bytes / total_operations,
                    "pattern_category": PatternCategory.IO.value,
                },
            )

        return None

    def _detect_synchronous_io_in_loops(
        self, io_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect synchronous I/O operations in loops."""

        # Group I/O events by location and check for patterns
        location_events = defaultdict(list)
        for event in io_events:
            location_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.line_number}"
            location_events[location_key].append(event)

        for location, events in location_events.items():
            if len(events) > 10:  # Many I/O operations at same location
                # Check if operations are sequential (suggesting loop)
                timestamps = [e.timestamp for e in events]
                timestamps.sort()

                # Calculate time gaps between operations
                time_gaps = [
                    timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
                ]

                if time_gaps:
                    avg_gap = statistics.mean(time_gaps)
                    gap_std = statistics.stdev(time_gaps) if len(time_gaps) > 1 else 0

                    # If gaps are regular, likely in a loop
                    if gap_std < avg_gap * 0.5:  # Low variance suggests regularity
                        location_event = events[0]

                        return DetectedPattern(
                            pattern_type="synchronous_io_in_loop",
                            severity=PatternSeverity.HIGH.value,
                            source_location=location_event.frame_info.source_location,
                            description=f"Synchronous I/O in loop detected - {len(events)} operations with regular timing",
                            impact_estimate=min(0.8, len(events) / 50),
                            evidence={
                                "io_operation_count": len(events),
                                "avg_gap_between_operations": avg_gap,
                                "timing_regularity": 1
                                - (gap_std / avg_gap if avg_gap > 0 else 0),
                                "pattern_category": PatternCategory.IO.value,
                            },
                        )

        return None

    def _detect_cpu_patterns(self, session: ProfileSession) -> List[DetectedPattern]:
        """Detect CPU-related performance patterns."""
        patterns = []

        # Find CPU-intensive events
        cpu_events = [
            e
            for e in session.execution_events
            if e.event_data and e.event_data.get("operation") == "function_cpu"
        ]

        if not cpu_events:
            return patterns

        # Detect CPU hotspots
        hotspot_pattern = self._detect_cpu_hotspots(cpu_events)
        if hotspot_pattern:
            patterns.append(hotspot_pattern)

        return patterns

    def _detect_cpu_hotspots(
        self, cpu_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect CPU hotspot patterns."""

        # Group by function
        function_times = defaultdict(list)
        for event in cpu_events:
            func_key = f"{event.frame_info.source_location.filename}:{event.frame_info.source_location.function_name}"
            if event.execution_time:
                function_times[func_key].append(event.execution_time)

        # Find functions consuming most CPU time
        function_totals = {func: sum(times) for func, times in function_times.items()}

        if function_totals:
            total_cpu_time = sum(function_totals.values())
            max_func, max_time = max(function_totals.items(), key=lambda x: x[1])

            # If one function consumes >20% of CPU time
            cpu_percentage = max_time / total_cpu_time
            if cpu_percentage > 0.2:
                # Find representative event for location
                location_event = next(
                    e
                    for e in cpu_events
                    if f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.function_name}"
                    == max_func
                )

                return DetectedPattern(
                    pattern_type="cpu_hotspot",
                    severity=(
                        PatternSeverity.HIGH.value
                        if cpu_percentage > 0.5
                        else PatternSeverity.MEDIUM.value
                    ),
                    source_location=location_event.frame_info.source_location,
                    description=f"CPU hotspot detected - function consumes {cpu_percentage*100:.1f}% of CPU time",
                    impact_estimate=cpu_percentage,
                    evidence={
                        "cpu_percentage": cpu_percentage * 100,
                        "total_time_ms": max_time / 1_000_000,
                        "call_count": len(function_times[max_func]),
                        "pattern_category": PatternCategory.CPU.value,
                    },
                )

        return None

    def _detect_exception_patterns(
        self, session: ProfileSession
    ) -> List[DetectedPattern]:
        """Detect exception-related performance patterns."""
        patterns = []

        exception_events = [
            e for e in session.execution_events if e.event_type == EventType.EXCEPTION
        ]

        if not exception_events:
            return patterns

        # Detect exception hotspots
        hotspot_pattern = self._detect_exception_hotspots(exception_events)
        if hotspot_pattern:
            patterns.append(hotspot_pattern)

        return patterns

    def _detect_exception_hotspots(
        self, exception_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect exception hotspot patterns."""

        if len(exception_events) < 10:
            return None

        # Group by location
        location_counts = Counter(
            f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.line_number}"
            for e in exception_events
        )

        most_common_location, count = location_counts.most_common(1)[0]

        if count > 5:  # More than 5 exceptions at same location
            location_event = next(
                e
                for e in exception_events
                if f"{e.frame_info.source_location.filename}:{e.frame_info.source_location.line_number}"
                == most_common_location
            )

            return DetectedPattern(
                pattern_type="exception_hotspot",
                severity=PatternSeverity.HIGH.value,
                source_location=location_event.frame_info.source_location,
                description=f"Exception hotspot detected - {count} exceptions at this location",
                impact_estimate=min(0.8, count / 20),
                evidence={
                    "exception_count": count,
                    "total_exceptions": len(exception_events),
                    "pattern_category": PatternCategory.EXCEPTION.value,
                },
            )

        return None

    def _detect_import_patterns(self, session: ProfileSession) -> List[DetectedPattern]:
        """Detect import-related performance patterns."""
        patterns = []

        import_events = [
            e
            for e in session.execution_events
            if e.event_data and e.event_data.get("operation") == "import"
        ]

        if not import_events:
            return patterns

        # Detect slow imports
        slow_import_pattern = self._detect_slow_imports(import_events)
        if slow_import_pattern:
            patterns.append(slow_import_pattern)

        return patterns

    def _detect_slow_imports(
        self, import_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect slow import patterns."""

        slow_imports = [
            e
            for e in import_events
            if e.execution_time and e.execution_time > 500_000_000  # 500ms
        ]

        if slow_imports:
            slowest_import = max(slow_imports, key=lambda e: e.execution_time or 0)

            return DetectedPattern(
                pattern_type="slow_imports",
                severity=PatternSeverity.MEDIUM.value,
                source_location=slowest_import.frame_info.source_location,
                description=f"Slow imports detected - slowest import took {(slowest_import.execution_time or 0)/1_000_000:.0f}ms",
                impact_estimate=min(
                    0.6, (slowest_import.execution_time or 0) / 5_000_000_000
                ),
                evidence={
                    "slow_import_count": len(slow_imports),
                    "slowest_time_ms": (slowest_import.execution_time or 0) / 1_000_000,
                    "module_name": slowest_import.event_data.get(
                        "module_name", "unknown"
                    ),
                    "pattern_category": PatternCategory.IMPORT.value,
                },
            )

        return None

    def _detect_gc_patterns(self, session: ProfileSession) -> List[DetectedPattern]:
        """Detect garbage collection related patterns."""
        patterns = []

        gc_events = [
            e
            for e in session.execution_events
            if e.event_type in [EventType.GC_START, EventType.GC_END]
        ]

        if not gc_events:
            return patterns

        # Detect frequent GC
        frequent_gc_pattern = self._detect_frequent_gc(gc_events)
        if frequent_gc_pattern:
            patterns.append(frequent_gc_pattern)

        return patterns

    def _detect_frequent_gc(
        self, gc_events: List[ExecutionEvent]
    ) -> Optional[DetectedPattern]:
        """Detect frequent garbage collection patterns."""

        gc_starts = [e for e in gc_events if e.event_type == EventType.GC_START]

        if len(gc_starts) > 20:  # Frequent GC
            # Calculate GC frequency
            if len(gc_starts) >= 2:
                timestamps = [e.timestamp for e in gc_starts]
                timestamps.sort()
                time_span = timestamps[-1] - timestamps[0]

                if time_span > 0:
                    gc_frequency = len(gc_starts) / (
                        time_span / 1_000_000_000
                    )  # GCs per second

                    if gc_frequency > 2:  # More than 2 GCs per second
                        return DetectedPattern(
                            pattern_type="frequent_gc",
                            severity=PatternSeverity.MEDIUM.value,
                            source_location=SourceLocation(
                                "gc_analysis", 1, "frequent_gc"
                            ),
                            description=f"Frequent garbage collection detected - {gc_frequency:.1f} GCs per second",
                            impact_estimate=min(0.7, gc_frequency / 10),
                            evidence={
                                "gc_count": len(gc_starts),
                                "gc_frequency_per_second": gc_frequency,
                                "time_span_seconds": time_span / 1_000_000_000,
                                "pattern_category": PatternCategory.GC.value,
                            },
                        )

        return None

    def _detect_correlation_patterns(
        self, session: ProfileSession
    ) -> List[DetectedPattern]:
        """Detect cross-collector correlation patterns."""
        patterns = []

        # Detect memory allocation -> GC correlation
        memory_gc_pattern = self._detect_memory_gc_correlation(session)
        if memory_gc_pattern:
            patterns.append(memory_gc_pattern)

        # Detect I/O -> CPU correlation (I/O blocking CPU)
        io_cpu_pattern = self._detect_io_cpu_correlation(session)
        if io_cpu_pattern:
            patterns.append(io_cpu_pattern)

        return patterns

    def _detect_memory_gc_correlation(
        self, session: ProfileSession
    ) -> Optional[DetectedPattern]:
        """Detect correlation between memory allocation and GC frequency."""

        alloc_events = [
            e
            for e in session.execution_events
            if e.event_type == EventType.MEMORY_ALLOC
        ]

        gc_events = [
            e for e in session.execution_events if e.event_type == EventType.GC_START
        ]

        if len(alloc_events) > 50 and len(gc_events) > 5:
            # Simple correlation: if many allocations lead to frequent GC
            alloc_rate = len(alloc_events) / len(gc_events)

            if alloc_rate < 20:  # Less than 20 allocations per GC suggests frequent GC
                return DetectedPattern(
                    pattern_type="memory_pressure",
                    severity=PatternSeverity.MEDIUM.value,
                    source_location=SourceLocation(
                        "correlation_analysis", 1, "memory_pressure"
                    ),
                    description=f"Memory pressure detected - frequent GC with low allocation rate",
                    impact_estimate=min(0.6, 1 - (alloc_rate / 50)),
                    evidence={
                        "allocation_count": len(alloc_events),
                        "gc_count": len(gc_events),
                        "allocations_per_gc": alloc_rate,
                        "pattern_category": PatternCategory.MEMORY.value,
                    },
                )

        return None

    def _detect_io_cpu_correlation(
        self, session: ProfileSession
    ) -> Optional[DetectedPattern]:
        """Detect correlation between I/O operations and CPU usage."""

        io_events = [
            e
            for e in session.execution_events
            if e.event_type in [EventType.IO_READ, EventType.IO_WRITE]
        ]

        cpu_events = [
            e
            for e in session.execution_events
            if e.event_data and e.event_data.get("operation") == "function_cpu"
        ]

        if len(io_events) > 10 and len(cpu_events) > 10:
            # Look for I/O events that might be blocking CPU
            total_io_time = sum(e.execution_time or 0 for e in io_events)
            total_cpu_time = sum(e.execution_time or 0 for e in cpu_events)

            if total_io_time > 0 and total_cpu_time > 0:
                io_ratio = total_io_time / (total_io_time + total_cpu_time)

                if io_ratio > 0.3:  # I/O takes up >30% of time
                    return DetectedPattern(
                        pattern_type="io_cpu_blocking",
                        severity=PatternSeverity.MEDIUM.value,
                        source_location=SourceLocation(
                            "correlation_analysis", 1, "io_blocking"
                        ),
                        description=f"I/O blocking CPU detected - I/O operations consume {io_ratio*100:.1f}% of time",
                        impact_estimate=io_ratio,
                        evidence={
                            "io_time_ms": total_io_time / 1_000_000,
                            "cpu_time_ms": total_cpu_time / 1_000_000,
                            "io_percentage": io_ratio * 100,
                            "pattern_category": PatternCategory.CONCURRENCY.value,
                        },
                    )

        return None

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate the slope of a trend in values."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_values = list(range(n))

        # Simple linear regression
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_pattern_confidence(self, pattern: DetectedPattern) -> float:
        """Calculate confidence score for a detected pattern."""

        # Base confidence from evidence
        evidence = pattern.evidence
        base_confidence = 0.5

        # Adjust based on evidence quality
        if "statistical_significance" in evidence:
            base_confidence += evidence["statistical_significance"] * 0.3

        if "sample_size" in evidence:
            sample_size = evidence["sample_size"]
            # More samples = higher confidence
            sample_confidence = min(0.3, sample_size / 100)
            base_confidence += sample_confidence

        # Adjust based on pattern type
        pattern_confidence_modifiers = {
            "memory_leak": 0.9,
            "cpu_hotspot": 0.8,
            "io_bottleneck": 0.85,
            "nested_loops": 0.7,
            "exception_hotspot": 0.9,
        }

        modifier = pattern_confidence_modifiers.get(pattern.pattern_type, 0.7)
        final_confidence = base_confidence * modifier

        return min(1.0, final_confidence)
