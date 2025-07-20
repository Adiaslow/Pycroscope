"""
Cross-correlation analyzer for Pycroscope.

Analyzes relationships between data from multiple collectors to identify
sophisticated performance patterns that aren't visible from single collectors.
"""

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from ..core.models import (
    AnalysisResult,
    DetectedPattern,
    DynamicAnalysisResult,
    EventType,
    OptimizationRecommendation,
    ProfileSession,
    SourceLocation,
    StaticAnalysisResult,
)
from .base_analyzer import BaseAnalyzer


@dataclass
class CorrelationInsight:
    """Represents a correlation between different metrics."""

    metric_a: str
    metric_b: str
    correlation_coefficient: float
    strength: str  # 'weak', 'moderate', 'strong'
    pattern_type: str
    evidence: Dict[str, Any]


@dataclass
class CausalRelationship:
    """Represents a potential causal relationship between events."""

    cause_events: List[str]
    effect_events: List[str]
    time_lag_ms: float
    confidence: float
    description: str


class CrossCorrelationAnalyzer(BaseAnalyzer):
    """
    Cross-correlation analysis engine.

    Analyzes relationships between data from multiple collectors to identify
    complex performance patterns and causal relationships.
    """

    def __init__(self, config=None):
        """
        Initialize the correlation analyzer.

        Args:
            config: Optional analysis configuration
        """
        super().__init__(config)
        self.correlation_threshold = 0.5  # Minimum correlation to report
        self.causality_window_ms = 1000  # Time window for causality analysis

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        return "correlation"

    @property
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        return ["line", "memory", "call", "io", "cpu", "gc", "exception", "import"]

    def _perform_analysis(self, profile_data: ProfileSession) -> AnalysisResult:
        """
        Perform cross-correlation analysis.

        Args:
            profile_data: Profiling session data

        Returns:
            Analysis results with correlation insights
        """
        patterns = []
        recommendations = []

        # Group events by collector type
        events_by_collector = self._group_events_by_collector(
            profile_data.execution_events
        )

        # Perform cross-collector correlation analysis
        correlation_patterns = self._analyze_cross_collector_correlations(
            events_by_collector
        )
        patterns.extend(correlation_patterns)

        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(
            profile_data.execution_events
        )
        patterns.extend(temporal_patterns)

        # Detect causal relationships
        causal_patterns = self._detect_causal_relationships(events_by_collector)
        patterns.extend(causal_patterns)

        # Analyze resource contention patterns
        contention_patterns = self._analyze_resource_contention(events_by_collector)
        patterns.extend(contention_patterns)

        # Generate correlation-based recommendations
        correlation_recommendations = self._generate_correlation_recommendations(
            patterns
        )
        recommendations.extend(correlation_recommendations)

        # Calculate overall correlation score
        overall_score = self._calculate_correlation_score(patterns)

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

    def _group_events_by_collector(self, events) -> Dict[str, List]:
        """Group events by their originating collector."""
        grouped = defaultdict(list)

        for event in events:
            # Determine collector based on event type and data
            collector_type = self._determine_collector_type(event)
            grouped[collector_type].append(event)

        return dict(grouped)

    def _determine_collector_type(self, event) -> str:
        """Determine which collector produced this event."""

        if event.event_type == EventType.EXCEPTION:
            return "exception"
        elif event.event_type in [EventType.IO_READ, EventType.IO_WRITE]:
            return "io"
        elif event.event_type in [EventType.GC_START, EventType.GC_END]:
            return "gc"
        elif event.event_type in [EventType.MEMORY_ALLOC, EventType.MEMORY_DEALLOC]:
            return "memory"
        elif event.event_type == EventType.CALL:
            # Distinguish between call collector and CPU collector
            if event.event_data and event.event_data.get("operation") == "function_cpu":
                return "cpu"
            elif event.event_data and event.event_data.get("operation") == "import":
                return "import"
            else:
                return "call"
        elif event.event_type == EventType.LINE:
            return "line"
        else:
            return "unknown"

    def _analyze_cross_collector_correlations(
        self, events_by_collector: Dict[str, List]
    ) -> List[DetectedPattern]:
        """Analyze correlations between different collector metrics."""
        patterns = []

        collectors = list(events_by_collector.keys())

        # Analyze pairwise correlations
        for i, collector_a in enumerate(collectors):
            for collector_b in collectors[i + 1 :]:
                correlation_pattern = self._analyze_collector_pair_correlation(
                    collector_a,
                    events_by_collector[collector_a],
                    collector_b,
                    events_by_collector[collector_b],
                )
                if correlation_pattern:
                    patterns.append(correlation_pattern)

        return patterns

    def _analyze_collector_pair_correlation(
        self, collector_a: str, events_a: List, collector_b: str, events_b: List
    ) -> Optional[DetectedPattern]:
        """Analyze correlation between two collectors."""

        if len(events_a) < 10 or len(events_b) < 10:
            return None

        # Memory-GC correlation
        if (collector_a == "memory" and collector_b == "gc") or (
            collector_a == "gc" and collector_b == "memory"
        ):
            return self._analyze_memory_gc_correlation(events_a, events_b)

        # I/O-CPU correlation
        elif (collector_a == "io" and collector_b == "cpu") or (
            collector_a == "cpu" and collector_b == "io"
        ):
            return self._analyze_io_cpu_correlation(events_a, events_b)

        # Exception-Memory correlation
        elif (collector_a == "exception" and collector_b == "memory") or (
            collector_a == "memory" and collector_b == "exception"
        ):
            return self._analyze_exception_memory_correlation(events_a, events_b)

        # Import-CPU correlation
        elif (collector_a == "import" and collector_b == "cpu") or (
            collector_a == "cpu" and collector_b == "import"
        ):
            return self._analyze_import_cpu_correlation(events_a, events_b)

        return None

    def _analyze_memory_gc_correlation(
        self, memory_events: List, gc_events: List
    ) -> Optional[DetectedPattern]:
        """Analyze correlation between memory allocations and GC frequency."""

        if not memory_events or not gc_events:
            return None

        # Calculate allocation rate over time
        alloc_events = [
            e for e in memory_events if e.event_type == EventType.MEMORY_ALLOC
        ]

        if len(alloc_events) < 5:
            return None

        # Group events by time windows
        time_windows = self._create_time_windows(
            alloc_events + gc_events, window_size_ms=1000
        )

        correlations = []
        for window_start, window_events in time_windows.items():
            window_allocs = [
                e for e in window_events if e.event_type == EventType.MEMORY_ALLOC
            ]
            window_gcs = [
                e
                for e in window_events
                if e.event_type in [EventType.GC_START, EventType.GC_END]
            ]

            if window_allocs and window_gcs:
                # Calculate allocation size in window
                total_allocated = sum(
                    e.event_data.get("bytes_allocated", 0)
                    for e in window_allocs
                    if e.event_data
                )
                gc_count = len(
                    [e for e in window_gcs if e.event_type == EventType.GC_START]
                )

                correlations.append((total_allocated, gc_count))

        if len(correlations) < 3:
            return None

        # Calculate correlation coefficient
        alloc_sizes = [c[0] for c in correlations]
        gc_counts = [c[1] for c in correlations]

        correlation_coeff = self._calculate_correlation(alloc_sizes, gc_counts)

        if abs(correlation_coeff) > self.correlation_threshold:
            strength = self._categorize_correlation_strength(correlation_coeff)

            return DetectedPattern(
                pattern_type="memory_gc_correlation",
                severity="medium" if abs(correlation_coeff) > 0.7 else "low",
                source_location=SourceLocation("correlation_analysis", 1, "memory_gc"),
                description=f"{strength} correlation between memory allocations and GC frequency (r={correlation_coeff:.2f})",
                impact_estimate=abs(correlation_coeff) * 0.6,
                evidence={
                    "correlation_coefficient": correlation_coeff,
                    "correlation_strength": strength,
                    "time_windows_analyzed": len(correlations),
                    "avg_allocations_per_window": statistics.mean(alloc_sizes),
                    "avg_gc_per_window": statistics.mean(gc_counts),
                },
            )

        return None

    def _analyze_io_cpu_correlation(
        self, io_events: List, cpu_events: List
    ) -> Optional[DetectedPattern]:
        """Analyze correlation between I/O operations and CPU usage."""

        if not io_events or not cpu_events:
            return None

        # Create time windows
        time_windows = self._create_time_windows(
            io_events + cpu_events, window_size_ms=500
        )

        correlations = []
        for window_start, window_events in time_windows.items():
            window_io = [
                e
                for e in window_events
                if e.event_type in [EventType.IO_READ, EventType.IO_WRITE]
            ]
            window_cpu = [
                e
                for e in window_events
                if e.event_data and e.event_data.get("operation") == "function_cpu"
            ]

            if window_io and window_cpu:
                # Calculate I/O wait time and CPU usage
                io_wait_time = sum(e.execution_time or 0 for e in window_io)
                cpu_time = sum(e.execution_time or 0 for e in window_cpu)

                correlations.append((io_wait_time, cpu_time))

        if len(correlations) < 3:
            return None

        io_times = [c[0] for c in correlations]
        cpu_times = [c[1] for c in correlations]

        correlation_coeff = self._calculate_correlation(io_times, cpu_times)

        # Look for negative correlation (high I/O, low CPU suggests blocking)
        if correlation_coeff < -self.correlation_threshold:
            return DetectedPattern(
                pattern_type="io_cpu_blocking",
                severity="high",
                source_location=SourceLocation(
                    "correlation_analysis", 1, "io_cpu_blocking"
                ),
                description=f"I/O operations blocking CPU utilization (negative correlation r={correlation_coeff:.2f})",
                impact_estimate=abs(correlation_coeff) * 0.8,
                evidence={
                    "correlation_coefficient": correlation_coeff,
                    "avg_io_wait_time_ms": statistics.mean(io_times) / 1_000_000,
                    "avg_cpu_time_ms": statistics.mean(cpu_times) / 1_000_000,
                    "blocking_evidence": correlation_coeff < -0.6,
                },
            )

        return None

    def _analyze_exception_memory_correlation(
        self, exception_events: List, memory_events: List
    ) -> Optional[DetectedPattern]:
        """Analyze correlation between exceptions and memory patterns."""

        if not exception_events or not memory_events:
            return None

        # Look for memory spikes around exception times
        memory_spikes = []
        exception_times = [e.timestamp for e in exception_events]

        for memory_event in memory_events:
            if memory_event.event_data and "bytes_allocated" in memory_event.event_data:
                bytes_allocated = memory_event.event_data["bytes_allocated"]

                # Find nearest exception
                nearest_exception_time = min(
                    exception_times, key=lambda t: abs(t - memory_event.timestamp)
                )

                time_diff = abs(memory_event.timestamp - nearest_exception_time)

                # If memory allocation happened close to exception (within 100ms)
                if time_diff < 100_000_000:  # 100ms in nanoseconds
                    memory_spikes.append((bytes_allocated, time_diff))

        if len(memory_spikes) > 5:  # Significant pattern
            avg_spike_size = statistics.mean([spike[0] for spike in memory_spikes])
            avg_time_diff = statistics.mean([spike[1] for spike in memory_spikes])

            return DetectedPattern(
                pattern_type="exception_memory_correlation",
                severity="medium",
                source_location=SourceLocation(
                    "correlation_analysis", 1, "exception_memory"
                ),
                description=f"Memory allocations correlated with exceptions - {len(memory_spikes)} allocations near exception times",
                impact_estimate=min(0.6, len(memory_spikes) / 20),
                evidence={
                    "correlated_allocations": len(memory_spikes),
                    "avg_allocation_size": avg_spike_size,
                    "avg_time_difference_ms": avg_time_diff / 1_000_000,
                    "total_exceptions": len(exception_events),
                },
            )

        return None

    def _analyze_import_cpu_correlation(
        self, import_events: List, cpu_events: List
    ) -> Optional[DetectedPattern]:
        """Analyze correlation between import operations and CPU usage."""

        if not import_events or not cpu_events:
            return None

        # Find CPU spikes during import operations
        cpu_during_imports = []

        for import_event in import_events:
            import_start = import_event.timestamp
            import_duration = import_event.execution_time or 0
            import_end = import_start + import_duration

            # Find CPU events during this import
            concurrent_cpu = [
                e for e in cpu_events if import_start <= e.timestamp <= import_end
            ]

            if concurrent_cpu:
                total_cpu_time = sum(e.execution_time or 0 for e in concurrent_cpu)
                cpu_during_imports.append((import_duration, total_cpu_time))

        if len(cpu_during_imports) > 3:
            import_times = [c[0] for c in cpu_during_imports]
            cpu_times = [c[1] for c in cpu_during_imports]

            correlation_coeff = self._calculate_correlation(import_times, cpu_times)

            if correlation_coeff > self.correlation_threshold:
                return DetectedPattern(
                    pattern_type="import_cpu_correlation",
                    severity="medium",
                    source_location=SourceLocation(
                        "correlation_analysis", 1, "import_cpu"
                    ),
                    description=f"CPU usage correlated with import operations (r={correlation_coeff:.2f})",
                    impact_estimate=correlation_coeff * 0.5,
                    evidence={
                        "correlation_coefficient": correlation_coeff,
                        "import_cpu_pairs": len(cpu_during_imports),
                        "avg_import_time_ms": statistics.mean(import_times) / 1_000_000,
                        "avg_cpu_time_ms": statistics.mean(cpu_times) / 1_000_000,
                    },
                )

        return None

    def _analyze_temporal_patterns(self, events) -> List[DetectedPattern]:
        """Analyze temporal patterns across all events."""
        patterns = []

        if len(events) < 50:
            return patterns

        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        # Detect performance degradation over time
        degradation_pattern = self._detect_performance_degradation(sorted_events)
        if degradation_pattern:
            patterns.append(degradation_pattern)

        # Detect periodic patterns
        periodic_pattern = self._detect_periodic_performance_issues(sorted_events)
        if periodic_pattern:
            patterns.append(periodic_pattern)

        return patterns

    def _detect_performance_degradation(
        self, sorted_events
    ) -> Optional[DetectedPattern]:
        """Detect performance degradation over time."""

        # Group events into time chunks and analyze performance trends
        chunk_size = len(sorted_events) // 5  # 5 chunks
        if chunk_size < 10:
            return None

        chunk_performance = []
        for i in range(0, len(sorted_events), chunk_size):
            chunk = sorted_events[i : i + chunk_size]
            avg_execution_time = statistics.mean([e.execution_time or 0 for e in chunk])
            chunk_performance.append(avg_execution_time)

        if len(chunk_performance) < 3:
            return None

        # Calculate trend
        trend_slope = self._calculate_trend_slope(chunk_performance)

        # If performance is degrading significantly
        if trend_slope > 1_000_000:  # 1ms increase per chunk
            first_chunk_avg = chunk_performance[0]
            last_chunk_avg = chunk_performance[-1]
            degradation_factor = (
                last_chunk_avg / first_chunk_avg if first_chunk_avg > 0 else 1
            )

            return DetectedPattern(
                pattern_type="performance_degradation",
                severity="high" if degradation_factor > 2 else "medium",
                source_location=SourceLocation("temporal_analysis", 1, "degradation"),
                description=f"Performance degradation over time - {degradation_factor:.1f}x slower at end",
                impact_estimate=min(0.9, (degradation_factor - 1) / 3),
                evidence={
                    "degradation_factor": degradation_factor,
                    "trend_slope": trend_slope,
                    "first_chunk_avg_ms": first_chunk_avg / 1_000_000,
                    "last_chunk_avg_ms": last_chunk_avg / 1_000_000,
                    "chunks_analyzed": len(chunk_performance),
                },
            )

        return None

    def _detect_periodic_performance_issues(
        self, sorted_events
    ) -> Optional[DetectedPattern]:
        """Detect periodic performance issues."""

        # Look for events with unusually high execution times
        slow_events = [
            e
            for e in sorted_events
            if e.execution_time and e.execution_time > 10_000_000  # 10ms
        ]

        if len(slow_events) < 5:
            return None

        # Analyze time intervals between slow events
        slow_timestamps = [e.timestamp for e in slow_events]
        intervals = [
            slow_timestamps[i] - slow_timestamps[i - 1]
            for i in range(1, len(slow_timestamps))
        ]

        if len(intervals) < 3:
            return None

        # Check for periodic pattern (low variance in intervals)
        avg_interval = statistics.mean(intervals)
        interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0

        # If intervals are relatively consistent, might be periodic
        if (
            interval_std < avg_interval * 0.3 and avg_interval > 100_000_000
        ):  # 100ms intervals
            return DetectedPattern(
                pattern_type="periodic_performance_issues",
                severity="medium",
                source_location=SourceLocation("temporal_analysis", 1, "periodic"),
                description=f"Periodic performance issues detected - slow events every {avg_interval/1_000_000:.0f}ms",
                impact_estimate=min(0.6, len(slow_events) / 20),
                evidence={
                    "slow_event_count": len(slow_events),
                    "avg_interval_ms": avg_interval / 1_000_000,
                    "interval_consistency": 1
                    - (interval_std / avg_interval if avg_interval > 0 else 0),
                    "avg_slow_event_time_ms": statistics.mean(
                        [e.execution_time for e in slow_events]
                    )
                    / 1_000_000,
                },
            )

        return None

    def _detect_causal_relationships(
        self, events_by_collector: Dict[str, List]
    ) -> List[DetectedPattern]:
        """Detect causal relationships between different types of events."""
        patterns = []

        # Memory allocation -> GC causality
        if "memory" in events_by_collector and "gc" in events_by_collector:
            gc_causality = self._detect_memory_gc_causality(
                events_by_collector["memory"], events_by_collector["gc"]
            )
            if gc_causality:
                patterns.append(gc_causality)

        # Exception -> Memory allocation causality
        if "exception" in events_by_collector and "memory" in events_by_collector:
            exception_memory_causality = self._detect_exception_memory_causality(
                events_by_collector["exception"], events_by_collector["memory"]
            )
            if exception_memory_causality:
                patterns.append(exception_memory_causality)

        return patterns

    def _detect_memory_gc_causality(
        self, memory_events: List, gc_events: List
    ) -> Optional[DetectedPattern]:
        """Detect if memory allocations cause GC events."""

        alloc_events = [
            e for e in memory_events if e.event_type == EventType.MEMORY_ALLOC
        ]
        gc_start_events = [e for e in gc_events if e.event_type == EventType.GC_START]

        if len(alloc_events) < 10 or len(gc_start_events) < 3:
            return None

        # Find GC events that happen shortly after large allocations
        causal_pairs = []

        for gc_event in gc_start_events:
            # Look for memory allocations in the preceding time window
            recent_allocs = [
                e
                for e in alloc_events
                if gc_event.timestamp - self.causality_window_ms * 1_000_000
                <= e.timestamp
                < gc_event.timestamp
            ]

            if recent_allocs:
                total_allocated = sum(
                    e.event_data.get("bytes_allocated", 0)
                    for e in recent_allocs
                    if e.event_data
                )

                if total_allocated > 1024 * 1024:  # 1MB threshold
                    time_to_gc = gc_event.timestamp - max(
                        e.timestamp for e in recent_allocs
                    )
                    causal_pairs.append((total_allocated, time_to_gc))

        if len(causal_pairs) >= 3:  # Need multiple instances for confidence
            avg_allocation = statistics.mean([pair[0] for pair in causal_pairs])
            avg_time_to_gc = statistics.mean([pair[1] for pair in causal_pairs])

            return DetectedPattern(
                pattern_type="memory_gc_causality",
                severity="medium",
                source_location=SourceLocation("causality_analysis", 1, "memory_gc"),
                description=f"Memory allocations causing GC - {len(causal_pairs)} instances found",
                impact_estimate=min(0.7, len(causal_pairs) / 10),
                evidence={
                    "causal_instances": len(causal_pairs),
                    "avg_allocation_before_gc_mb": avg_allocation / (1024 * 1024),
                    "avg_time_to_gc_ms": avg_time_to_gc / 1_000_000,
                    "causality_confidence": min(1.0, len(causal_pairs) / 5),
                },
            )

        return None

    def _detect_exception_memory_causality(
        self, exception_events: List, memory_events: List
    ) -> Optional[DetectedPattern]:
        """Detect if exceptions cause memory allocations (e.g., for cleanup)."""

        if len(exception_events) < 5 or len(memory_events) < 10:
            return None

        causal_pairs = []

        for exception_event in exception_events:
            # Look for memory allocations shortly after exceptions
            subsequent_allocs = [
                e
                for e in memory_events
                if exception_event.timestamp
                < e.timestamp
                <= exception_event.timestamp + self.causality_window_ms * 1_000_000
                and e.event_type == EventType.MEMORY_ALLOC
            ]

            if subsequent_allocs:
                total_allocated = sum(
                    e.event_data.get("bytes_allocated", 0)
                    for e in subsequent_allocs
                    if e.event_data
                )

                if total_allocated > 10000:  # 10KB threshold
                    time_to_alloc = (
                        min(e.timestamp for e in subsequent_allocs)
                        - exception_event.timestamp
                    )
                    causal_pairs.append((total_allocated, time_to_alloc))

        if len(causal_pairs) >= 3:
            avg_allocation = statistics.mean([pair[0] for pair in causal_pairs])
            avg_time_to_alloc = statistics.mean([pair[1] for pair in causal_pairs])

            return DetectedPattern(
                pattern_type="exception_memory_causality",
                severity="low",
                source_location=SourceLocation(
                    "causality_analysis", 1, "exception_memory"
                ),
                description=f"Exceptions triggering memory allocations - {len(causal_pairs)} instances",
                impact_estimate=min(0.4, len(causal_pairs) / 10),
                evidence={
                    "causal_instances": len(causal_pairs),
                    "avg_allocation_after_exception_kb": avg_allocation / 1024,
                    "avg_time_to_allocation_ms": avg_time_to_alloc / 1_000_000,
                },
            )

        return None

    def _analyze_resource_contention(
        self, events_by_collector: Dict[str, List]
    ) -> List[DetectedPattern]:
        """Analyze resource contention patterns."""
        patterns = []

        # CPU contention analysis
        if "cpu" in events_by_collector:
            cpu_contention = self._detect_cpu_contention(events_by_collector["cpu"])
            if cpu_contention:
                patterns.append(cpu_contention)

        # I/O contention analysis
        if "io" in events_by_collector:
            io_contention = self._detect_io_contention(events_by_collector["io"])
            if io_contention:
                patterns.append(io_contention)

        return patterns

    def _detect_cpu_contention(self, cpu_events: List) -> Optional[DetectedPattern]:
        """Detect CPU contention patterns."""

        if len(cpu_events) < 20:
            return None

        # Group events by short time windows to detect concurrent CPU usage
        time_windows = self._create_time_windows(cpu_events, window_size_ms=100)

        high_contention_windows = []
        for window_start, window_events in time_windows.items():
            if len(window_events) > 5:  # Multiple CPU events in short window
                total_cpu_time = sum(e.execution_time or 0 for e in window_events)
                high_contention_windows.append((len(window_events), total_cpu_time))

        if len(high_contention_windows) > 3:
            avg_concurrent_events = statistics.mean(
                [w[0] for w in high_contention_windows]
            )
            avg_cpu_time = statistics.mean([w[1] for w in high_contention_windows])

            return DetectedPattern(
                pattern_type="cpu_contention",
                severity="medium",
                source_location=SourceLocation(
                    "contention_analysis", 1, "cpu_contention"
                ),
                description=f"CPU contention detected - {avg_concurrent_events:.1f} concurrent operations on average",
                impact_estimate=min(0.6, avg_concurrent_events / 10),
                evidence={
                    "high_contention_windows": len(high_contention_windows),
                    "avg_concurrent_operations": avg_concurrent_events,
                    "avg_cpu_time_ms": avg_cpu_time / 1_000_000,
                },
            )

        return None

    def _detect_io_contention(self, io_events: List) -> Optional[DetectedPattern]:
        """Detect I/O contention patterns."""

        if len(io_events) < 10:
            return None

        # Group I/O events by file/resource
        io_by_resource = defaultdict(list)
        for event in io_events:
            if event.event_data and "file_path" in event.event_data:
                resource = event.event_data["file_path"]
                io_by_resource[resource].append(event)

        contended_resources = []
        for resource, resource_events in io_by_resource.items():
            if len(resource_events) > 10:  # Frequently accessed resource
                # Check for overlapping I/O operations
                overlapping_count = 0
                sorted_events = sorted(resource_events, key=lambda e: e.timestamp)

                for i, event in enumerate(sorted_events[:-1]):
                    next_event = sorted_events[i + 1]
                    if (
                        event.execution_time
                        and next_event.timestamp
                        < event.timestamp + event.execution_time
                    ):
                        overlapping_count += 1

                if overlapping_count > 3:  # Multiple overlapping operations
                    contended_resources.append(
                        (resource, overlapping_count, len(resource_events))
                    )

        if contended_resources:
            most_contended = max(contended_resources, key=lambda x: x[1])
            resource, overlap_count, total_operations = most_contended

            return DetectedPattern(
                pattern_type="io_contention",
                severity="medium",
                source_location=SourceLocation(
                    "contention_analysis", 1, "io_contention"
                ),
                description=f"I/O contention on {resource} - {overlap_count} overlapping operations",
                impact_estimate=min(0.7, overlap_count / 10),
                evidence={
                    "contended_resource": resource,
                    "overlapping_operations": overlap_count,
                    "total_operations": total_operations,
                    "contention_ratio": overlap_count / total_operations,
                },
            )

        return None

    def _generate_correlation_recommendations(
        self, patterns: List[DetectedPattern]
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations based on correlation patterns."""
        recommendations = []

        for pattern in patterns:
            if pattern.pattern_type == "memory_gc_correlation":
                recommendations.append(
                    OptimizationRecommendation(
                        category="memory",
                        title="Optimize Memory Allocation Pattern",
                        description="Strong correlation between memory allocations and GC frequency suggests memory pressure",
                        estimated_improvement=pattern.impact_estimate,
                        confidence=0.8,
                        implementation_effort="medium",
                        suggested_actions=[
                            "Implement object pooling for frequently allocated objects",
                            "Use memory-efficient data structures",
                            "Consider increasing GC thresholds if appropriate",
                        ],
                        code_examples=[
                            "# Use object pooling\npool = ObjectPool()\nobj = pool.get_object()  # Instead of creating new"
                        ],
                    )
                )

            elif pattern.pattern_type == "io_cpu_blocking":
                recommendations.append(
                    OptimizationRecommendation(
                        category="concurrency",
                        title="Implement Asynchronous I/O",
                        description="I/O operations are blocking CPU utilization",
                        estimated_improvement=pattern.impact_estimate,
                        confidence=0.9,
                        implementation_effort="high",
                        suggested_actions=[
                            "Convert to asynchronous I/O operations",
                            "Use threading for I/O-bound operations",
                            "Implement I/O batching where possible",
                        ],
                        code_examples=[
                            "# Use async I/O\nasync def read_file():\n    async with aiofiles.open(file) as f:\n        return await f.read()"
                        ],
                    )
                )

            elif pattern.pattern_type == "cpu_contention":
                recommendations.append(
                    OptimizationRecommendation(
                        category="concurrency",
                        title="Reduce CPU Contention",
                        description="Multiple operations competing for CPU resources",
                        estimated_improvement=pattern.impact_estimate,
                        confidence=0.7,
                        implementation_effort="medium",
                        suggested_actions=[
                            "Implement workload scheduling",
                            "Use process pools for CPU-intensive tasks",
                            "Consider reducing parallelism in compute-heavy sections",
                        ],
                        code_examples=[
                            "# Use process pool for CPU-intensive tasks\nwith ProcessPoolExecutor() as executor:\n    results = executor.map(cpu_intensive_func, data)"
                        ],
                    )
                )

        return recommendations

    def _create_time_windows(self, events, window_size_ms: int) -> Dict[int, List]:
        """Create time windows for temporal analysis."""
        windows = defaultdict(list)

        for event in events:
            window_key = event.timestamp // (window_size_ms * 1_000_000)
            windows[window_key].append(event)

        return dict(windows)

    def _calculate_correlation(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))

        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        y_variance = sum((y - y_mean) ** 2 for y in y_values)

        denominator = math.sqrt(x_variance * y_variance)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _categorize_correlation_strength(self, correlation: float) -> str:
        """Categorize correlation strength."""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        else:
            return "weak"

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_values = list(range(n))

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_correlation_score(self, patterns: List[DetectedPattern]) -> float:
        """Calculate overall correlation analysis score."""
        if not patterns:
            return 0.5  # Neutral score

        # Score based on pattern severity and impact
        severity_weights = {"low": 0.3, "medium": 0.6, "high": 0.9, "critical": 1.0}

        total_weighted_impact = sum(
            pattern.impact_estimate * severity_weights.get(pattern.severity, 0.5)
            for pattern in patterns
        )

        # Normalize by number of patterns
        avg_impact = total_weighted_impact / len(patterns)

        # Invert score (lower correlation issues = higher score)
        return max(0.0, 1.0 - avg_impact)
