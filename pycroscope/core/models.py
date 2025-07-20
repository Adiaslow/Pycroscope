"""
Core data models for the Pycroscope profiling system.

Defines immutable data structures that represent profiling sessions,
execution events, analysis results, and optimization recommendations.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


class EventType(Enum):
    """Types of profiling events that can be collected."""

    CALL = "call"
    RETURN = "return"
    LINE = "line"
    EXCEPTION = "exception"
    MEMORY_ALLOC = "memory_alloc"
    MEMORY_DEALLOC = "memory_dealloc"
    GC_START = "gc_start"
    GC_END = "gc_end"
    IO_READ = "io_read"
    IO_WRITE = "io_write"


class MetricType(Enum):
    """Types of performance metrics that can be measured."""

    TIME = "time"
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    CALLS = "calls"
    COMPLEXITY = "complexity"


@dataclass(frozen=True)
class SourceLocation:
    """Immutable representation of a source code location."""

    filename: str
    line_number: int
    function_name: str
    module_name: Optional[str] = None

    def __str__(self) -> str:
        if self.module_name:
            return f"{self.module_name}.{self.function_name}:{self.filename}:{self.line_number}"
        return f"{self.function_name}:{self.filename}:{self.line_number}"


@dataclass(frozen=True)
class FrameInfo:
    """Immutable representation of a stack frame."""

    source_location: SourceLocation
    local_variables: Dict[str, Any] = field(default_factory=dict)
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class ExecutionEvent:
    """
    Immutable representation of a single profiling event.

    Contains complete context information for the event including
    timing, memory, and execution context.
    """

    timestamp: int  # nanoseconds since epoch
    event_type: EventType
    thread_id: int
    frame_info: FrameInfo

    # Performance metrics (optional based on event type)
    execution_time: Optional[int] = None  # nanoseconds
    memory_delta: Optional[int] = None  # bytes
    cpu_usage: Optional[float] = None  # percentage

    # Additional context
    call_stack: List[str] = field(default_factory=list)
    event_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Execution time in milliseconds."""
        return self.execution_time / 1_000_000 if self.execution_time else None


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable representation of memory state at a point in time."""

    timestamp: int
    total_memory: int  # total allocated bytes
    peak_memory: int  # peak memory during measurement
    gc_collections: int  # number of GC cycles
    object_counts: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CallNode:
    """Immutable node in a call tree structure."""

    source_location: SourceLocation
    total_time: int  # nanoseconds
    self_time: int  # nanoseconds (excluding children)
    call_count: int
    memory_allocated: int  # bytes
    children: List["CallNode"] = field(default_factory=list)

    @property
    def children_time(self) -> int:
        """Total time spent in child calls."""
        return self.total_time - self.self_time


@dataclass(frozen=True)
class CallTree:
    """Immutable representation of complete call hierarchy."""

    root: CallNode
    total_calls: int
    total_time: int
    max_depth: int

    def find_hotspots(self, threshold_percent: float = 5.0) -> List[CallNode]:
        """Find call nodes consuming more than threshold% of total time."""
        threshold_time = self.total_time * (threshold_percent / 100)
        hotspots = []

        def traverse(node: CallNode) -> None:
            if node.self_time >= threshold_time:
                hotspots.append(node)
            for child in node.children:
                traverse(child)

        traverse(self.root)
        return sorted(hotspots, key=lambda n: n.self_time, reverse=True)


@dataclass(frozen=True)
class EnvironmentInfo:
    """Immutable representation of execution environment."""

    python_version: str
    platform: str
    cpu_count: int
    memory_total: int
    working_directory: str
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable representation of execution context."""

    command_line: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None

    @property
    def duration(self) -> Optional[float]:
        """Execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


@dataclass(frozen=True)
class ProfileSession:
    """
    Immutable representation of a complete profiling session.

    Contains all collected data, analysis results, and metadata
    for a single profiling run.
    """

    session_id: str
    timestamp: datetime
    target_package: str
    configuration: "ProfileConfig"

    # Raw profiling data
    execution_events: List[ExecutionEvent] = field(default_factory=list)
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)
    call_tree: Optional[CallTree] = None
    source_mapping: Dict[str, SourceLocation] = field(default_factory=dict)

    # Context information
    environment_info: Optional[EnvironmentInfo] = None
    execution_context: Optional[ExecutionContext] = None

    # Analysis results (populated after analysis)
    analysis_result: Optional["AnalysisResult"] = None

    @property
    def total_events(self) -> int:
        """Total number of profiling events collected."""
        return len(self.execution_events)

    @property
    def total_time(self) -> int:
        """Total execution time in nanoseconds."""
        if self.call_tree:
            return self.call_tree.total_time
        return 0

    @property
    def peak_memory(self) -> int:
        """Peak memory usage in bytes."""
        if not self.memory_snapshots:
            return 0
        return max(snapshot.peak_memory for snapshot in self.memory_snapshots)


@dataclass(frozen=True)
class DetectedPattern:
    """Immutable representation of a detected performance pattern."""

    pattern_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    source_location: SourceLocation
    description: str
    impact_estimate: float  # estimated performance impact (0.0 to 1.0)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationRecommendation:
    """Immutable representation of an optimization recommendation."""

    recommendation_id: str
    title: str
    description: str
    target_location: SourceLocation
    estimated_improvement: float  # estimated improvement factor
    confidence: float  # confidence in recommendation (0.0 to 1.0)
    complexity: str  # 'low', 'medium', 'high'

    # Implementation guidance
    suggested_approach: str
    code_example: Optional[str] = None
    resources: List[str] = field(default_factory=list)

    # Related patterns
    addresses_patterns: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StaticAnalysisResult:
    """Results from static code analysis."""

    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    detected_patterns: List[DetectedPattern] = field(default_factory=list)
    code_quality_score: float = 0.0


@dataclass(frozen=True)
class DynamicAnalysisResult:
    """Results from dynamic execution analysis."""

    hotspots: List[CallNode] = field(default_factory=list)
    memory_leaks: List[DetectedPattern] = field(default_factory=list)
    performance_metrics: Dict[MetricType, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    """
    Immutable representation of complete analysis results.

    Contains all insights derived from profiling data including
    detected patterns, optimization recommendations, and metrics.
    """

    session_id: str
    analysis_timestamp: datetime

    # Analysis components
    static_analysis: StaticAnalysisResult
    dynamic_analysis: DynamicAnalysisResult

    # Synthesis results
    detected_patterns: List[DetectedPattern] = field(default_factory=list)
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)

    # Summary metrics
    overall_score: float = 0.0
    performance_grade: str = "unknown"

    @property
    def critical_issues(self) -> List[DetectedPattern]:
        """Patterns marked as critical severity."""
        return [p for p in self.detected_patterns if p.severity == "critical"]

    @property
    def high_impact_recommendations(self) -> List[OptimizationRecommendation]:
        """Recommendations with high estimated improvement."""
        return [r for r in self.recommendations if r.estimated_improvement >= 2.0]


@dataclass(frozen=True)
class Dashboard:
    """Immutable representation of a visualization dashboard."""

    dashboard_id: str
    title: str
    charts: List[Dict[str, Any]] = field(default_factory=list)
    interactions: Dict[str, Any] = field(default_factory=dict)


# Utility functions for creating model instances


def create_session_id() -> str:
    """Generate a unique session identifier."""
    return f"session_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"


def create_recommendation_id() -> str:
    """Generate a unique recommendation identifier."""
    return f"rec_{uuid.uuid4().hex[:8]}"
