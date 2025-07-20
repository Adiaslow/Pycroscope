"""
Unit tests for core data models.

Tests all immutable data structures including ProfileSession,
ExecutionEvent, AnalysisResult, and related classes.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from pycroscope.core.models import (
    ProfileSession,
    ExecutionEvent,
    MemorySnapshot,
    CallNode,
    CallTree,
    EnvironmentInfo,
    ExecutionContext,
    SourceLocation,
    FrameInfo,
    DetectedPattern,
    OptimizationRecommendation,
    AnalysisResult,
    EventType,
    MetricType,
)
from pycroscope.core.config import ProfileConfig


class TestSourceLocation:
    """Test SourceLocation immutable data class."""

    def test_creation(self):
        """Test basic SourceLocation creation."""
        location = SourceLocation(
            filename="test.py", line_number=42, function_name="test_func"
        )

        assert location.filename == "test.py"
        assert location.line_number == 42
        assert location.function_name == "test_func"
        assert location.module_name is None

    def test_with_module(self):
        """Test SourceLocation with module name."""
        location = SourceLocation(
            filename="test.py",
            line_number=42,
            function_name="test_func",
            module_name="mymodule",
        )

        assert location.module_name == "mymodule"

    def test_string_representation(self):
        """Test SourceLocation string representation."""
        location = SourceLocation(
            filename="test.py", line_number=42, function_name="test_func"
        )

        expected = "test_func:test.py:42"
        assert str(location) == expected

    def test_string_representation_with_module(self):
        """Test SourceLocation string representation with module."""
        location = SourceLocation(
            filename="test.py",
            line_number=42,
            function_name="test_func",
            module_name="mymodule",
        )

        expected = "mymodule.test_func:test.py:42"
        assert str(location) == expected

    def test_immutability(self):
        """Test that SourceLocation is immutable."""
        location = SourceLocation(
            filename="test.py", line_number=42, function_name="test_func"
        )

        with pytest.raises((AttributeError, TypeError)):
            location.filename = "other.py"


class TestFrameInfo:
    """Test FrameInfo immutable data class."""

    def test_creation(self, sample_source_location):
        """Test basic FrameInfo creation."""
        frame = FrameInfo(
            source_location=sample_source_location,
            local_variables={"x": 1, "y": "test"},
        )

        assert frame.source_location == sample_source_location
        assert frame.local_variables == {"x": 1, "y": "test"}
        assert frame.frame_id is not None

    def test_auto_generated_frame_id(self, sample_source_location):
        """Test that frame_id is auto-generated."""
        frame1 = FrameInfo(source_location=sample_source_location)
        frame2 = FrameInfo(source_location=sample_source_location)

        assert frame1.frame_id != frame2.frame_id
        assert len(frame1.frame_id) > 0

    def test_empty_variables(self, sample_source_location):
        """Test FrameInfo with empty local variables."""
        frame = FrameInfo(source_location=sample_source_location)

        assert frame.local_variables == {}

    def test_immutability(self, sample_source_location):
        """Test that FrameInfo is immutable."""
        frame = FrameInfo(source_location=sample_source_location)

        with pytest.raises((AttributeError, TypeError)):
            frame.local_variables = {"new": "value"}


class TestExecutionEvent:
    """Test ExecutionEvent immutable data class."""

    def test_creation(self, sample_frame_info):
        """Test basic ExecutionEvent creation."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        event = ExecutionEvent(
            timestamp=timestamp_ns,
            event_type=EventType.LINE,
            thread_id=12345,
            frame_info=sample_frame_info,
            execution_time=1_000_000,  # 1ms in nanoseconds
            memory_delta=1024,
        )

        assert event.timestamp == timestamp_ns
        assert event.event_type == EventType.LINE
        assert event.thread_id == 12345
        assert event.frame_info == sample_frame_info
        assert event.execution_time == 1_000_000
        assert event.memory_delta == 1024

    def test_duration_ms_property(self, sample_frame_info):
        """Test duration_ms property calculation."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        event = ExecutionEvent(
            timestamp=timestamp_ns,
            event_type=EventType.CALL,
            thread_id=12345,
            frame_info=sample_frame_info,
            execution_time=2_500_000,  # 2.5ms in nanoseconds
        )

        assert event.duration_ms == 2.5

    def test_duration_ms_none_when_no_execution_time(self, sample_frame_info):
        """Test duration_ms returns None when execution_time is None."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        event = ExecutionEvent(
            timestamp=timestamp_ns,
            event_type=EventType.LINE,
            thread_id=12345,
            frame_info=sample_frame_info,
        )

        assert event.duration_ms is None

    def test_optional_fields(self, sample_frame_info):
        """Test ExecutionEvent with minimal required fields."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        event = ExecutionEvent(
            timestamp=timestamp_ns,
            event_type=EventType.EXCEPTION,
            thread_id=12345,
            frame_info=sample_frame_info,
        )

        assert event.execution_time is None
        assert event.memory_delta is None
        assert event.cpu_usage is None
        assert event.call_stack == []
        assert event.event_data == {}

    def test_event_types(self, sample_frame_info):
        """Test different event types."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        for event_type in EventType:
            event = ExecutionEvent(
                timestamp=timestamp_ns,
                event_type=event_type,
                thread_id=12345,
                frame_info=sample_frame_info,
            )
            assert event.event_type == event_type


class TestMemorySnapshot:
    """Test MemorySnapshot immutable data class."""

    def test_creation(self):
        """Test basic MemorySnapshot creation."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        snapshot = MemorySnapshot(
            timestamp=timestamp_ns,
            total_memory=1024 * 1024,  # 1MB
            peak_memory=1024 * 1024 * 2,  # 2MB
            gc_collections=5,
            object_counts={"list": 100, "dict": 50},
        )

        assert snapshot.timestamp == timestamp_ns
        assert snapshot.total_memory == 1024 * 1024
        assert snapshot.peak_memory == 1024 * 1024 * 2
        assert snapshot.gc_collections == 5
        assert snapshot.object_counts == {"list": 100, "dict": 50}

    def test_empty_object_counts(self):
        """Test MemorySnapshot with empty object counts."""
        timestamp_ns = int(datetime.now().timestamp() * 1_000_000_000)

        snapshot = MemorySnapshot(
            timestamp=timestamp_ns,
            total_memory=1024,
            peak_memory=2048,
            gc_collections=0,
        )

        assert snapshot.object_counts == {}


class TestCallNode:
    """Test CallNode immutable data class."""

    def test_creation(self, sample_source_location):
        """Test basic CallNode creation."""
        node = CallNode(
            source_location=sample_source_location,
            total_time=1_000_000_000,  # 1 second in nanoseconds
            self_time=500_000_000,  # 0.5 seconds
            call_count=10,
            memory_allocated=4096,
        )

        assert node.source_location == sample_source_location
        assert node.total_time == 1_000_000_000
        assert node.self_time == 500_000_000
        assert node.call_count == 10
        assert node.memory_allocated == 4096
        assert node.children == []

    def test_children_time_property(self, sample_source_location):
        """Test children_time property calculation."""
        node = CallNode(
            source_location=sample_source_location,
            total_time=1_000_000_000,
            self_time=300_000_000,
            call_count=1,
            memory_allocated=1024,
        )

        expected_children_time = 1_000_000_000 - 300_000_000
        assert node.children_time == expected_children_time

    def test_with_children(self, sample_source_location):
        """Test CallNode with child nodes."""
        child1 = CallNode(
            source_location=sample_source_location,
            total_time=200_000_000,
            self_time=200_000_000,
            call_count=1,
            memory_allocated=512,
        )

        child2 = CallNode(
            source_location=sample_source_location,
            total_time=300_000_000,
            self_time=300_000_000,
            call_count=2,
            memory_allocated=1024,
        )

        parent = CallNode(
            source_location=sample_source_location,
            total_time=1_000_000_000,
            self_time=500_000_000,
            call_count=1,
            memory_allocated=2048,
            children=[child1, child2],
        )

        assert len(parent.children) == 2
        assert child1 in parent.children
        assert child2 in parent.children


class TestCallTree:
    """Test CallTree immutable data class."""

    def test_creation(self, sample_call_tree):
        """Test basic CallTree creation."""
        assert sample_call_tree.root is not None
        assert sample_call_tree.total_calls > 0
        assert sample_call_tree.total_time > 0
        assert sample_call_tree.max_depth >= 1

    def test_find_hotspots(self, sample_call_tree):
        """Test find_hotspots method."""
        # Find hotspots with 1% threshold (should find most nodes)
        hotspots = sample_call_tree.find_hotspots(threshold_percent=1.0)

        assert isinstance(hotspots, list)
        assert len(hotspots) > 0

        # Hotspots should be sorted by self_time descending
        for i in range(len(hotspots) - 1):
            assert hotspots[i].self_time >= hotspots[i + 1].self_time

    def test_find_hotspots_high_threshold(self, sample_call_tree):
        """Test find_hotspots with high threshold."""
        # High threshold should find fewer or no hotspots
        hotspots = sample_call_tree.find_hotspots(threshold_percent=90.0)

        # Should find fewer hotspots with high threshold
        assert len(hotspots) <= 1


class TestEnvironmentInfo:
    """Test EnvironmentInfo immutable data class."""

    def test_creation(self):
        """Test basic EnvironmentInfo creation."""
        env = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/dir",
        )

        assert env.python_version == "3.9.7"
        assert env.platform == "linux"
        assert env.cpu_count == 4
        assert env.memory_total == 8589934592
        assert env.working_directory == "/test/dir"
        assert env.environment_variables == {}

    def test_with_environment_variables(self):
        """Test EnvironmentInfo with environment variables."""
        env_vars = {"PYTHON_PATH": "/usr/bin/python", "DEBUG": "1"}

        env = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/dir",
            environment_variables=env_vars,
        )

        assert env.environment_variables == env_vars


class TestExecutionContext:
    """Test ExecutionContext immutable data class."""

    def test_creation(self):
        """Test basic ExecutionContext creation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)

        context = ExecutionContext(
            command_line=["python", "test.py"],
            start_time=start_time,
            end_time=end_time,
            exit_code=0,
        )

        assert context.command_line == ["python", "test.py"]
        assert context.start_time == start_time
        assert context.end_time == end_time
        assert context.exit_code == 0

    def test_duration_property(self):
        """Test duration property calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5.5)

        context = ExecutionContext(
            command_line=["python", "test.py"], start_time=start_time, end_time=end_time
        )

        assert context.duration == 5.5

    def test_duration_none_when_no_end_time(self):
        """Test duration returns None when end_time is None."""
        context = ExecutionContext(
            command_line=["python", "test.py"], start_time=datetime.now()
        )

        assert context.duration is None

    def test_minimal_creation(self):
        """Test ExecutionContext with minimal required fields."""
        context = ExecutionContext(
            command_line=["python", "test.py"], start_time=datetime.now()
        )

        assert context.end_time is None
        assert context.exit_code is None


class TestProfileSession:
    """Test ProfileSession immutable data class."""

    def test_creation(self, sample_profile_session):
        """Test basic ProfileSession creation."""
        assert sample_profile_session.session_id == "test_session_123"
        assert sample_profile_session.target_package == "test_package"
        assert isinstance(sample_profile_session.timestamp, datetime)
        assert sample_profile_session.configuration is not None

    def test_total_events_property(self, sample_profile_session):
        """Test total_events property."""
        assert sample_profile_session.total_events == len(
            sample_profile_session.execution_events
        )

    def test_total_time_property(self, sample_profile_session):
        """Test total_time property."""
        if sample_profile_session.call_tree:
            assert (
                sample_profile_session.total_time
                == sample_profile_session.call_tree.total_time
            )
        else:
            assert sample_profile_session.total_time == 0

    def test_peak_memory_property(self, sample_profile_session):
        """Test peak_memory property."""
        if sample_profile_session.memory_snapshots:
            expected_peak = max(
                snap.peak_memory for snap in sample_profile_session.memory_snapshots
            )
            assert sample_profile_session.peak_memory == expected_peak
        else:
            assert sample_profile_session.peak_memory == 0

    def test_empty_session(self, sample_config):
        """Test ProfileSession with minimal data."""
        session = ProfileSession(
            session_id="empty_session",
            timestamp=datetime.now(),
            target_package="empty_package",
            configuration=sample_config,
        )

        assert session.total_events == 0
        assert session.total_time == 0
        assert session.peak_memory == 0


class TestDetectedPattern:
    """Test DetectedPattern immutable data class."""

    def test_creation(self, sample_source_location):
        """Test basic DetectedPattern creation."""
        pattern = DetectedPattern(
            pattern_type="performance_bottleneck",
            severity="high",
            source_location=sample_source_location,
            description="Function call in tight loop",
            impact_estimate=0.8,
            evidence={"loop_count": 1000, "call_frequency": "high"},
        )

        assert pattern.pattern_type == "performance_bottleneck"
        assert pattern.severity == "high"
        assert pattern.source_location == sample_source_location
        assert pattern.description == "Function call in tight loop"
        assert pattern.impact_estimate == 0.8
        assert pattern.evidence == {"loop_count": 1000, "call_frequency": "high"}

    def test_empty_evidence(self, sample_source_location):
        """Test DetectedPattern with empty evidence."""
        pattern = DetectedPattern(
            pattern_type="minor_issue",
            severity="low",
            source_location=sample_source_location,
            description="Minor optimization opportunity",
            impact_estimate=0.1,
        )

        assert pattern.evidence == {}


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation immutable data class."""

    def test_creation(self, sample_source_location):
        """Test basic OptimizationRecommendation creation."""
        recommendation = OptimizationRecommendation(
            recommendation_id="opt_001",
            title="Use list comprehension",
            description="Replace loop with list comprehension for better performance",
            target_location=sample_source_location,
            estimated_improvement=2.5,
            confidence=0.9,
            complexity="low",
            suggested_approach="Replace the for loop with a list comprehension",
            code_example="result = [x*2 for x in items]",
            resources=["https://docs.python.org/3/tutorial/datastructures.html"],
            addresses_patterns=["inefficient_loop"],
        )

        assert recommendation.recommendation_id == "opt_001"
        assert recommendation.title == "Use list comprehension"
        assert recommendation.estimated_improvement == 2.5
        assert recommendation.confidence == 0.9
        assert recommendation.complexity == "low"
        assert len(recommendation.resources) == 1
        assert len(recommendation.addresses_patterns) == 1

    def test_minimal_creation(self, sample_source_location):
        """Test OptimizationRecommendation with minimal required fields."""
        recommendation = OptimizationRecommendation(
            recommendation_id="opt_minimal",
            title="Basic optimization",
            description="Simple optimization",
            target_location=sample_source_location,
            estimated_improvement=1.0,
            confidence=0.5,
            complexity="medium",
            suggested_approach="Apply optimization",
        )

        assert recommendation.code_example is None
        assert recommendation.resources == []
        assert recommendation.addresses_patterns == []
