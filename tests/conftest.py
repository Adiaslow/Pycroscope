"""
Pytest configuration and shared fixtures for Pycroscope tests.

Provides common test fixtures, utilities, and configuration
for all test modules.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest

from pycroscope.core.config import (
    AnalysisConfig,
    AnalysisType,
    CollectorConfig,
    CollectorType,
    ProfileConfig,
    StorageConfig,
    StorageType,
)
from pycroscope.core.models import (
    CallTree,
    EnvironmentInfo,
    ExecutionContext,
    ExecutionEvent,
    FrameInfo,
    MemorySnapshot,
    ProfileSession,
    SourceLocation,
)
from pycroscope.storage.file_store import FileDataStore
from pycroscope.storage.memory_store import MemoryDataStore

# Add pycroscope to path for testing - disabled for editable install
# sys.path.insert(0, str(Path(__file__).parent.parent))



@pytest.fixture(scope="session")
def test_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory(prefix="pycroscope_test_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_storage_dir():
    """Temporary directory for storage tests."""
    with tempfile.TemporaryDirectory(prefix="pycroscope_storage_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Sample ProfileConfig for testing."""
    config = ProfileConfig()
    config.target_package = "test_package"
    config.debug_mode = True
    config.verbose = True
    config.parallel_collection = False  # Disable for predictable testing
    config.max_threads = 2
    config.timeout_seconds = 30
    return config


@pytest.fixture
def file_storage_config(temp_storage_dir):
    """Storage configuration for file-based storage."""
    config = StorageConfig()
    config.storage_type = StorageType.FILE
    config.storage_path = temp_storage_dir / "data"
    config.compression = True
    config.max_sessions = 10
    config.retention_days = 7
    config.auto_cleanup = False  # Disable for testing
    return config


@pytest.fixture
def memory_storage_config():
    """Storage configuration for memory-based storage."""
    config = StorageConfig()
    config.storage_type = StorageType.MEMORY
    config.max_sessions = 5
    return config


@pytest.fixture
def sample_environment_info():
    """Sample EnvironmentInfo for testing."""
    return EnvironmentInfo(
        python_version="3.9.7",
        platform="linux",
        cpu_count=4,
        memory_total=8589934592,  # 8GB
        working_directory="/test/working/dir",
    )


@pytest.fixture
def sample_execution_context():
    """Sample ExecutionContext for testing."""
    start_time = datetime.now()
    return ExecutionContext(
        command_line=["python", "test_script.py"],
        start_time=start_time,
        end_time=start_time + timedelta(seconds=5),
        exit_code=0,
    )


@pytest.fixture
def sample_source_location():
    """Sample SourceLocation for testing."""
    return SourceLocation(
        filename="/test/module.py", line_number=42, function_name="test_function"
    )


@pytest.fixture
def sample_frame_info(sample_source_location):
    """Sample FrameInfo for testing."""
    return FrameInfo(
        source_location=sample_source_location,
        local_variables={"x": 10, "y": "test"},
    )


@pytest.fixture
def sample_execution_events(sample_frame_info):
    """Sample ExecutionEvent list for testing."""
    from pycroscope.core.models import EventType

    base_time = datetime.now()

    events = []
    for i in range(10):
        # Convert timestamp to nanoseconds
        timestamp_ns = int(
            (base_time + timedelta(microseconds=i * 1000)).timestamp() * 1_000_000_000
        )

        event = ExecutionEvent(
            timestamp=timestamp_ns,
            event_type=EventType.LINE if i % 2 == 0 else EventType.CALL,
            thread_id=12345,
            frame_info=sample_frame_info,
            execution_time=i * 100_000,  # nanoseconds
            memory_delta=1024 * (i + 1),
            event_data={"test_data": f"value_{i}"},
        )
        events.append(event)

    return events


@pytest.fixture
def sample_memory_snapshots():
    """Sample MemorySnapshot list for testing."""
    base_time = datetime.now()

    snapshots = []
    for i in range(5):
        # Convert timestamp to nanoseconds
        timestamp_ns = int(
            (base_time + timedelta(seconds=i)).timestamp() * 1_000_000_000
        )

        snapshot = MemorySnapshot(
            timestamp=timestamp_ns,
            total_memory=1024 * 1024 * (10 + i),  # Start at 10MB
            peak_memory=1024 * 1024 * (12 + i),  # Peak higher than total
            gc_collections=i * 2,
            object_counts={
                "list": 100 + i * 10,
                "dict": 50 + i * 5,
                "str": 200 + i * 20,
            },
        )
        snapshots.append(snapshot)

    return snapshots


@pytest.fixture
def sample_call_tree(sample_source_location):
    """Sample CallTree for testing."""
    from pycroscope.core.models import CallNode, CallTree

    # Create child nodes first
    children = []
    for i in range(3):
        child_location = SourceLocation(
            filename=f"/test/child_{i}.py",
            line_number=10 + i,
            function_name=f"child_function_{i}",
        )

        child = CallNode(
            source_location=child_location,
            total_time=(25 + i * 5) * 1_000_000,  # Convert to nanoseconds
            self_time=(15 + i * 2) * 1_000_000,  # Convert to nanoseconds
            call_count=2 + i,
            memory_allocated=1024 * (i + 1),
            children=[],
        )
        children.append(child)

    # Create root node
    root = CallNode(
        source_location=sample_source_location,
        total_time=100 * 1_000_000,  # Convert to nanoseconds
        self_time=20 * 1_000_000,  # Convert to nanoseconds
        call_count=1,
        memory_allocated=4096,
        children=children,
    )

    # Create CallTree wrapper
    call_tree = CallTree(
        root=root,
        total_calls=1 + sum(child.call_count for child in children),
        total_time=root.total_time,
        max_depth=2,  # root + children
    )

    return call_tree


@pytest.fixture
def sample_profile_session(
    sample_config,
    sample_environment_info,
    sample_execution_context,
    sample_execution_events,
    sample_memory_snapshots,
    sample_call_tree,
):
    """Complete sample ProfileSession for testing."""
    # Source mapping
    source_mapping = {
        "/test/module.py": SourceLocation(
            filename="/test/module.py", line_number=1, function_name="<module>"
        )
    }

    session = ProfileSession(
        session_id="test_session_123",
        timestamp=datetime.now(),
        target_package="test_package",
        configuration=sample_config,
        execution_events=sample_execution_events,
        memory_snapshots=sample_memory_snapshots,
        call_tree=sample_call_tree,
        source_mapping=source_mapping,
        environment_info=sample_environment_info,
        execution_context=sample_execution_context,
    )

    return session


@pytest.fixture
def file_data_store(file_storage_config):
    """FileDataStore instance for testing."""
    return FileDataStore(file_storage_config)


@pytest.fixture
def memory_data_store(memory_storage_config):
    """MemoryDataStore instance for testing."""
    return MemoryDataStore(memory_storage_config)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for CLI testing."""
    mock = MagicMock()
    mock.run.return_value.returncode = 0
    mock.run.return_value.stdout = "Mock output"
    mock.run.return_value.stderr = ""
    return mock


@pytest.fixture
def cleanup_registry():
    """Clean up component registry after each test."""
    yield
    # Reset registry state
    from pycroscope.core.registry import ComponentRegistry

    registry = ComponentRegistry()
    registry.reset()


@pytest.fixture
def capture_output():
    """Capture stdout and stderr for CLI testing."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        yield {"stdout": stdout_capture, "stderr": stderr_capture}


# Test data generators
class TestDataGenerator:
    """Utility class for generating test data."""

    @staticmethod
    def create_execution_events(
        count: int, base_time: Optional[datetime] = None
    ) -> List[ExecutionEvent]:
        """Generate a list of ExecutionEvent objects for testing."""
        if base_time is None:
            base_time = datetime.now()

        events = []
        for i in range(count):
            event = ExecutionEvent(
                event_id=f"generated_event_{i}",
                timestamp=base_time + timedelta(microseconds=i * 100),
                event_type="line" if i % 3 == 0 else "call" if i % 3 == 1 else "return",
                source_location=SourceLocation(
                    filename=f"/generated/file_{i % 5}.py",
                    line_number=10 + (i % 50),
                    function_name=f"function_{i % 10}",
                ),
                frame_info=FrameInfo(
                    filename=f"/generated/file_{i % 5}.py",
                    line_number=10 + (i % 50),
                    function_name=f"function_{i % 10}",
                    code_context=[f"# Line {10 + (i % 50)}"],
                    local_variables={"var": i},
                ),
                execution_time_ms=float(i * 0.01),
                memory_usage=1000 + i * 10,
                thread_id=12345,
                metadata={"generated": True, "index": i},
            )
            events.append(event)

        return events

    @staticmethod
    def create_memory_snapshots(
        count: int, base_time: Optional[datetime] = None
    ) -> List[MemorySnapshot]:
        """Generate a list of MemorySnapshot objects for testing."""
        if base_time is None:
            base_time = datetime.now()

        snapshots = []
        for i in range(count):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i),
                total_memory=1024 * 1024 * (100 + i * 10),
                available_memory=1024 * 1024 * (50 + i * 5),
                memory_percent=40.0 + i * 2,
                swap_memory=1024 * 1024 * i,
                memory_objects={
                    "list": 50 + i * 3,
                    "dict": 30 + i * 2,
                    "str": 100 + i * 5,
                },
            )
            snapshots.append(snapshot)

        return snapshots


@pytest.fixture
def test_data_generator():
    """TestDataGenerator instance for testing."""
    return TestDataGenerator()


# Utility functions for tests
def assert_events_equal(
    event1: ExecutionEvent, event2: ExecutionEvent, ignore_timestamps: bool = False
):
    """Assert that two ExecutionEvent objects are equal."""
    assert event1.event_id == event2.event_id
    assert event1.event_type == event2.event_type
    assert event1.source_location.filename == event2.source_location.filename
    assert event1.source_location.line_number == event2.source_location.line_number
    assert event1.source_location.function_name == event2.source_location.function_name
    assert event1.execution_time_ms == event2.execution_time_ms
    assert event1.memory_usage == event2.memory_usage
    assert event1.thread_id == event2.thread_id
    assert event1.metadata == event2.metadata

    if not ignore_timestamps:
        assert event1.timestamp == event2.timestamp


def assert_sessions_equal(
    session1: ProfileSession, session2: ProfileSession, ignore_timestamps: bool = False
):
    """Assert that two ProfileSession objects are equal."""
    assert session1.session_id == session2.session_id
    assert session1.target_package == session2.target_package
    assert len(session1.execution_events) == len(session2.execution_events)
    assert len(session1.memory_snapshots) == len(session2.memory_snapshots)

    if not ignore_timestamps:
        assert session1.timestamp == session2.timestamp

    # Compare events
    for e1, e2 in zip(session1.execution_events, session2.execution_events):
        assert_events_equal(e1, e2, ignore_timestamps)


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, *args):
            self.stop()

    return Timer()
