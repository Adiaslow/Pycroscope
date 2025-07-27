"""
Pytest configuration for Pycroscope test suite.

Provides fixtures, markers, and test environment setup
following clean architecture and testing best practices.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pycroscope.core.config import ProfileConfig
from pycroscope.core.session import ProfileSession
from pycroscope.core.container import get_container, reset_container
from pycroscope.core.constants import ProfilerType
from pycroscope.core.session import ProfileResult


# Pytest markers for test organization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "core: Core layer tests")
    config.addinivalue_line("markers", "application: Application layer tests")
    config.addinivalue_line("markers", "infrastructure: Infrastructure layer tests")


@pytest.fixture(scope="session")
def temp_base_dir():
    """Create base temporary directory for test session."""
    temp_dir = Path(tempfile.mkdtemp(prefix="pycroscope_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir(temp_base_dir):
    """Create temporary directory for individual test."""
    test_id = getattr(pytest, "current_test_id", "test")
    test_dir = temp_base_dir / f"test_{test_id}"
    test_dir.mkdir(exist_ok=True)
    yield test_dir


@pytest.fixture
def clean_container():
    """Provide clean dependency injection container for each test."""
    reset_container()
    container = get_container()
    yield container
    reset_container()


@pytest.fixture
def minimal_config(temp_dir):
    """Create minimal ProfileConfig for testing."""
    return ProfileConfig(
        line_profiling=False,
        memory_profiling=False,
        call_profiling=True,
        output_dir=temp_dir,
        session_name="test_session",
    )


@pytest.fixture
def full_config(temp_dir):
    """Create full ProfileConfig for comprehensive testing."""
    return ProfileConfig(
        line_profiling=True,
        memory_profiling=True,
        call_profiling=True,
        # Skip for tests to avoid external dependencies
        output_dir=temp_dir,
        session_name="full_test_session",
        memory_precision=3,
        max_call_depth=100,
        generate_reports=True,
        create_visualizations=True,
        analyze_patterns=True,
        use_thread_isolation=True,
        cleanup_on_exit=True,
        save_raw_data=True,
    )


@pytest.fixture
def profile_session(minimal_config):
    """Create ProfileSession for testing."""
    return ProfileSession.create(minimal_config)


@pytest.fixture
def completed_session(full_config, temp_dir):
    """Create completed ProfileSession with sample data."""
    from datetime import datetime, timezone

    session = ProfileSession.create(full_config)
    session.start()

    # Add some sample results
    sample_result = ProfileResult(
        profiler_type="call",
        data={"test": "data"},
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        success=True,
    )
    session.add_result("call", sample_result)
    session.complete()
    return session


@pytest.fixture
def sample_call_data():
    """Sample call profiler data for testing."""
    return {
        "stats": {
            "function_a": {"ncalls": 100, "tottime": 1.5, "cumtime": 2.0},
            "function_b": {"ncalls": 50, "tottime": 0.8, "cumtime": 1.2},
            "function_c": {"ncalls": 200, "tottime": 0.3, "cumtime": 0.5},
        }
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory profiler data for testing."""
    return {
        "samples": [
            {"timestamp": 0.0, "rss_mb": 8.0, "vms_mb": 12.0, "percent": 4.0},
            {"timestamp": 0.2, "rss_mb": 10.0, "vms_mb": 14.0, "percent": 5.0},
            {"timestamp": 0.4, "rss_mb": 15.0, "vms_mb": 18.0, "percent": 7.5},
            {"timestamp": 0.6, "rss_mb": 12.0, "vms_mb": 15.0, "percent": 6.0},
            {"timestamp": 0.8, "rss_mb": 9.0, "vms_mb": 13.0, "percent": 4.5},
        ],
        "peak_memory_mb": 15.0,
        "avg_memory_mb": 10.8,
        "memory_delta_mb": 1.0,
        "memory_growth_rate": 1.25,
        "memory_volatility": 2.8,
        "memory_spikes": 1,
    }


@pytest.fixture
def mock_profiler():
    """Create mock profiler for testing orchestration."""
    profiler = Mock()
    profiler.name = "test_profiler"
    profiler.is_available.return_value = True
    profiler.start.return_value = True
    profiler.stop.return_value = {"test": "data"}
    profiler.duration = 1.0
    profiler.check_conflicts.return_value = None
    return profiler


@pytest.fixture
def mock_profiler_factory():
    """Create mock profiler factory for testing."""
    factory = Mock()
    factory.supports.return_value = True
    factory.create.return_value = Mock()
    return factory


# Test data constants
TEST_PROFILER_TYPES = [
    ProfilerType.CALL.value,
    ProfilerType.MEMORY.value,
    ProfilerType.LINE.value,
]


# Utility functions for tests
def get_test_id():
    """Get current test identifier."""
    return getattr(pytest, "current_test_id", "unknown_test")


def create_test_file(temp_dir: Path, filename: str, content: str) -> Path:
    """Create test file with content."""
    file_path = temp_dir / filename
    file_path.write_text(content)
    return file_path


def assert_valid_config(config: ProfileConfig):
    """Assert config is valid for testing."""
    assert isinstance(config, ProfileConfig)
    assert config.output_dir is not None
    assert isinstance(config.output_dir, Path)


def assert_valid_session(session: ProfileSession):
    """Assert session is valid for testing."""
    assert isinstance(session, ProfileSession)
    assert session.session_id is not None
    assert session.config is not None


def assert_no_fallbacks_in_code(file_content: str):
    """Assert no fallback patterns exist in code."""
    fallback_patterns = [" or ", "except ImportError", "try:\n.*import"]
    for pattern in fallback_patterns:
        assert (
            pattern not in file_content
        ), f"Fallback pattern '{pattern}' found in code"


def assert_clean_imports(file_content: str):
    """Assert imports follow our principles."""
    lines = file_content.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            # Should be at module level (not in function)
            assert not any(
                "def " in prev_line for prev_line in lines[:i] if prev_line.strip()
            )


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Simple performance timer for tests."""
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
        def duration(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

    return Timer()


# Custom pytest hooks
def pytest_runtest_setup(item):
    """Setup for each test."""
    # Store test identifier for fixtures (safe attribute setting)
    if not hasattr(pytest, "current_test_id"):
        pytest.current_test_id = item.nodeid.replace("::", "_").replace("/", "_")


def pytest_runtest_teardown(item, nextitem):
    """Teardown for each test."""
    # Clean up any global state
    reset_container()
