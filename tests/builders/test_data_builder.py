"""
Test data builders for Pycroscope tests.

Provides fluent builders for creating test data objects
following the Builder pattern and testing best practices.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import sys

# Ensure proper import path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pycroscope.core.config import ProfileConfig
from pycroscope.core.session import ProfileSession, ProfileResult, SessionStatus
from pycroscope.core.constants import ProfilerType


class ConfigBuilder:
    """
    Fluent builder for ProfileConfig test instances.

    Provides clean interface for creating test configurations
    without complex parameter passing.
    """

    def __init__(self):
        self._data = {
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": True,
            "output_dir": None,
            "session_name": "test_session",
        }

    def with_line_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable line profiling."""
        self._data["line_profiling"] = enabled
        return self

    def with_memory_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable memory profiling."""
        self._data["memory_profiling"] = enabled
        return self

    def with_call_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable call profiling."""
        self._data["call_profiling"] = enabled
        return self

    def with_output_dir(self, output_dir: Path) -> "ConfigBuilder":
        """Set output directory."""
        self._data["output_dir"] = output_dir
        return self

    def with_session_name(self, name: str) -> "ConfigBuilder":
        """Set session name."""
        self._data["session_name"] = name
        return self

    def with_memory_precision(self, precision: int) -> "ConfigBuilder":
        """Set memory precision."""
        self._data["memory_precision"] = precision
        return self

    def with_max_call_depth(self, depth: int) -> "ConfigBuilder":
        """Set maximum call depth."""
        self._data["max_call_depth"] = depth
        return self

    def with_thread_isolation(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable thread isolation."""
        self._data["use_thread_isolation"] = enabled
        return self

    def with_reports(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable report generation."""
        self._data["generate_reports"] = enabled
        return self

    def with_visualizations(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable visualization creation."""
        self._data["create_visualizations"] = enabled
        return self

    def with_pattern_analysis(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable pattern analysis."""
        self._data["analyze_patterns"] = enabled
        return self

    def with_cleanup_on_exit(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable cleanup on exit."""
        self._data["cleanup_on_exit"] = enabled
        return self

    def with_raw_data_saving(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable/disable raw data saving."""
        self._data["save_raw_data"] = enabled
        return self

    def minimal(self) -> "ConfigBuilder":
        """Configure for minimal overhead testing."""
        self._data.update(
            {
                "line_profiling": False,
                "memory_profiling": False,
                "call_profiling": True,
                "generate_reports": False,
                "create_visualizations": False,
                "analyze_patterns": False,
            }
        )
        return self

    def comprehensive(self) -> "ConfigBuilder":
        """Configure for comprehensive testing."""
        self._data.update(
            {
                "line_profiling": True,
                "memory_profiling": True,
                "call_profiling": True,
                # Skip for tests
                "generate_reports": True,
                "create_visualizations": True,
                "analyze_patterns": True,
            }
        )
        return self

    def build(self) -> ProfileConfig:
        """Build ProfileConfig instance."""
        return ProfileConfig(**self._data)


class SessionBuilder:
    """
    Fluent builder for ProfileSession test instances.

    Provides clean interface for creating test sessions
    with various states and data.
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        self._config = config or ConfigBuilder().build()
        self._results = {}
        self._start_time = None
        self._end_time = None

    def with_config(self, config: ProfileConfig) -> "SessionBuilder":
        """Set session configuration."""
        self._config = config
        return self

    def started(self) -> "SessionBuilder":
        """Mark session as started."""
        self._start_time = datetime.now(timezone.utc)
        return self

    def completed(self) -> "SessionBuilder":
        """Mark session as completed."""
        if not self._start_time:
            self._start_time = datetime.now(timezone.utc)
        self._end_time = datetime.now(timezone.utc)
        return self

    def with_call_result(
        self, data: Optional[Dict[str, Any]] = None
    ) -> "SessionBuilder":
        """Add call profiler result."""
        if data is None:
            data = {
                "stats": {
                    "test_function": {"ncalls": 10, "tottime": 0.5, "cumtime": 0.8},
                    "helper_function": {"ncalls": 5, "tottime": 0.2, "cumtime": 0.3},
                }
            }

        result = ProfileResult(
            profiler_type="call",
            start_time=self._start_time or datetime.now(timezone.utc),
            end_time=self._end_time or datetime.now(timezone.utc),
            data=data,
            success=True,
        )
        self._results["call"] = result
        return self

    def with_memory_result(
        self, data: Optional[Dict[str, Any]] = None
    ) -> "SessionBuilder":
        """Add memory profiler result."""
        if data is None:
            data = {
                "samples": [
                    {"timestamp": 0.0, "rss_mb": 10.0, "vms_mb": 15.0, "percent": 5.0},
                    {"timestamp": 0.5, "rss_mb": 12.0, "vms_mb": 17.0, "percent": 6.0},
                    {"timestamp": 1.0, "rss_mb": 11.0, "vms_mb": 16.0, "percent": 5.5},
                ],
                "peak_memory_mb": 12.0,
            }

        result = ProfileResult(
            profiler_type="memory",
            start_time=self._start_time or datetime.now(timezone.utc),
            end_time=self._end_time or datetime.now(timezone.utc),
            data=data,
            success=True,
        )
        self._results["memory"] = result
        return self

    def with_line_result(
        self, data: Optional[Dict[str, Any]] = None
    ) -> "SessionBuilder":
        """Add line profiler result."""
        if data is None:
            data = {
                "line_stats": {
                    "test.py:10": {"hits": 100, "time": 0.05, "line_content": "x = 1"}
                }
            }

        result = ProfileResult(
            profiler_type="line",
            start_time=self._start_time or datetime.now(timezone.utc),
            end_time=self._end_time or datetime.now(timezone.utc),
            data=data,
            success=True,
        )
        self._results["line"] = result
        return self

    def build(self) -> "ProfileSession":
        """Build the ProfileSession."""
        session = ProfileSession.create(self._config)

        if self._start_time:
            session.start_time = self._start_time
            session.status = SessionStatus.RUNNING

        if self._end_time:
            session.end_time = self._end_time
            session.status = SessionStatus.COMPLETED

        for profiler_type, result in self._results.items():
            session.results[profiler_type] = result

        return session


class CallDataBuilder:
    """
    Builder for call profiler test data.

    Creates realistic call profiler statistics
    for comprehensive testing.
    """

    def __init__(self):
        self._stats = {}

    def with_function(
        self, name: str, ncalls: int, tottime: float, cumtime: float
    ) -> "CallDataBuilder":
        """Add function statistics."""
        self._stats[name] = {"ncalls": ncalls, "tottime": tottime, "cumtime": cumtime}
        return self

    def with_hot_function(self, name: str) -> "CallDataBuilder":
        """Add hot function (high call count and time)."""
        self._stats[name] = {"ncalls": 1000, "tottime": 2.5, "cumtime": 3.0}
        return self

    def with_expensive_function(self, name: str) -> "CallDataBuilder":
        """Add expensive function (high time per call)."""
        self._stats[name] = {"ncalls": 5, "tottime": 1.8, "cumtime": 2.2}
        return self

    def with_frequent_function(self, name: str) -> "CallDataBuilder":
        """Add frequently called function."""
        self._stats[name] = {"ncalls": 500, "tottime": 0.8, "cumtime": 1.0}
        return self

    def with_sample_dataset(self) -> "CallDataBuilder":
        """Add realistic sample dataset."""
        functions = [
            ("main", 1, 0.1, 5.0),
            ("process_data", 100, 2.0, 3.5),
            ("validate_input", 150, 0.5, 0.6),
            ("calculate_result", 75, 1.2, 1.8),
            ("save_output", 25, 0.8, 1.0),
        ]

        for name, ncalls, tottime, cumtime in functions:
            self.with_function(name, ncalls, tottime, cumtime)

        return self

    def build(self) -> Dict[str, Any]:
        """Build call profiler data."""
        return {"stats": self._stats}


class MemoryDataBuilder:
    """
    Builder for memory profiler test data.

    Creates realistic memory usage patterns
    for testing memory analysis.
    """

    def __init__(self):
        self._samples = []
        self._metadata = {}

    def with_sample(
        self, timestamp: float, rss_mb: float, vms_mb: float, percent: float
    ) -> "MemoryDataBuilder":
        """Add memory sample."""
        self._samples.append(
            {
                "timestamp": timestamp,
                "rss_mb": rss_mb,
                "vms_mb": vms_mb,
                "percent": percent,
            }
        )
        return self

    def with_steady_usage(
        self, base_memory: float = 10.0, duration: float = 1.0, samples: int = 10
    ) -> "MemoryDataBuilder":
        """Add steady memory usage pattern."""
        for i in range(samples):
            timestamp = (duration / samples) * i
            # Small random variation around base
            variation = (i % 3 - 1) * 0.2  # -0.2, 0, +0.2
            rss_mb = base_memory + variation
            vms_mb = rss_mb * 1.3
            percent = (rss_mb / 100) * 5  # Assume 100MB = 5%

            self.with_sample(timestamp, rss_mb, vms_mb, percent)

        return self

    def with_growth_pattern(
        self, start_memory: float = 8.0, end_memory: float = 15.0, samples: int = 10
    ) -> "MemoryDataBuilder":
        """Add memory growth pattern."""
        growth_per_sample = (end_memory - start_memory) / (samples - 1)

        for i in range(samples):
            timestamp = i * 0.1
            rss_mb = start_memory + (growth_per_sample * i)
            vms_mb = rss_mb * 1.2
            percent = (rss_mb / 200) * 10

            self.with_sample(timestamp, rss_mb, vms_mb, percent)

        return self

    def with_spike_pattern(
        self,
        base_memory: float = 10.0,
        spike_memory: float = 20.0,
        spike_at: float = 0.5,
    ) -> "MemoryDataBuilder":
        """Add memory spike pattern."""
        samples = [
            (0.0, base_memory),
            (0.3, base_memory + 1),
            (spike_at, spike_memory),
            (0.7, base_memory + 0.5),
            (1.0, base_memory),
        ]

        for timestamp, rss_mb in samples:
            vms_mb = rss_mb * 1.25
            percent = (rss_mb / 150) * 8
            self.with_sample(timestamp, rss_mb, vms_mb, percent)

        return self

    def build(self) -> Dict[str, Any]:
        """Build memory profiler data."""
        if not self._samples:
            # Default to steady usage if no samples added
            self.with_steady_usage()

        rss_values = [s["rss_mb"] for s in self._samples]

        data = {
            "samples": self._samples,
            "sample_count": len(self._samples),
            "peak_memory_mb": max(rss_values),
            "avg_memory_mb": sum(rss_values) / len(rss_values),
            "memory_delta_mb": (
                rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0.0
            ),
        }

        data.update(self._metadata)
        return data


# Convenience functions
def config() -> ConfigBuilder:
    """Create ConfigBuilder instance."""
    return ConfigBuilder()


def session(config: Optional[ProfileConfig] = None) -> SessionBuilder:
    """Create SessionBuilder instance."""
    return SessionBuilder(config)


def call_data() -> CallDataBuilder:
    """Create CallDataBuilder instance."""
    return CallDataBuilder()


def memory_data() -> MemoryDataBuilder:
    """Create MemoryDataBuilder instance."""
    return MemoryDataBuilder()
