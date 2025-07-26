"""
Configuration builder for creating ProfileConfig instances in tests.

Implements the Builder pattern for clean, readable test data creation
with fluent interface and fail-fast validation.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

from src.pycroscope.core.config import ProfileConfig
from src.pycroscope.core.constants import Defaults, Limits


class ConfigBuilder:
    """
    Builder for creating ProfileConfig instances with fluent interface.

    Provides a clean, readable way to create configuration objects
    for testing with proper validation and no fallbacks.
    """

    def __init__(self):
        self._config_data: Dict[str, Any] = {}
        self._temp_dir: Optional[Path] = None

    def with_line_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure line profiling."""
        self._config_data["line_profiling"] = enabled
        return self

    def with_memory_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure memory profiling."""
        self._config_data["memory_profiling"] = enabled
        return self

    def with_call_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure call profiling."""
        self._config_data["call_profiling"] = enabled
        return self

    def with_sampling_profiling(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure sampling profiling."""
        self._config_data["sampling_profiling"] = enabled
        return self

    def with_output_dir(self, output_dir: Path) -> "ConfigBuilder":
        """Set output directory."""
        self._config_data["output_dir"] = output_dir
        return self

    def with_temp_output_dir(self) -> "ConfigBuilder":
        """Set temporary output directory."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="pycroscope_test_"))
        self._config_data["output_dir"] = self._temp_dir
        return self

    def with_session_name(self, name: str) -> "ConfigBuilder":
        """Set session name."""
        self._config_data["session_name"] = name
        return self

    def with_sampling_interval(self, interval: float) -> "ConfigBuilder":
        """Set sampling interval."""
        if not (
            Limits.MIN_SAMPLING_INTERVAL <= interval <= Limits.MAX_SAMPLING_INTERVAL
        ):
            raise ValueError(
                f"Sampling interval must be between {Limits.MIN_SAMPLING_INTERVAL} and {Limits.MAX_SAMPLING_INTERVAL}"
            )
        self._config_data["sampling_interval"] = interval
        return self

    def with_memory_precision(self, precision: int) -> "ConfigBuilder":
        """Set memory precision."""
        if not (
            Limits.MIN_MEMORY_PRECISION <= precision <= Limits.MAX_MEMORY_PRECISION
        ):
            raise ValueError(
                f"Memory precision must be between {Limits.MIN_MEMORY_PRECISION} and {Limits.MAX_MEMORY_PRECISION}"
            )
        self._config_data["memory_precision"] = precision
        return self

    def with_max_call_depth(self, depth: int) -> "ConfigBuilder":
        """Set maximum call depth."""
        if not (Limits.MIN_CALL_DEPTH <= depth <= Limits.MAX_CALL_DEPTH):
            raise ValueError(
                f"Call depth must be between {Limits.MIN_CALL_DEPTH} and {Limits.MAX_CALL_DEPTH}"
            )
        self._config_data["max_call_depth"] = depth
        return self

    def with_profiler_prefix(self, prefix: str) -> "ConfigBuilder":
        """Set profiler prefix."""
        if len(prefix) > Limits.MAX_PROFILER_PREFIX_LENGTH:
            raise ValueError(
                f"Profiler prefix must be at most {Limits.MAX_PROFILER_PREFIX_LENGTH} characters"
            )
        self._config_data["profiler_prefix"] = prefix
        return self

    def with_reports_enabled(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure report generation."""
        self._config_data["generate_reports"] = enabled
        return self

    def with_visualizations_enabled(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure visualization creation."""
        self._config_data["create_visualizations"] = enabled
        return self

    def with_pattern_analysis_enabled(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure pattern analysis."""
        self._config_data["analyze_patterns"] = enabled
        return self

    def with_thread_isolation(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure thread isolation."""
        self._config_data["use_thread_isolation"] = enabled
        return self

    def with_cleanup_on_exit(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure cleanup on exit."""
        self._config_data["cleanup_on_exit"] = enabled
        return self

    def with_raw_data_saving(self, enabled: bool = True) -> "ConfigBuilder":
        """Configure raw data saving."""
        self._config_data["save_raw_data"] = enabled
        return self

    def with_minimal_overhead(self) -> "ConfigBuilder":
        """Configure for minimal overhead."""
        self._config_data.update(
            {
                "line_profiling": False,
                "memory_profiling": False,
                "call_profiling": True,
                "sampling_profiling": False,
                "sampling_interval": 0.1,
                "memory_precision": 1,
                "max_call_depth": 20,
                "generate_reports": False,
                "create_visualizations": False,
                "analyze_patterns": False,
            }
        )
        return self

    def with_defaults(self) -> "ConfigBuilder":
        """Apply default configuration values."""
        self._config_data.update(
            {
                "line_profiling": Defaults.LINE_PROFILING,
                "memory_profiling": Defaults.MEMORY_PROFILING,
                "call_profiling": Defaults.CALL_PROFILING,
                "sampling_profiling": Defaults.SAMPLING_PROFILING,
                "sampling_interval": Defaults.SAMPLING_INTERVAL,
                "memory_precision": Defaults.MEMORY_PRECISION,
                "max_call_depth": Defaults.MAX_CALL_DEPTH,
                "generate_reports": Defaults.GENERATE_REPORTS,
                "create_visualizations": Defaults.CREATE_VISUALIZATIONS,
                "analyze_patterns": Defaults.ANALYZE_PATTERNS,
                "use_thread_isolation": Defaults.USE_THREAD_ISOLATION,
                "cleanup_on_exit": Defaults.CLEANUP_ON_EXIT,
                "save_raw_data": Defaults.SAVE_RAW_DATA,
            }
        )
        return self

    def build(self) -> ProfileConfig:
        """Build and return ProfileConfig instance."""
        # Ensure output_dir is set
        if "output_dir" not in self._config_data:
            self.with_temp_output_dir()

        return ProfileConfig(**self._config_data)

    def build_minimal_overhead(self) -> ProfileConfig:
        """Build ProfileConfig with minimal overhead configuration."""
        base_config = self.build()
        return base_config.with_minimal_overhead()

    def build_thread_isolated(self, prefix: Optional[str] = None) -> ProfileConfig:
        """Build ProfileConfig with thread isolation."""
        base_config = self.build()
        return base_config.with_thread_isolation(prefix)

    def reset(self) -> "ConfigBuilder":
        """Reset builder to clean state."""
        self._config_data.clear()
        self._temp_dir = None
        return self

    @classmethod
    def create_valid(cls) -> "ConfigBuilder":
        """Create builder with valid default configuration."""
        return cls().with_defaults().with_temp_output_dir()

    @classmethod
    def create_minimal(cls) -> "ConfigBuilder":
        """Create builder with minimal configuration."""
        return cls().with_minimal_overhead().with_temp_output_dir()

    @classmethod
    def create_comprehensive(cls) -> "ConfigBuilder":
        """Create builder with comprehensive profiling enabled."""
        return (
            cls()
            .with_line_profiling(True)
            .with_memory_profiling(True)
            .with_call_profiling(True)
            .with_sampling_profiling(True)
            .with_reports_enabled(True)
            .with_visualizations_enabled(True)
            .with_pattern_analysis_enabled(True)
            .with_temp_output_dir()
        )

    @classmethod
    def create_performance_focused(cls) -> "ConfigBuilder":
        """Create builder focused on performance profiling."""
        return (
            cls()
            .with_line_profiling(True)
            .with_memory_profiling(False)
            .with_call_profiling(True)
            .with_sampling_profiling(True)
            .with_temp_output_dir()
        )

    @classmethod
    def create_memory_focused(cls) -> "ConfigBuilder":
        """Create builder focused on memory profiling."""
        return (
            cls()
            .with_line_profiling(False)
            .with_memory_profiling(True)
            .with_call_profiling(False)
            .with_sampling_profiling(False)
            .with_temp_output_dir()
        )
