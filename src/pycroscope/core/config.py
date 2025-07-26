"""
ProfileConfig: Configuration management for profiling sessions.

Centralizes all profiling configuration with validation and defaults,
including profiling Pycroscope itself without conflicts or special cases.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import tempfile

from pydantic import BaseModel, Field, field_validator, model_validator

from .exceptions import ConfigurationError


class ProfileConfig(BaseModel):
    """
    Configuration for a profiling session.

    Provides centralized configuration management with validation
    and support for different profiling scenarios including
    profiling Pycroscope itself.
    """

    # Core profiling options
    line_profiling: bool = Field(
        default=True, description="Enable line-by-line profiling"
    )
    memory_profiling: bool = Field(
        default=True, description="Enable memory usage profiling"
    )
    call_profiling: bool = Field(
        default=True, description="Enable function call profiling"
    )
    sampling_profiling: bool = Field(
        default=False, description="Enable sampling profiler"
    )

    # Output configuration
    output_dir: Path = Field(description="Directory for output files")
    session_name: Optional[str] = Field(
        default=None, description="Name for this profiling session"
    )
    save_raw_data: bool = Field(
        default=True, description="Save raw profiling data to files"
    )

    # Profiler-specific settings
    sampling_interval: float = Field(
        default=0.1, ge=0.001, le=1.0, description="Sampling interval in seconds"
    )
    memory_precision: int = Field(
        default=1, ge=1, le=6, description="Memory measurement precision"
    )
    max_call_depth: int = Field(
        default=20, ge=1, le=1000, description="Maximum call stack depth to profile"
    )

    # Processing options
    generate_reports: bool = Field(
        default=True, description="Generate analysis reports"
    )
    create_visualizations: bool = Field(
        default=True, description="Create visualization charts"
    )
    analyze_patterns: bool = Field(default=True, description="Perform pattern analysis")

    # Isolation and safety
    profiler_prefix: str = Field(
        default="pycroscope",
        max_length=20,
        description="Prefix for profiler output files",
    )
    use_thread_isolation: bool = Field(
        default=True, description="Use thread isolation for profilers"
    )
    cleanup_on_exit: bool = Field(
        default=True, description="Clean up temporary files on exit"
    )

    # Advanced configuration
    extra_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional configuration options"
    )

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Optional[Path]) -> Path:
        """Ensure output_dir is provided and is a valid Path object."""
        if v is None:
            from .exceptions import ConfigurationError

            raise ConfigurationError(
                "output_dir is required and cannot be None", config_key="output_dir"
            )
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("session_name")
    @classmethod
    def validate_session_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate session name length."""
        if v is not None and len(v) > 100:
            raise ValueError("session_name must be at most 100 characters")
        return v

    @field_validator("profiler_prefix")
    @classmethod
    def validate_profiler_prefix(cls, v: str) -> str:
        """Validate profiler prefix length."""
        if len(v) > 20:
            raise ValueError("profiler_prefix must be at most 20 characters")
        return v

    @property
    def enabled_profilers(self) -> List[str]:
        """Get list of enabled profiler types."""
        profilers = []
        if self.line_profiling:
            profilers.append("line")
        if self.memory_profiling:
            profilers.append("memory")
        if self.call_profiling:
            profilers.append("call")
        if self.sampling_profiling:
            profilers.append("sampling")
        return profilers

    def with_minimal_overhead(self) -> "ProfileConfig":
        """
        Create configuration optimized for minimal overhead.

        Enables only essential profiling and disables expensive operations
        or when profiling code that itself uses profiling.
        """
        return self.model_copy(
            update={
                "line_profiling": False,
                "memory_profiling": False,
                "call_profiling": True,  # Keep only essential call profiling
                "sampling_profiling": False,
                "generate_reports": False,
                "create_visualizations": False,
                "analyze_patterns": False,
                "memory_precision": 1,
                "max_call_depth": 10,
            }
        )

    def with_thread_isolation(self, prefix: Optional[str]) -> "ProfileConfig":
        """
        Create configuration with thread isolation.

        Args:
            prefix: Profiler prefix for isolation (required)

        Returns:
            New configuration with thread isolation enabled

        Raises:
            ConfigurationError: If prefix is None
        """
        if prefix is None:
            raise ConfigurationError("Thread isolation requires a profiler prefix")

        return self.model_copy(
            update={"use_thread_isolation": True, "profiler_prefix": prefix}
        )
