"""
Configuration model for Pycroscope profiling and analysis.

Uses Pydantic V2 for robust validation and type safety.
Integrates pattern analysis as a core feature alongside profiling.
"""

from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, Dict, Any, List


class ProfileConfig(BaseModel):
    """
    Configuration for profiling and pattern analysis sessions.

    Combines profiling configuration with integrated pattern analysis
    for comprehensive performance and code quality assessment.
    """

    # Profiling configuration
    line_profiling: bool = Field(
        default=True, description="Enable line-by-line profiling"
    )
    memory_profiling: bool = Field(
        default=True, description="Enable memory usage profiling"
    )
    call_profiling: bool = Field(
        default=True, description="Enable function call profiling"
    )

    # Output configuration
    output_dir: Optional[Path] = Field(
        default=None, description="Directory to save profiling and analysis results"
    )
    session_name: Optional[str] = Field(
        default=None, max_length=100, description="Name for this profiling session"
    )
    save_raw_data: bool = Field(
        default=True, description="Save raw profiling data to files"
    )

    # Sampling and precision
    sampling_interval: float = Field(
        default=0.01, gt=0.0, description="Sampling interval in seconds"
    )
    memory_precision: int = Field(
        default=3, ge=1, le=6, description="Memory measurement precision"
    )
    max_call_depth: int = Field(
        default=50, gt=0, description="Maximum call stack depth to profile"
    )

    # Analysis and reporting
    generate_reports: bool = Field(
        default=True, description="Generate comprehensive analysis reports"
    )
    create_visualizations: bool = Field(
        default=True, description="Create charts and visualizations"
    )

    # Pattern Analysis Configuration (Core Feature)
    analyze_patterns: bool = Field(
        default=True, description="Enable performance anti-pattern analysis"
    )

    # Pattern Detection Settings
    pattern_complexity_threshold: int = Field(
        default=10,
        gt=0,
        description="Cyclomatic complexity threshold for pattern detection",
    )
    pattern_maintainability_threshold: float = Field(
        default=20.0, gt=0.0, description="Maintainability index threshold"
    )
    pattern_severity_threshold: str = Field(
        default="medium",
        description="Minimum severity threshold for reporting patterns",
    )
    pattern_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for pattern reporting",
    )

    # Pattern Analysis Focus
    detect_nested_loops: bool = Field(
        default=True, description="Detect nested loop anti-patterns"
    )
    detect_dead_code: bool = Field(
        default=True, description="Detect unused code and imports"
    )
    detect_complexity_issues: bool = Field(
        default=True, description="Detect high complexity functions"
    )
    detect_maintainability_issues: bool = Field(
        default=True, description="Detect maintainability problems"
    )

    # Function Analysis Thresholds
    max_function_lines: int = Field(
        default=50, gt=0, description="Maximum recommended function length"
    )
    max_function_parameters: int = Field(
        default=5, gt=0, description="Maximum recommended function parameters"
    )

    # Pattern-Profiling Integration
    correlate_patterns_with_profiling: bool = Field(
        default=True, description="Correlate detected patterns with profiling hotspots"
    )
    prioritize_hotspot_patterns: bool = Field(
        default=True, description="Prioritize patterns found in performance hotspots"
    )
    hotspot_correlation_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Threshold for hotspot correlation"
    )

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
            raise ValidationError("session_name must be at most 100 characters")
        return v

    @field_validator("profiler_prefix")
    @classmethod
    def validate_profiler_prefix(cls, v: str) -> str:
        """Validate profiler prefix length."""
        if len(v) > 20:
            raise ValidationError("profiler_prefix must be at most 20 characters")
        return v

    @field_validator("pattern_severity_threshold")
    @classmethod
    def validate_pattern_severity_threshold(cls, v: str) -> str:
        """Validate pattern severity threshold."""
        valid_severities = ["low", "medium", "high", "critical"]
        if v not in valid_severities:
            raise ValidationError(
                f"Pattern severity threshold must be one of {valid_severities}"
            )
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

        return profilers

    @property
    def enabled_pattern_types(self) -> List[str]:
        """Get list of enabled pattern detection types."""
        patterns = []
        if self.analyze_patterns:
            if self.detect_nested_loops:
                patterns.extend(["nested_loops", "quadratic_complexity"])
            if self.detect_dead_code:
                patterns.extend(["dead_code", "unused_imports"])
            if self.detect_complexity_issues:
                patterns.extend(
                    ["high_cyclomatic_complexity", "recursive_without_memoization"]
                )
            if self.detect_maintainability_issues:
                patterns.extend(
                    [
                        "low_maintainability_index",
                        "long_function",
                        "too_many_parameters",
                    ]
                )
        return patterns

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
                "generate_reports": False,
                "create_visualizations": False,
                "analyze_patterns": False,  # Disable pattern analysis for minimal overhead
                "memory_precision": 1,
                "max_call_depth": 10,
            }
        )

    def with_performance_focus(self) -> "ProfileConfig":
        """
        Create configuration focused on performance analysis.

        Emphasizes pattern detection for algorithmic complexity and performance hotspots.
        """
        return self.model_copy(
            update={
                "analyze_patterns": True,
                "detect_nested_loops": True,
                "detect_complexity_issues": True,
                "detect_dead_code": False,
                "detect_maintainability_issues": False,
                "correlate_patterns_with_profiling": True,
                "prioritize_hotspot_patterns": True,
                "pattern_severity_threshold": "medium",
            }
        )

    def with_maintainability_focus(self) -> "ProfileConfig":
        """
        Create configuration focused on code maintainability.

        Emphasizes pattern detection for code quality and maintainability issues.
        """
        return self.model_copy(
            update={
                "analyze_patterns": True,
                "detect_nested_loops": False,
                "detect_complexity_issues": True,
                "detect_dead_code": True,
                "detect_maintainability_issues": True,
                "correlate_patterns_with_profiling": False,
                "pattern_severity_threshold": "low",
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
            ConfigurationError: If prefix is None or empty
        """
        if not prefix:
            from .exceptions import ConfigurationError

            raise ConfigurationError(
                "Thread isolation requires a non-empty prefix", config_key="prefix"
            )

        return self.model_copy(
            update={
                "use_thread_isolation": True,
                "profiler_prefix": prefix,
            }
        )
