"""
Configuration model for Pycroscope profiling and analysis.

Uses Pydantic V2 for robust validation and type safety.
Integrates pattern analysis as a core feature alongside profiling.
"""

from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, Dict, Any, List
from enum import Enum


class PatternType(Enum):
    """Types of patterns that can be detected."""

    # Performance Anti-Patterns
    NESTED_LOOPS = "nested_loops"
    INEFFICIENT_DATA_STRUCTURE = "inefficient_data_structure"
    UNNECESSARY_COMPUTATION = "unnecessary_computation"
    MEMORY_LEAK_PATTERN = "memory_leak_pattern"

    # Algorithmic Complexity Issues
    QUADRATIC_COMPLEXITY = "quadratic_complexity"
    EXPONENTIAL_COMPLEXITY = "exponential_complexity"
    RECURSIVE_WITHOUT_MEMOIZATION = "recursive_without_memoization"

    # Code Quality Issues
    DEAD_CODE = "dead_code"
    UNUSED_IMPORTS = "unused_imports"
    DUPLICATE_CODE = "duplicate_code"
    HIGH_CYCLOMATIC_COMPLEXITY = "high_cyclomatic_complexity"

    # Maintainability Issues
    LOW_MAINTAINABILITY_INDEX = "low_maintainability_index"
    LONG_FUNCTION = "long_function"
    TOO_MANY_PARAMETERS = "too_many_parameters"

    # Scientific Computing Anti-Patterns (Sequential CPU Focus)
    MISSED_VECTORIZATION = "missed_vectorization"
    INEFFICIENT_ARRAY_OPERATIONS = "inefficient_array_operations"
    SUBOPTIMAL_MATRIX_OPERATIONS = "suboptimal_matrix_operations"
    NON_CONTIGUOUS_MEMORY_ACCESS = "non_contiguous_memory_access"
    UNNECESSARY_ARRAY_COPY = "unnecessary_array_copy"
    INEFFICIENT_BROADCASTING = "inefficient_broadcasting"
    SCALAR_ARRAY_OPERATIONS = "scalar_array_operations"
    WRONG_DTYPE_USAGE = "wrong_dtype_usage"
    INEFFICIENT_ARRAY_CONCATENATION = "inefficient_array_concatenation"
    SUBOPTIMAL_LINEAR_ALGEBRA = "suboptimal_linear_algebra"


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
    enabled_patterns: List[PatternType] = Field(
        default_factory=lambda: [
            # Performance Anti-Patterns
            PatternType.NESTED_LOOPS,
            PatternType.INEFFICIENT_DATA_STRUCTURE,
            PatternType.UNNECESSARY_COMPUTATION,
            PatternType.MEMORY_LEAK_PATTERN,
            # Algorithmic Complexity Issues
            PatternType.QUADRATIC_COMPLEXITY,
            PatternType.EXPONENTIAL_COMPLEXITY,
            PatternType.RECURSIVE_WITHOUT_MEMOIZATION,
            # Code Quality Issues
            PatternType.DEAD_CODE,
            PatternType.UNUSED_IMPORTS,
            PatternType.DUPLICATE_CODE,
            PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
            # Maintainability Issues
            PatternType.LOW_MAINTAINABILITY_INDEX,
            PatternType.LONG_FUNCTION,
            PatternType.TOO_MANY_PARAMETERS,
            # Scientific Computing Anti-Patterns
            PatternType.MISSED_VECTORIZATION,
            PatternType.INEFFICIENT_ARRAY_OPERATIONS,
            PatternType.SUBOPTIMAL_MATRIX_OPERATIONS,
            PatternType.NON_CONTIGUOUS_MEMORY_ACCESS,
            PatternType.UNNECESSARY_ARRAY_COPY,
            PatternType.INEFFICIENT_BROADCASTING,
            PatternType.SCALAR_ARRAY_OPERATIONS,
            PatternType.WRONG_DTYPE_USAGE,
            PatternType.INEFFICIENT_ARRAY_CONCATENATION,
            PatternType.SUBOPTIMAL_LINEAR_ALGEBRA,
        ],
        description="List of pattern types to detect (all patterns enabled by default)",
    )

    # Analysis Thresholds
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
    max_results_per_file: int = Field(
        default=50, gt=0, description="Maximum number of results per file"
    )

    # Analysis Behavior Settings
    correlate_patterns_with_profiling: bool = Field(
        default=True, description="Correlate detected patterns with profiling hotspots"
    )
    include_suggestions: bool = Field(
        default=True, description="Include improvement suggestions in results"
    )
    prioritize_hotspot_patterns: bool = Field(
        default=True, description="Prioritize patterns found in performance hotspots"
    )
    hotspot_correlation_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Threshold for hotspot correlation"
    )

    # Code Quality Thresholds
    pattern_complexity_threshold: int = Field(
        default=10,
        gt=0,
        description="Cyclomatic complexity threshold for pattern detection",
    )
    pattern_maintainability_threshold: float = Field(
        default=20.0, gt=0.0, description="Maintainability index threshold"
    )
    max_function_lines: int = Field(
        default=50, gt=0, description="Maximum recommended function length"
    )
    max_function_parameters: int = Field(
        default=5, gt=0, description="Maximum recommended function parameters"
    )

    # Dead Code Detection Settings
    exclude_test_files: bool = Field(
        default=True, description="Exclude test files from dead code detection"
    )
    test_file_patterns: List[str] = Field(
        default_factory=lambda: ["*test*.py", "*_test.py", "test_*.py"],
        description="Patterns for identifying test files",
    )

    # Analysis Output Settings
    generate_detailed_analysis_report: bool = Field(
        default=True, description="Generate detailed analysis report"
    )
    save_intermediate_analysis_results: bool = Field(
        default=False, description="Save intermediate analysis results for debugging"
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

    @field_validator("enabled_patterns")
    @classmethod
    def validate_enabled_patterns(cls, v: List[PatternType]) -> List[PatternType]:
        """Validate enabled patterns - all patterns are always included."""
        # Always ensure all patterns are enabled
        all_patterns = list(PatternType)
        return all_patterns

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
        if not self.analyze_patterns:
            return []

        return [pattern.value for pattern in self.enabled_patterns]

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

    def is_pattern_enabled(self, pattern_type: PatternType) -> bool:
        """Check if a specific pattern type is enabled."""
        return pattern_type in self.enabled_patterns

    def get_severity_weight(self, severity: str) -> int:
        """Get numeric weight for severity level."""
        weights = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return weights.get(severity, 0)

    def should_report_analysis_result(self, severity: str, confidence: float) -> bool:
        """Check if an analysis result should be reported based on thresholds."""
        severity_ok = self.get_severity_weight(severity) >= self.get_severity_weight(
            self.pattern_severity_threshold
        )
        confidence_ok = confidence >= self.pattern_confidence_threshold
        return severity_ok and confidence_ok

    def with_performance_focus(self) -> "ProfileConfig":
        """
        Create configuration focused on performance analysis.

        All patterns are enabled, but emphasizes performance correlation and medium severity.
        """
        return self.model_copy(
            update={
                "analyze_patterns": True,
                "correlate_patterns_with_profiling": True,
                "prioritize_hotspot_patterns": True,
                "pattern_severity_threshold": "medium",
                "pattern_confidence_threshold": 0.8,
            }
        )

    def with_maintainability_focus(self) -> "ProfileConfig":
        """
        Create configuration focused on code maintainability.

        All patterns are enabled, but emphasizes maintainability with lower severity threshold.
        """
        return self.model_copy(
            update={
                "analyze_patterns": True,
                "correlate_patterns_with_profiling": False,
                "pattern_severity_threshold": "low",
                "pattern_confidence_threshold": 0.6,
                "generate_detailed_analysis_report": True,
            }
        )

    def with_security_focus(self) -> "ProfileConfig":
        """
        Create configuration focused on security-related patterns.

        All patterns are enabled, but emphasizes security with low severity threshold.
        """
        return self.model_copy(
            update={
                "analyze_patterns": True,
                "correlate_patterns_with_profiling": True,
                "pattern_severity_threshold": "low",
                "pattern_confidence_threshold": 0.7,
                "prioritize_hotspot_patterns": True,
            }
        )

    def with_comprehensive_analysis(self) -> "ProfileConfig":
        """
        Create configuration for comprehensive analysis.

        All patterns enabled with maximum sensitivity and detailed reporting.
        """
        return self.model_copy(
            update={
                "analyze_patterns": True,
                "correlate_patterns_with_profiling": True,
                "prioritize_hotspot_patterns": True,
                "pattern_severity_threshold": "low",
                "pattern_confidence_threshold": 0.5,
                "generate_detailed_analysis_report": True,
                "save_intermediate_analysis_results": True,
                "include_suggestions": True,
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
