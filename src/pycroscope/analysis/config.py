"""
Configuration for performance anti-pattern analysis.

Integrates with Pycroscope's existing Pydantic-based configuration system
while providing specific settings for pattern analysis and detection.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path

from .interfaces import PatternType


class AnalysisConfig(BaseModel):
    """Configuration for performance anti-pattern analysis."""

    # Enable/disable analysis
    enabled: bool = Field(
        default=False, description="Enable performance anti-pattern analysis"
    )

    # Pattern detection settings
    enabled_patterns: List[PatternType] = Field(
        default_factory=lambda: [
            PatternType.NESTED_LOOPS,
            PatternType.QUADRATIC_COMPLEXITY,
            PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
            PatternType.DEAD_CODE,
            PatternType.UNUSED_IMPORTS,
        ],
        description="List of pattern types to detect",
    )

    # Thresholds
    severity_threshold: str = Field(
        default="medium", description="Minimum severity threshold for reporting"
    )

    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for reporting",
    )

    max_results_per_file: int = Field(
        default=50, gt=0, description="Maximum number of results per file"
    )

    # Integration settings
    correlate_with_profiling: bool = Field(
        default=True, description="Whether to correlate patterns with profiling data"
    )

    include_suggestions: bool = Field(
        default=True, description="Include improvement suggestions in results"
    )

    # Complexity analysis settings
    complexity_threshold: int = Field(
        default=10, gt=0, description="Cyclomatic complexity threshold"
    )

    maintainability_threshold: float = Field(
        default=20.0, gt=0.0, description="Maintainability index threshold"
    )

    # Function length thresholds
    max_function_lines: int = Field(
        default=50, gt=0, description="Maximum recommended function length in lines"
    )

    max_function_parameters: int = Field(
        default=5, gt=0, description="Maximum recommended function parameters"
    )

    # Dead code detection settings
    exclude_test_files: bool = Field(
        default=True, description="Exclude test files from dead code detection"
    )

    test_file_patterns: List[str] = Field(
        default_factory=lambda: ["*test*.py", "*_test.py", "test_*.py"],
        description="Patterns for identifying test files",
    )

    # Performance correlation settings
    hotspot_correlation_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for correlating patterns with performance hotspots",
    )

    prioritize_hotspots: bool = Field(
        default=True, description="Prioritize patterns found in performance hotspots"
    )

    # Output settings
    generate_detailed_report: bool = Field(
        default=True, description="Generate detailed analysis report"
    )

    save_intermediate_results: bool = Field(
        default=False, description="Save intermediate analysis results for debugging"
    )

    # External tool configuration
    external_tools: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration for external analysis tools"
    )

    @validator("severity_threshold")
    def validate_severity_threshold(cls, v):
        """Validate severity threshold value."""
        valid_severities = ["low", "medium", "high", "critical"]
        if v not in valid_severities:
            raise ValueError(f"Severity threshold must be one of {valid_severities}")
        return v

    @validator("enabled_patterns")
    def validate_enabled_patterns(cls, v):
        """Validate enabled patterns."""
        if not v:
            raise ValueError("At least one pattern type must be enabled")
        return v

    def is_pattern_enabled(self, pattern_type: PatternType) -> bool:
        """Check if a specific pattern type is enabled."""
        return pattern_type in self.enabled_patterns

    def get_severity_weight(self, severity: str) -> int:
        """Get numeric weight for severity level."""
        weights = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return weights.get(severity, 0)

    def should_report_result(self, severity: str, confidence: float) -> bool:
        """Check if a result should be reported based on thresholds."""
        severity_ok = self.get_severity_weight(severity) >= self.get_severity_weight(
            self.severity_threshold
        )
        confidence_ok = confidence >= self.confidence_threshold
        return severity_ok and confidence_ok

    def with_external_tool_config(self, tool_name: str, **config) -> "AnalysisConfig":
        """Create a new config with additional external tool configuration."""
        new_config = self.copy()
        new_config.external_tools[tool_name] = config
        return new_config
