"""
Integration module for performance anti-pattern analysis.

Extends Pycroscope's existing ProfileConfig and ProfileSession to seamlessly
integrate pattern analysis capabilities without breaking existing APIs.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from .core.config import ProfileConfig as BaseProfileConfig
from .core.session import ProfileSession
from .analysis.config import AnalysisConfig
from .analysis.orchestrator import (
    create_analysis_orchestrator,
    PerformanceAnalysisOrchestrator,
)
from .analysis.interfaces import PatternType, AnalysisResult


class ExtendedProfileConfig(BaseProfileConfig):
    """
    Extended ProfileConfig that includes analysis configuration.

    Maintains backward compatibility while adding analysis capabilities.
    """

    # Analysis configuration
    analysis_config: Optional[AnalysisConfig] = None

    def with_analysis(
        self,
        enabled: bool = True,
        patterns: Optional[List[PatternType]] = None,
        severity_threshold: str = "medium",
        confidence_threshold: float = 0.7,
        correlate_with_profiling: bool = True,
        **kwargs,
    ) -> "ExtendedProfileConfig":
        """
        Create configuration with analysis enabled.

        Args:
            enabled: Enable pattern analysis
            patterns: List of pattern types to detect (None = defaults)
            severity_threshold: Minimum severity to report
            confidence_threshold: Minimum confidence to report
            correlate_with_profiling: Correlate patterns with profiling data
            **kwargs: Additional analysis configuration options

        Returns:
            New config with analysis enabled
        """
        analysis_config = AnalysisConfig(
            enabled=enabled,
            enabled_patterns=patterns
            or [
                PatternType.NESTED_LOOPS,
                PatternType.QUADRATIC_COMPLEXITY,
                PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
                PatternType.DEAD_CODE,
                PatternType.UNUSED_IMPORTS,
                PatternType.LOW_MAINTAINABILITY_INDEX,
            ],
            severity_threshold=severity_threshold,
            confidence_threshold=confidence_threshold,
            correlate_with_profiling=correlate_with_profiling,
            **kwargs,
        )

        return self.model_copy(
            update={
                "analysis_config": analysis_config,
                "analyze_patterns": enabled,  # Keep existing field in sync
            }
        )

    def with_security_focus(self) -> "ExtendedProfileConfig":
        """Create configuration focused on security-related patterns."""
        return self.with_analysis(
            patterns=[
                PatternType.MEMORY_LEAK_PATTERN,
                PatternType.INEFFICIENT_DATA_STRUCTURE,
                PatternType.UNNECESSARY_COMPUTATION,
            ],
            severity_threshold="low",
            correlate_with_profiling=True,
        )

    def with_performance_focus(self) -> "ExtendedProfileConfig":
        """Create configuration focused on performance patterns."""
        return self.with_analysis(
            patterns=[
                PatternType.NESTED_LOOPS,
                PatternType.QUADRATIC_COMPLEXITY,
                PatternType.EXPONENTIAL_COMPLEXITY,
                PatternType.RECURSIVE_WITHOUT_MEMOIZATION,
            ],
            severity_threshold="medium",
            correlate_with_profiling=True,
            prioritize_hotspots=True,
        )

    def with_maintainability_focus(self) -> "ExtendedProfileConfig":
        """Create configuration focused on maintainability patterns."""
        return self.with_analysis(
            patterns=[
                PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
                PatternType.LOW_MAINTAINABILITY_INDEX,
                PatternType.LONG_FUNCTION,
                PatternType.TOO_MANY_PARAMETERS,
                PatternType.DEAD_CODE,
                PatternType.UNUSED_IMPORTS,
            ],
            severity_threshold="low",
            correlate_with_profiling=False,
        )


class EnhancedProfileSession:
    """
    Enhanced ProfileSession that includes pattern analysis capabilities.

    Wraps the existing ProfileSession to add analysis features without
    modifying the core session logic.
    """

    def __init__(self, base_session: ProfileSession):
        self.base_session = base_session
        self._analysis_orchestrator: Optional[PerformanceAnalysisOrchestrator] = None
        self._analysis_results: Optional[Dict[str, List[AnalysisResult]]] = None

        # Initialize analysis if configured
        analysis_config = getattr(base_session.config, "analysis_config", None)
        if analysis_config and analysis_config.enabled:
            self._analysis_orchestrator = create_analysis_orchestrator(
                base_session, analysis_config
            )

    def __getattr__(self, name):
        """Delegate all other attributes to the base session."""
        return getattr(self.base_session, name)

    def run_analysis(
        self,
        code_files: Optional[List[Path]] = None,
        include_project_files: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Run performance anti-pattern analysis.

        Args:
            code_files: Specific files to analyze (None = auto-discover)
            include_project_files: Include all Python files in project

        Returns:
            Analysis report or None if analysis not enabled
        """
        if not self._analysis_orchestrator:
            return None

        if code_files is None:
            code_files = self._discover_project_files() if include_project_files else []

        # Run analysis
        self._analysis_results = self._analysis_orchestrator.run_analysis(code_files)

        # Generate report
        if self._analysis_results:
            return self._analysis_orchestrator.generate_report(self._analysis_results)

        return None

    def get_analysis_results(self) -> Optional[Dict[str, List[AnalysisResult]]]:
        """Get raw analysis results."""
        return self._analysis_results

    def get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top patterns from analysis results."""
        if not self._analysis_results:
            return []

        all_results = []
        for file_results in self._analysis_results.values():
            all_results.extend(file_results)

        if self._analysis_orchestrator:
            prioritized = self._analysis_orchestrator.prioritize_findings(all_results)
            return [
                {
                    "pattern_type": result.pattern_type.value,
                    "severity": result.severity,
                    "description": result.description,
                    "file": result.location.get("file", ""),
                    "line": result.location.get("line", 0),
                    "suggestion": result.suggestion,
                    "performance_correlated": result.profiling_correlation is not None,
                }
                for result in prioritized[:limit]
            ]

        return []

    def get_patterns_by_type(self, pattern_type: PatternType) -> List[AnalysisResult]:
        """Get all patterns of a specific type."""
        if not self._analysis_results:
            return []

        matching_patterns = []
        for file_results in self._analysis_results.values():
            matching_patterns.extend(
                [
                    result
                    for result in file_results
                    if result.pattern_type == pattern_type
                ]
            )

        return matching_patterns

    def get_hotspot_correlated_patterns(self) -> List[AnalysisResult]:
        """Get patterns that correlate with performance hotspots."""
        if not self._analysis_results:
            return []

        correlated_patterns = []
        for file_results in self._analysis_results.values():
            correlated_patterns.extend(
                [
                    result
                    for result in file_results
                    if result.profiling_correlation is not None
                ]
            )

        return correlated_patterns

    def _discover_project_files(self) -> List[Path]:
        """Auto-discover Python files in the project."""
        project_files = []

        # Start from output directory's parent or current working directory
        if self.base_session.config.output_dir:
            search_root = self.base_session.config.output_dir.parent
        else:
            search_root = Path.cwd()

        # Find Python files, excluding common directories
        exclude_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
        }

        for python_file in search_root.rglob("*.py"):
            # Skip files in excluded directories
            if any(excluded in python_file.parts for excluded in exclude_dirs):
                continue

            # Skip files that are too large (> 1MB)
            try:
                if python_file.stat().st_size > 1024 * 1024:
                    continue
            except OSError:
                continue

            project_files.append(python_file)

        return project_files[:100]  # Limit to prevent overwhelming analysis


def profile_with_analysis(
    func=None,
    *,
    # Standard profiling options
    line_profiling: bool = True,
    memory_profiling: bool = True,
    call_profiling: bool = True,
    output_dir: Optional[Path] = None,
    # Analysis options
    analyze_patterns: bool = True,
    analysis_patterns: Optional[List[PatternType]] = None,
    analysis_severity_threshold: str = "medium",
    analysis_confidence_threshold: float = 0.7,
    correlate_with_profiling: bool = True,
    **kwargs,
):
    """
    Enhanced profile function with integrated pattern analysis.

    This provides the same API as the original profile function but adds
    comprehensive pattern analysis capabilities.

    Example usage:

    @profile_with_analysis(
        analyze_patterns=True,
        analysis_patterns=[PatternType.NESTED_LOOPS, PatternType.DEAD_CODE],
        correlate_with_profiling=True
    )
    def my_function():
        # Code to profile and analyze
        pass

    # Or as context manager:
    with profile_with_analysis(analyze_patterns=True) as session:
        # Code to profile
        pass

        # Run analysis
        analysis_report = session.run_analysis()
        print(f"Found {len(session.get_top_patterns())} top issues")
    """
    from .core.session import ProfileSession
    from .application.services import ProfilingService

    # Create extended configuration
    config = ExtendedProfileConfig(
        line_profiling=line_profiling,
        memory_profiling=memory_profiling,
        call_profiling=call_profiling,
        output_dir=output_dir or Path("./profiling_results"),
        analyze_patterns=analyze_patterns,
        **kwargs,
    )

    # Add analysis configuration if enabled
    if analyze_patterns:
        config = config.with_analysis(
            enabled=True,
            patterns=analysis_patterns,
            severity_threshold=analysis_severity_threshold,
            confidence_threshold=analysis_confidence_threshold,
            correlate_with_profiling=correlate_with_profiling,
        )

    # Use existing profiling service but enhance the session
    profiling_service = ProfilingService()

    if func is not None:
        # Decorator usage
        def wrapper(*args, **kwargs):
            base_session = profiling_service.start_profiling(**config.model_dump())
            enhanced_session = EnhancedProfileSession(base_session)

            try:
                result = func(*args, **kwargs)
                profiling_service.stop_profiling(base_session)

                # Run analysis if enabled
                if analyze_patterns:
                    enhanced_session.run_analysis()

                return result
            except Exception as e:
                profiling_service.stop_profiling(base_session)
                raise e

        return wrapper
    else:
        # Context manager usage
        class AnalysisProfileContext:
            def __init__(self):
                self.base_session = None
                self.enhanced_session = None

            def __enter__(self):
                self.base_session = profiling_service.start_profiling(
                    **config.model_dump()
                )
                self.enhanced_session = EnhancedProfileSession(self.base_session)
                return self.enhanced_session

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.base_session:
                    profiling_service.stop_profiling(self.base_session)

        return AnalysisProfileContext()
