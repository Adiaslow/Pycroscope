"""
Profiler orchestration and management.

Coordinates multiple profilers with conflict resolution and result aggregation
following clean architecture principles.
"""

from typing import Dict, Any, List, Set, Optional
from datetime import datetime
import threading
import time
from pathlib import Path

from ...core.config import ProfileConfig
from ...core.session import ProfileSession, ProfileResult
from ...core.factories import get_profiler_factory
from ...core.exceptions import ProfilerError, ConfigurationError
from ..reporting.report_generator import ReportGenerator
from ..visualization.chart_generator import ChartGenerator


class ProfilerOrchestra:
    """
    Coordinates multiple profilers for comprehensive performance analysis.

    Manages profiler lifecycle, conflict resolution, and result aggregation
    with graceful degradation for conflicting profilers.
    """

    def __init__(self, session: ProfileSession):
        self.session = session
        self._running_profilers: Dict[str, Any] = {}
        self._profiler_results: Dict[str, Dict[str, Any]] = {}
        self._sequential_profilers: List[str] = []
        self._lock = threading.Lock()

    @property
    def is_profiling_active(self) -> bool:
        """Check if any profilers are currently active."""
        return len(self._running_profilers) > 0

    @property
    def active_profilers(self) -> List[str]:
        """Get list of currently active profiler types."""
        return list(self._running_profilers.keys())

    def is_active(self, profiler_type: str) -> bool:
        """Check if specific profiler type is active."""
        return profiler_type in self._running_profilers

    def start_profiling(self) -> List[str]:
        """
        Start all configured profilers with robust conflict resolution.

        Uses multiple sequential runs for sys.settrace() profilers to handle
        nested profiling scenarios (e.g., Pycroscope profiling Pycroscope).

        Returns:
            List of successfully started profiler types

        Raises:
            ProfilerError: If any profiler fails to start
        """
        enabled_profilers = self.session.config.enabled_profilers
        if not enabled_profilers:
            raise ConfigurationError("No profilers are enabled")

        self.session.start()

        # Check if we're already being profiled (nested profiling scenario)
        import sys

        current_trace = sys.gettrace()

        if current_trace is not None:
            print("🔍 Detected active trace function - we're being profiled!")
            print(f"   Current tracer: {type(current_trace).__name__}")
            print(
                "[GRACEFUL] Graceful degradation: Disabling trace-based profilers to avoid conflicts"
            )

            # Filter out conflicting profilers when we're being profiled
            trace_profilers = ["line", "call"]
            safe_profilers = [p for p in enabled_profilers if p not in trace_profilers]

            if not safe_profilers:
                print(
                    "   [WARNING] All requested profilers conflict with active tracer"
                )
                print("   [INFO] Proceeding with non-trace profilers only")
                # At minimum, try memory profiler as it doesn't use sys.settrace()
                safe_profilers = [p for p in ["memory"] if p in enabled_profilers]

            if safe_profilers:
                print(f"   [OK] Using safe profilers: {', '.join(safe_profilers)}")
                return self._simultaneous_profiling(safe_profilers)
            else:
                print(
                    "   [ERROR] No safe profilers available - profiling disabled in nested context"
                )
                return []

        # Use robust trace multiplexer to run all profilers simultaneously
        print(
            "[ROBUST] Using robust trace multiplexer - all profilers run simultaneously"
        )
        return self._simultaneous_profiling(enabled_profilers)

    def _simultaneous_profiling(self, enabled_profilers: List[str]) -> List[str]:
        """Start all profilers simultaneously with robust trace multiplexing."""
        started_profilers = []
        registry = get_profiler_factory()

        # Start profilers without the context manager - tracing must stay active
        # until stop_profiling() is called
        for profiler_type in enabled_profilers:
            factory = registry.get_factory_for_type(profiler_type)
            profiler = factory.create(self.session.config.model_dump())

            print(f"[OK] Started {profiler_type} profiler")

            # Start profiler - let any exceptions bubble up immediately
            profiler.start()

            self._running_profilers[profiler_type] = profiler
            started_profilers.append(profiler_type)

        print(
            f"[SUCCESS] Successfully started {len(started_profilers)} profilers: {', '.join(started_profilers)}"
        )
        return started_profilers

    def _sequential_profiling(self, enabled_profilers: List[str]) -> List[str]:
        """Run conflicting profilers sequentially to avoid conflicts."""
        print("🔄 Sequential profiling: Running conflicting profilers one at a time")

        # Separate conflicting profilers from compatible ones
        conflicting = ["line", "call"]
        compatible = [p for p in enabled_profilers if p not in conflicting]
        sequential_profilers = [p for p in conflicting if p in enabled_profilers]

        # Start compatible profilers first (they can run together)
        if compatible:
            started_compatible = self._simultaneous_profiling(compatible)
            print(f"[OK] Compatible profilers running: {', '.join(started_compatible)}")

        # Sequential profilers will be handled later in stop_profiling to avoid sys.settrace() conflicts
        # Store them for true sequential execution after compatible profilers finish
        self._sequential_profilers = sequential_profilers
        if sequential_profilers:
            print(
                f"[QUEUE] Sequential profilers queued: {', '.join(sequential_profilers)} (will run after compatible profilers)"
            )

        all_profilers = (compatible if compatible else []) + sequential_profilers
        print(f"[ACTIVE] All profilers active: {', '.join(all_profilers)}")

        return all_profilers

    def stop_profiling(self) -> Dict[str, ProfileResult]:
        """
        Stop all active profilers and collect results.

        For sequential profiling, this runs conflicting profilers one at a time
        to avoid sys.settrace() conflicts.

        Returns:
            Dictionary of profiler results by type
        """
        results = {}

        # Handle simultaneous profilers first (already running)
        if self._running_profilers:
            for profiler_type, profiler_instance in self._running_profilers.items():
                end_time = datetime.now()
                profiler_data = profiler_instance.stop()

                # Create ProfileResult
                start_time = datetime.now()
                if (
                    hasattr(profiler_instance, "start_time")
                    and profiler_instance.start_time
                ):
                    # Convert float timestamp to datetime
                    import time

                    start_time = datetime.fromtimestamp(
                        time.time()
                        - (time.perf_counter() - profiler_instance.start_time)
                    )

                result = ProfileResult(
                    profiler_type=profiler_type,
                    data=profiler_data,
                    start_time=start_time,
                    end_time=end_time,
                    success=True,
                )

                self.session.add_result(profiler_type, result)
                results[profiler_type] = result

        # All profilers now run simultaneously with robust trace multiplexer
        # No need for complex sequential execution

        if not results:
            return {}

        # Clear running profilers
        self._running_profilers.clear()

        # Clean up trace multiplexer now that all profilers are stopped
        from .trace_multiplexer import _trace_multiplexer

        if _trace_multiplexer._is_multiplexer_active:
            _trace_multiplexer._deactivate()

        # Clean up sequential profilers if they were configured
        if hasattr(self, "_sequential_profilers"):
            delattr(self, "_sequential_profilers")

        # Complete the session
        self.session.complete()

        # Generate outputs if enabled
        if (
            self.session.config.generate_reports
            or self.session.config.create_visualizations
        ):
            self._generate_outputs()

        return results

    def _generate_outputs(self) -> None:
        """Generate reports, visualizations, and pattern analysis if enabled."""

        # Run pattern analysis first if enabled
        analysis_results = None
        if self.session.config.analyze_patterns:
            analysis_results = self._run_pattern_analysis()

        if self.session.config.generate_reports:
            report_generator = ReportGenerator(self.session)
            report_path = report_generator.generate_comprehensive_report(
                pattern_analysis_results=analysis_results
            )
            print(f"[REPORT] Generated comprehensive report: {report_path}")

        if self.session.config.create_visualizations:
            chart_generator = ChartGenerator(self.session)
            charts = chart_generator.generate_all_charts()
            if charts:
                print(f"[CHARTS] Generated {len(charts)} visualization charts:")
                for chart_name, chart_path in charts.items():
                    print(f"   - {chart_name}: {chart_path}")
            else:
                print("[CHARTS] No charts generated (no compatible profiling data)")

    def _run_pattern_analysis(self) -> Optional[Dict[str, Any]]:
        """Run pattern analysis on the profiled code."""
        from ...analysis.config import AnalysisConfig
        from ...analysis.orchestrator import create_analysis_orchestrator
        from ...analysis.interfaces import PatternType

        # Convert ProfileConfig to AnalysisConfig
        enabled_patterns = []
        pattern_map = {
            "nested_loops": PatternType.NESTED_LOOPS,
            "quadratic_complexity": PatternType.QUADRATIC_COMPLEXITY,
            "dead_code": PatternType.DEAD_CODE,
            "unused_imports": PatternType.UNUSED_IMPORTS,
            "high_cyclomatic_complexity": PatternType.HIGH_CYCLOMATIC_COMPLEXITY,
            "recursive_without_memoization": PatternType.RECURSIVE_WITHOUT_MEMOIZATION,
            "low_maintainability_index": PatternType.LOW_MAINTAINABILITY_INDEX,
            "long_function": PatternType.LONG_FUNCTION,
            "too_many_parameters": PatternType.TOO_MANY_PARAMETERS,
        }

        for pattern_name in self.session.config.enabled_pattern_types:
            if pattern_name in pattern_map:
                enabled_patterns.append(pattern_map[pattern_name])

        if not enabled_patterns:
            return None

        # Create analysis configuration from profile configuration
        analysis_config = AnalysisConfig(
            enabled=True,
            enabled_patterns=enabled_patterns,
            severity_threshold=self.session.config.pattern_severity_threshold,
            confidence_threshold=self.session.config.pattern_confidence_threshold,
            correlate_with_profiling=self.session.config.correlate_patterns_with_profiling,
            complexity_threshold=self.session.config.pattern_complexity_threshold,
            maintainability_threshold=self.session.config.pattern_maintainability_threshold,
            max_function_lines=self.session.config.max_function_lines,
            max_function_parameters=self.session.config.max_function_parameters,
            prioritize_hotspots=self.session.config.prioritize_hotspot_patterns,
            hotspot_correlation_threshold=self.session.config.hotspot_correlation_threshold,
        )

        # Create and run analysis orchestrator
        analysis_orchestrator = create_analysis_orchestrator(
            self.session, analysis_config
        )

        # Extract files that were actually profiled from session data
        code_files = self._extract_profiled_files()
        if not code_files:
            print(
                "[ANALYSIS] No user Python files found in profiling data for pattern analysis"
            )
            return None

        print(f"🔍 Running pattern analysis on {len(code_files)} profiled files...")

        # Run analysis
        analysis_results = analysis_orchestrator.run_analysis(code_files)

        if analysis_results:
            # Generate analysis report
            analysis_report = analysis_orchestrator.generate_report(analysis_results)

            print(
                f"[ANALYSIS] Pattern analysis complete - results integrated into comprehensive report"
            )

            # Print summary
            summary = analysis_report.get("summary", {})
            total_patterns = summary.get("total_patterns_detected", 0)
            if total_patterns > 0:
                print(
                    f"   [WARNING] Found {total_patterns} patterns across {summary.get('total_files_analyzed', 0)} files"
                )

                # Show pattern distribution
                pattern_dist = summary.get("pattern_distribution", {})
                if pattern_dist:
                    top_patterns = sorted(
                        pattern_dist.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    print(
                        f"   🏷️  Top patterns: {', '.join(f'{k}({v})' for k, v in top_patterns)}"
                    )

                # Show top issues
                top_issues = analysis_report.get("top_issues", [])
                if top_issues:
                    print(f"   🔥 Priority issues:")
                    for i, issue in enumerate(top_issues[:3], 1):
                        severity_emoji = {
                            "low": "[LOW]",
                            "medium": "[WARNING]",
                            "high": "[HIGH]",
                            "critical": "[CRITICAL]",
                        }.get(issue["severity"], "[WARNING]")
                        correlated = (
                            " [PERF]" if issue.get("performance_correlated") else ""
                        )
                        print(
                            f"      {i}. {severity_emoji} {issue['pattern_type']}{correlated}"
                        )
            else:
                print("   [OK] No significant patterns detected")

            return analysis_report
        else:
            print("[ANALYSIS] Pattern analysis completed - no issues found")
            return None

    def _extract_profiled_files(self) -> List[Path]:
        """Extract Python files that were actually profiled from session data."""
        profiled_files = set()

        # Extract files from line profiler data (most comprehensive source)
        line_result = self.session.get_result("line")
        if line_result and line_result.data:
            line_data = line_result.data
            # Line profiler data is aggregated data, need to look in line_stats
            if isinstance(line_data, dict) and "line_stats" in line_data:
                line_stats = line_data["line_stats"]
                if isinstance(line_stats, dict):
                    for filename_with_line in line_stats.keys():
                        if isinstance(filename_with_line, str):
                            # Extract filename by removing line number (format: "filename:line")
                            filename = filename_with_line.split(":")[0]
                            file_path = Path(filename)
                            if self._is_user_file(file_path):
                                profiled_files.add(file_path)

        # Extract files from call profiler data as backup
        call_result = self.session.get_result("call")
        if call_result and call_result.data:
            call_data = call_result.data
            if isinstance(call_data, dict) and "calls" in call_data:
                for call_info in call_data["calls"]:
                    if "filename" in call_info:
                        file_path = Path(call_info["filename"])
                        if self._is_user_file(file_path):
                            profiled_files.add(file_path)

        # Filter to only existing files and convert to list
        existing_files = []
        for file_path in profiled_files:
            if file_path.exists() and file_path.is_file():
                existing_files.append(file_path)

        return existing_files

    def _is_user_file(self, file_path: Path) -> bool:
        """Determine if a file path represents user code (not Pycroscope internals)."""
        file_str = str(file_path)

        # Must be a .py file
        if file_path.suffix != ".py":
            return False

        # Exclude Pycroscope's own source code (specifically src/pycroscope directory)
        if "/src/pycroscope/" in file_str or file_str.endswith("/src/pycroscope"):
            return False

        # Exclude standard library files
        if "/lib/python" in file_str or "/site-packages/" in file_str:
            return False

        # Exclude virtual environment files
        if "/venv/" in file_str or "/.venv/" in file_str:
            return False

        # Exclude common non-user directories
        exclude_dirs = {"__pycache__", ".git", ".pytest_cache", "node_modules"}
        if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
            return False

        return True

    def _cleanup_profilers(self) -> None:
        """Clean up profilers in case of errors."""
        for profiler_type, profiler_instance in self._running_profilers.items():
            profiler_instance.stop()  # Fail fast - any cleanup errors should be visible
        self._running_profilers.clear()
