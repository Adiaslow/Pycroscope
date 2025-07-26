"""
Profiler orchestration and management.

Coordinates multiple profilers with conflict resolution and result aggregation
following clean architecture principles.
"""

from typing import Dict, Any, List, Set, Optional
from datetime import datetime
import threading
import time

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
                "🎯 Graceful degradation: Disabling trace-based profilers to avoid conflicts"
            )

            # Filter out conflicting profilers when we're being profiled
            trace_profilers = ["line", "call"]
            safe_profilers = [p for p in enabled_profilers if p not in trace_profilers]

            if not safe_profilers:
                print("   ⚠️  All requested profilers conflict with active tracer")
                print("   📊 Proceeding with non-trace profilers only")
                # At minimum, try memory profiler as it doesn't use sys.settrace()
                safe_profilers = [
                    p for p in ["memory", "sampling"] if p in enabled_profilers
                ]

            if safe_profilers:
                print(f"   ✅ Using safe profilers: {', '.join(safe_profilers)}")
                return self._simultaneous_profiling(safe_profilers)
            else:
                print(
                    "   ❌ No safe profilers available - profiling disabled in nested context"
                )
                return []

        # Use robust trace multiplexer to run all profilers simultaneously
        print("🎯 Using robust trace multiplexer - all profilers run simultaneously")
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

            print(f"✓ Started {profiler_type} profiler")

            # Start profiler - let any exceptions bubble up immediately
            profiler.start()

            self._running_profilers[profiler_type] = profiler
            started_profilers.append(profiler_type)

        print(
            f"🎯 Successfully started {len(started_profilers)} profilers: {', '.join(started_profilers)}"
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
            print(f"✓ Compatible profilers running: {', '.join(started_compatible)}")

        # Sequential profilers will be handled later in stop_profiling to avoid sys.settrace() conflicts
        # Store them for true sequential execution after compatible profilers finish
        self._sequential_profilers = sequential_profilers
        if sequential_profilers:
            print(
                f"🎯 Sequential profilers queued: {', '.join(sequential_profilers)} (will run after compatible profilers)"
            )

        all_profilers = (compatible if compatible else []) + sequential_profilers
        print(f"🎯 All profilers active: {', '.join(all_profilers)}")

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
        """Generate reports and visualizations if enabled."""
        if self.session.config.generate_reports:
            report_generator = ReportGenerator(self.session)
            report_path = report_generator.generate_comprehensive_report()
            print(f"📄 Generated profiling report: {report_path}")

        if self.session.config.create_visualizations:
            chart_generator = ChartGenerator(self.session)
            charts = chart_generator.generate_all_charts()
            if charts:
                print(f"📊 Generated {len(charts)} visualization charts:")
                for chart_name, chart_path in charts.items():
                    print(f"   • {chart_name}: {chart_path}")
            else:
                print("📊 No charts generated (no compatible profiling data)")

    def _cleanup_profilers(self) -> None:
        """Clean up profilers in case of errors."""
        for profiler_type, profiler_instance in self._running_profilers.items():
            profiler_instance.stop()  # Fail fast - any cleanup errors should be visible
        self._running_profilers.clear()
