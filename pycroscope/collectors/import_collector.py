"""
Import timing profiler for Pycroscope.

Monitors module import operations to identify expensive imports
that could impact application startup time and overall performance.
"""

import sys
import time
import threading
import importlib.util
from typing import Any, Dict, Iterator, List, Optional, Set
from pathlib import Path

from .base import BaseCollector
from ..core.models import ExecutionEvent, FrameInfo, SourceLocation, EventType


class ImportCollector(BaseCollector):
    """
    Collector for import operation profiling.

    Tracks module import timing, dependency chains, and import costs
    to identify performance bottlenecks in module loading.
    """

    def __init__(self, config=None):
        """
        Initialize the import collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)
        self._original_import = None
        self._original_meta_path = None

        # Import tracking
        self._import_stack = []
        self._import_times = {}
        self._import_dependencies = {}
        self._import_stats = {
            "total_imports": 0,
            "total_import_time": 0,
            "expensive_imports": [],
            "circular_imports": [],
            "failed_imports": [],
        }

        # Thread safety
        self._import_lock = threading.RLock()

        # Thresholds for expensive imports (configurable)
        self._expensive_threshold_ms = 100  # 100ms
        self._very_expensive_threshold_ms = 1000  # 1s

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "import"

    def _install_hooks(self) -> None:
        """Install import monitoring hooks."""
        # Save original import function
        if isinstance(__builtins__, dict):
            self._original_import = __builtins__["__import__"]
            __builtins__["__import__"] = self._import_hook
        else:
            self._original_import = __builtins__.__import__
            __builtins__.__import__ = self._import_hook

        # Install meta path finder for additional tracking
        self._custom_finder = ImportMetaFinder(self)
        if self._custom_finder not in sys.meta_path:
            sys.meta_path.insert(0, self._custom_finder)

    def _uninstall_hooks(self) -> None:
        """Remove import monitoring hooks."""
        # Restore original import function
        if self._original_import:
            if isinstance(__builtins__, dict):
                __builtins__["__import__"] = self._original_import
            else:
                __builtins__.__import__ = self._original_import

        # Remove meta path finder
        if hasattr(self, "_custom_finder") and self._custom_finder in sys.meta_path:
            sys.meta_path.remove(self._custom_finder)

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """Collect import events from buffer."""
        # Return buffered events - BaseCollector handles the actual buffering
        yield from []

    def _import_hook(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Hook into import operations to track timing and dependencies."""
        import_start = time.perf_counter_ns()
        current_thread = threading.get_ident()

        # Get calling frame info
        frame = sys._getframe(1)
        caller_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        # Check for circular imports
        with self._import_lock:
            import_key = (name, level)
            if import_key in [item[0] for item in self._import_stack]:
                self._track_circular_import(name, level, caller_location)

            # Track import stack
            self._import_stack.append((import_key, import_start, caller_location))

        try:
            # Perform the actual import - check if original import is callable
            if callable(self._original_import):
                result = self._original_import(name, globals, locals, fromlist, level)
            else:
                # Fallback to importlib if original import is not available
                import importlib

                result = importlib.import_module(name)

            import_end = time.perf_counter_ns()
            import_duration = import_end - import_start

            # Record successful import
            self._record_import(
                name, level, fromlist, caller_location, import_duration, success=True
            )

            return result

        except Exception as e:
            import_end = time.perf_counter_ns()
            import_duration = import_end - import_start

            # Record failed import
            self._record_import(
                name,
                level,
                fromlist,
                caller_location,
                import_duration,
                success=False,
                error=str(e),
            )

            # Re-raise the exception
            raise

        finally:
            # Remove from import stack
            with self._import_lock:
                if self._import_stack:
                    self._import_stack.pop()

    def _record_import(
        self,
        module_name,
        level,
        fromlist,
        caller_location,
        duration,
        success=True,
        error=None,
    ):
        """Record import operation details."""
        current_time = time.perf_counter_ns()

        # Create frame info
        frame_info = FrameInfo(source_location=caller_location, local_variables={})

        # Determine import type
        if level > 0:
            import_type = "relative"
        elif fromlist:
            import_type = "from_import"
        else:
            import_type = "absolute"

        # Create execution event
        event = ExecutionEvent(
            timestamp=current_time,
            event_type=EventType.CALL,  # Using CALL for import operations
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            execution_time=duration,
            event_data={
                "operation": "import",
                "module_name": module_name,
                "import_type": import_type,
                "level": level,
                "fromlist": list(fromlist) if fromlist else [],
                "success": success,
                "error": error,
                "duration_ms": duration / 1_000_000,
                "is_expensive": duration > self._expensive_threshold_ms * 1_000_000,
                "is_very_expensive": duration
                > self._very_expensive_threshold_ms * 1_000_000,
                "import_depth": len(self._import_stack),
            },
        )

        # Buffer the event
        self._add_to_buffer(event.__dict__)

        # Update statistics
        self._update_import_stats(module_name, duration, success, error, import_type)

        # Track dependencies
        self._track_dependencies(module_name, caller_location)

    def _track_circular_import(self, name, level, caller_location):
        """Track circular import detection."""
        circular_info = {
            "module_name": name,
            "level": level,
            "caller_location": f"{caller_location.filename}:{caller_location.line_number}",
            "import_stack": [
                f"{item[2].filename}:{item[2].line_number}"
                for item in self._import_stack
            ],
            "timestamp": time.perf_counter_ns(),
        }

        with self._import_lock:
            self._import_stats["circular_imports"].append(circular_info)

    def _update_import_stats(self, module_name, duration, success, error, import_type):
        """Update import statistics."""
        with self._import_lock:
            self._import_stats["total_imports"] += 1

            if success:
                self._import_stats["total_import_time"] += duration

                # Track expensive imports
                duration_ms = duration / 1_000_000
                if duration_ms > self._expensive_threshold_ms:
                    expensive_info = {
                        "module_name": module_name,
                        "duration_ms": duration_ms,
                        "import_type": import_type,
                        "timestamp": time.perf_counter_ns(),
                        "severity": (
                            "very_expensive"
                            if duration_ms > self._very_expensive_threshold_ms
                            else "expensive"
                        ),
                    }
                    self._import_stats["expensive_imports"].append(expensive_info)

                # Store module timing
                if module_name not in self._import_times:
                    self._import_times[module_name] = []
                self._import_times[module_name].append(duration)
            else:
                # Track failed imports
                failed_info = {
                    "module_name": module_name,
                    "error": error,
                    "import_type": import_type,
                    "timestamp": time.perf_counter_ns(),
                }
                self._import_stats["failed_imports"].append(failed_info)

    def _track_dependencies(self, imported_module, caller_location):
        """Track import dependencies between modules."""
        caller_module = self._extract_module_name(caller_location.filename)

        with self._import_lock:
            if caller_module not in self._import_dependencies:
                self._import_dependencies[caller_module] = set()

            self._import_dependencies[caller_module].add(imported_module)

    def _extract_module_name(self, filename):
        """Extract module name from filename."""
        try:
            path = Path(filename)
            if path.name == "__init__.py":
                return path.parent.name
            else:
                return path.stem
        except Exception:
            return filename

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics including import analysis."""
        base_stats = super().get_stats()

        with self._import_lock:
            # Calculate average import time
            avg_import_time = 0
            if self._import_stats["total_imports"] > 0:
                avg_import_time = (
                    self._import_stats["total_import_time"]
                    / self._import_stats["total_imports"]
                )

            # Get slowest imports
            slowest_imports = self._get_slowest_imports()

            # Get most imported modules
            most_imported = self._get_most_imported_modules()

            # Get dependency analysis
            dependency_analysis = self._analyze_dependencies()

            import_analysis = {
                "total_imports": self._import_stats["total_imports"],
                "total_import_time_ms": self._import_stats["total_import_time"]
                / 1_000_000,
                "average_import_time_ms": avg_import_time / 1_000_000,
                "expensive_imports_count": len(self._import_stats["expensive_imports"]),
                "circular_imports_count": len(self._import_stats["circular_imports"]),
                "failed_imports_count": len(self._import_stats["failed_imports"]),
                "slowest_imports": slowest_imports,
                "most_imported_modules": most_imported,
                "dependency_analysis": dependency_analysis,
                "expensive_imports": self._import_stats["expensive_imports"][
                    -10:
                ],  # Last 10
                "circular_imports": self._import_stats["circular_imports"],
                "failed_imports": self._import_stats["failed_imports"][-10:],  # Last 10
            }

            base_stats["import_analysis"] = import_analysis

        return base_stats

    def _get_slowest_imports(self) -> List[Dict[str, Any]]:
        """Get the slowest module imports."""
        import_averages = []

        for module_name, times in self._import_times.items():
            avg_time = sum(times) / len(times)
            max_time = max(times)
            import_count = len(times)

            import_averages.append(
                {
                    "module_name": module_name,
                    "average_time_ms": avg_time / 1_000_000,
                    "max_time_ms": max_time / 1_000_000,
                    "import_count": import_count,
                    "total_time_ms": sum(times) / 1_000_000,
                }
            )

        return sorted(
            import_averages, key=lambda x: x["average_time_ms"], reverse=True
        )[:10]

    def _get_most_imported_modules(self) -> List[Dict[str, Any]]:
        """Get modules that are imported most frequently."""
        import_counts = [
            {
                "module_name": module_name,
                "import_count": len(times),
                "total_time_ms": sum(times) / 1_000_000,
                "average_time_ms": sum(times) / len(times) / 1_000_000,
            }
            for module_name, times in self._import_times.items()
        ]

        return sorted(import_counts, key=lambda x: x["import_count"], reverse=True)[:10]

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze import dependency patterns."""
        if not self._import_dependencies:
            return {}

        # Find modules with most dependencies
        modules_by_deps = sorted(
            self._import_dependencies.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Find most depended-upon modules
        all_dependencies = {}
        for module, deps in self._import_dependencies.items():
            for dep in deps:
                if dep not in all_dependencies:
                    all_dependencies[dep] = []
                all_dependencies[dep].append(module)

        most_depended = sorted(
            all_dependencies.items(), key=lambda x: len(x[1]), reverse=True
        )

        return {
            "total_modules_analyzed": len(self._import_dependencies),
            "modules_with_most_dependencies": [
                {"module": mod, "dependency_count": len(deps)}
                for mod, deps in modules_by_deps[:5]
            ],
            "most_depended_upon_modules": [
                {"module": mod, "dependent_count": len(dependents)}
                for mod, dependents in most_depended[:5]
            ],
        }


class ImportMetaFinder:
    """Meta path finder for additional import tracking."""

    def __init__(self, collector):
        self.collector = collector

    def find_spec(self, fullname, path, target=None):
        """Find module spec and track the operation."""
        # We don't actually handle the import, just track it
        # Return None to let other finders handle it
        return None

    def find_module(self, fullname, path=None):
        """Fallback for older Python versions."""
        return None
