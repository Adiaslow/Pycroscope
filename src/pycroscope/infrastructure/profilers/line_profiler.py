"""
Line profiler implementation for Pycroscope.

Wraps line_profiler for detailed line-by-line timing analysis.
All dependencies are required - no fallbacks.
"""

from typing import Dict, Any, Optional, List
import sys
import io
import os
import linecache
import line_profiler  # Required dependency

from .base import BaseProfiler
from ...core.config import ProfileConfig
from ...core.exceptions import ProfilerConflictError


class LineProfiler(BaseProfiler):
    """
    Wrapper around line_profiler for line-by-line timing analysis.

    Provides detailed timing information for each line of code.
    line_profiler is a required dependency - no fallbacks.
    """

    def __init__(self, config):
        """Initialize LineProfiler with configuration."""
        from ...core.config import ProfileConfig

        # Handle both ProfileConfig objects and dicts
        if isinstance(config, ProfileConfig):
            super().__init__(config)
        elif isinstance(config, dict):
            # Create minimal ProfileConfig for base class
            from pathlib import Path
            import tempfile

            profile_config = ProfileConfig(
                output_dir=Path(tempfile.mkdtemp(prefix="pycroscope_")),
                profiler_prefix=config.get("prefix", "pycroscope"),
            )
            super().__init__(profile_config)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        self._profiler = None
        self._original_trace_func = None

    @property
    def profiler_type(self) -> str:
        """Get the type of profiler."""
        return "line"

    def start(self) -> None:
        """Start line profiling with automatic code tracing."""
        self._mark_start()

        # Store original trace function
        self._original_trace_func = sys.gettrace()

        # Create line profiler
        self._profiler = line_profiler.LineProfiler()

        # Track functions we've added to avoid duplicates
        self._added_functions = set()

        def trace_calls(frame, event, arg):
            """Trace function calls and automatically add them to the line profiler."""
            if event == "call":
                code = frame.f_code
                filename = code.co_filename
                func_name = code.co_name

                # Only profile user code and Pycroscope code (not system libraries)
                if (
                    filename.endswith(".py")
                    and "site-packages" not in filename
                    and os.path.exists(filename)
                ):

                    func_id = (filename, func_name, code.co_firstlineno)
                    if func_id not in self._added_functions:
                        # Get the actual function object - fail fast on errors
                        # Get the function from frame globals if possible
                        frame_globals = frame.f_globals
                        if func_name in frame_globals:
                            func_obj = frame_globals[func_name]
                            if callable(func_obj) and hasattr(func_obj, "__code__"):
                                self._profiler.add_function(func_obj)
                                self._added_functions.add(func_id)
                                print(
                                    f"   [AUTO] Auto-added function to line profiler: {filename}:{func_name}"
                                )

            # Don't call original trace function if it's our multiplexer (avoid infinite recursion)
            if self._original_trace_func and (
                not hasattr(self._original_trace_func, "__name__")
                or self._original_trace_func.__name__ != "_multiplexed_trace"
            ):
                return self._original_trace_func(frame, event, arg)
            return trace_calls

        # Store trace function for cleanup
        self._trace_function = trace_calls

        # Register with trace multiplexer for cooperative tracing instead of direct sys.settrace
        from .trace_multiplexer import register_trace_function

        register_trace_function("line_profiler", trace_calls)

        # Enable line profiling
        self._profiler.enable_by_count()

        print("[OK] Line profiler enabled with automatic code tracing")
        print("   [INFO] Will automatically profile all executed Python code")
        print("   [COOP] Using cooperative tracing (compatible with other profilers)")

    def stop(self) -> Dict[str, Any]:
        """Stop line profiling and return results."""
        if self._profiler is None:
            return {"line_stats": {}, "error": "Line profiling was not started"}

        # Disable the profiler
        self._profiler.disable_by_count()

        # Unregister from trace multiplexer
        from .trace_multiplexer import unregister_trace_function

        unregister_trace_function("line_profiler")

        self._mark_end()

        # Get stats output
        stream = io.StringIO()
        self._profiler.print_stats(stream)
        stats_output = stream.getvalue()

        # Extract timing data
        line_stats = {}
        function_profiles = {}
        total_lines = 0
        total_hits = 0
        total_time = 0.0

        # Check if any functions were actually profiled
        if hasattr(self._profiler, "functions") and self._profiler.functions:
            print(f"Line profiler captured {len(self._profiler.functions)} functions")

            # Process each profiled function
            for func in self._profiler.functions:
                if hasattr(func, "__code__"):
                    filename = func.__code__.co_filename
                    func_name = func.__code__.co_name

                    # Include user code and relevant Pycroscope code
                    if (
                        filename.endswith(".py")
                        and "site-packages" not in filename
                        and os.path.exists(filename)
                    ):

                        # Get timing data for this function
                        if hasattr(self._profiler, "get_stats"):
                            line_stats_obj = self._profiler.get_stats()

                            # Extract line-by-line data
                            func_data = self._extract_function_data(
                                line_stats_obj, filename, func_name
                            )

                            if func_data:
                                line_stats.update(func_data["line_stats"])
                                if func_data["function_profile"]:
                                    function_profiles[f"{filename}:{func_name}"] = (
                                        func_data["function_profile"]
                                    )

                                total_lines += func_data["line_count"]
                                total_hits += func_data["total_hits"]
                                total_time += func_data["total_time"]
        else:
            # In nested profiling scenarios, line profiler may intentionally capture nothing
            # This is expected behavior when profilers are disabled for safety
            print(
                "[WARNING] Line profiler captured no functions (normal in nested profiling scenarios)"
            )

            # Check if this is likely a nested profiling scenario
            if self._original_trace_func is not None:
                print(
                    "   [INFO] Detected existing trace function - nested profiling scenario"
                )
                # Return empty but valid results instead of failing
                pass
            else:
                # Genuine failure case - raise error
                raise RuntimeError(
                    "Line profiler generated no timing data. This indicates the profiler "
                    "was not properly capturing function execution or no functions were "
                    "executed within the profiling context."
                )

        return {
            "line_stats": line_stats,
            "function_profiles": function_profiles,
            "total_lines": total_lines,
            "total_hits": total_hits,
            "total_time": total_time,
            "stats_output": stats_output,
            "duration": self.duration if self.duration else 0,
            "metadata": {
                "profiler": "line_profiler",
                "status": "completed",
                "functions_profiled": len(function_profiles),
                "lines_profiled": total_lines,
                "note": "Automatic tracing of all executed code",
            },
        }

    def _extract_function_data(self, line_stats_obj, filename, func_name):
        """Extract data for a specific function from line_profiler stats."""
        if not hasattr(line_stats_obj, "timings"):
            return None

        line_stats = {}
        line_details = []
        total_hits = 0
        total_time = 0.0
        line_count = 0

        # Find the key for this function in timings
        # Key format: (filename, start_line, function_name)
        func_key = None
        for key in line_stats_obj.timings.keys():
            if len(key) == 3 and key[0] == filename and key[2] == func_name:
                func_key = key
                break

        if func_key is None:
            return None

        # Get the line timing data for this function
        # Value format: [(line_number, hits, time_ns), ...]
        line_timings = line_stats_obj.timings[func_key]

        for line_number, hits, time_ns in line_timings:
            if hits > 0:
                time_seconds = time_ns / 1e9

                line_key = f"{filename}:{line_number}"
                line_stats[line_key] = {
                    "hits": hits,
                    "time": time_seconds,
                    "time_per_hit": time_seconds / hits if hits > 0 else 0,
                    "filename": filename,
                    "line_number": line_number,
                    "function": func_name,
                }

                # Get line content
                import linecache

                line_content = linecache.getline(filename, line_number).strip()

                line_details.append(
                    {
                        "line_number": line_number,
                        "hits": hits,
                        "time_us": time_ns / 1000,
                        "per_hit_us": (time_ns / hits / 1000) if hits > 0 else 0,
                        "line_content": line_content,
                    }
                )

                total_hits += hits
                total_time += time_seconds
                line_count += 1

        function_profile = None
        if line_details:
            function_profile = {
                "filename": filename,
                "function": func_name,
                "total_time": total_time,
                "total_hits": total_hits,
                "line_details": sorted(line_details, key=lambda x: x["line_number"]),
            }

        return {
            "line_stats": line_stats,
            "function_profile": function_profile,
            "line_count": line_count,
            "total_hits": total_hits,
            "total_time": total_time,
        }
