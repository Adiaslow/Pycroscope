"""
Call profiler using custom tracing to capture function call relationships.

This profiler tracks function calls, timing, and caller-callee relationships
without masking errors. Uses the trace multiplexer for cooperative profiling.
"""

from typing import Dict, Any
import time
from collections import defaultdict

from .base import BaseProfiler
from ...core.config import ProfileConfig


class CallProfiler(BaseProfiler):
    """
    Call profiler that tracks function calls and their relationships.

    Captures:
    - Function call counts and timing
    - Caller-callee relationships for call tree/flame graph generation
    - Compatible with other profilers via trace multiplexer
    """

    def __init__(self, config: ProfileConfig):
        """Initialize call profiler with configuration."""
        super().__init__(config)
        self.sort_key = "cumulative"
        self._call_stack = []
        self._call_stats = {}
        self._start_times = {}

    def _create_func_stats(self) -> Dict[str, Any]:
        """Create empty function statistics structure."""
        return {
            "count": 0,
            "total_time": 0.0,
            "self_time": 0.0,
            "callers": {},
            "callees": {},
        }

    @property
    def profiler_type(self) -> str:
        """Get the type of profiler."""
        return "call"

    def start(self) -> None:
        """Start call profiling with caller-callee relationship tracking."""
        from .trace_multiplexer import register_trace_function

        self._mark_start()

        def call_profiler_trace(frame, event, arg):
            """Custom call profiler that tracks function calls AND caller-callee relationships."""
            current_time = time.perf_counter()
            # Use filename:function without line number to ensure call/return events match
            func_name = f"{frame.f_code.co_filename}:{frame.f_code.co_name}"

            if event == "call":
                # Ensure function stats exist
                if func_name not in self._call_stats:
                    self._call_stats[func_name] = self._create_func_stats()

                # Track caller-callee relationship
                if self._call_stack:
                    caller = self._call_stack[-1]
                    # Ensure caller stats exist
                    if caller not in self._call_stats:
                        self._call_stats[caller] = self._create_func_stats()

                    # Record that caller calls this function
                    self._call_stats[caller]["callees"][func_name] = (
                        self._call_stats[caller]["callees"].get(func_name, 0) + 1
                    )
                    # Record that this function is called by caller
                    self._call_stats[func_name]["callers"][caller] = (
                        self._call_stats[func_name]["callers"].get(caller, 0) + 1
                    )

                # Function entry
                self._call_stack.append(func_name)
                self._start_times[func_name] = current_time
                self._call_stats[func_name]["count"] += 1

            elif event == "return":
                # Function exit
                if self._call_stack and self._call_stack[-1] == func_name:
                    self._call_stack.pop()
                    if func_name in self._start_times:
                        elapsed = current_time - self._start_times[func_name]
                        self._call_stats[func_name]["total_time"] += elapsed
                        self._call_stats[func_name]["self_time"] += elapsed
                        del self._start_times[func_name]

            return call_profiler_trace

        # Register our custom call profiler with the multiplexer
        register_trace_function("call_profiler", call_profiler_trace)
        print("[OK] Custom call profiler enabled and integrated with trace multiplexer")
        print("[COOP] Using cooperative tracing (compatible with other profilers)")

        # Store the profiler trace function for use in stop()
        self._trace_function = call_profiler_trace

    def stop(self) -> Dict[str, Any]:
        """Stop custom call profiling and return results with caller-callee data."""
        if not hasattr(self, "_call_stats"):
            raise RuntimeError("Call profiler was never started - no stats available")

        # Unregister from trace multiplexer
        from .trace_multiplexer import unregister_trace_function

        unregister_trace_function("call_profiler")
        self._mark_end()

        # Process collected call statistics
        total_calls = sum(stats["count"] for stats in self._call_stats.values())
        total_time = sum(stats["total_time"] for stats in self._call_stats.values())

        if total_calls == 0:
            raise RuntimeError(
                "Call profiler captured no function calls - profiling failed"
            )

        # Sort functions by total time (similar to cProfile's 'cumulative' sort)
        sorted_functions = sorted(
            self._call_stats.items(), key=lambda x: x[1]["total_time"], reverse=True
        )

        # Extract top functions for compatibility with visualization
        function_stats = {}
        for func_name, stats in sorted_functions[:50]:  # Top 50 functions
            # Parse function name to extract components
            parts = func_name.split(":")
            if len(parts) >= 2:
                filename = parts[0]
                func_name_only = parts[1]
                line_no = "0"  # No line number in our format
            else:
                filename = "unknown"
                func_name_only = func_name
                line_no = "0"

            func_id = f"{filename}:{line_no}({func_name_only})"

            # Convert caller/callee keys to match output format
            converted_callers = {}
            for caller_key, count in stats["callers"].items():
                caller_parts = caller_key.split(":")
                if len(caller_parts) >= 2:
                    caller_filename = caller_parts[0]
                    caller_func = caller_parts[1]
                    converted_caller_key = f"{caller_filename}:0({caller_func})"
                else:
                    converted_caller_key = caller_key
                converted_callers[converted_caller_key] = count

            converted_callees = {}
            for callee_key, count in stats["callees"].items():
                callee_parts = callee_key.split(":")
                if len(callee_parts) >= 2:
                    callee_filename = callee_parts[0]
                    callee_func = callee_parts[1]
                    converted_callee_key = f"{callee_filename}:0({callee_func})"
                else:
                    converted_callee_key = callee_key
                converted_callees[converted_callee_key] = count

            function_stats[func_id] = {
                "ncalls": stats["count"],
                "tottime": stats["self_time"],
                "cumtime": stats["total_time"],
                "percall": (
                    stats["total_time"] / stats["count"] if stats["count"] > 0 else 0.0
                ),
                "filename": filename,
                "function": func_name_only,
                "line": int(line_no),
                "callers": converted_callers,  # Convert caller keys to match output format
                "callees": converted_callees,  # Convert callee keys to match output format
            }

        # Create summary statistics in expected format
        results = {
            "total_calls": total_calls,
            "total_time": total_time,
            "stats": function_stats,  # Use "stats" not "function_stats" for consistency
            "stats_summary": f"Custom call profiler tracked {total_calls} function calls in {total_time:.6f} seconds",
            "sort_key": self.sort_key,
            "profiler_type": "custom_call_profiler",
        }

        # Clear tracking data
        self._call_stats.clear()
        self._call_stack.clear()
        self._start_times.clear()

        return results
