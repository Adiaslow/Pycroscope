"""
Call tree profiling collector.

Tracks function calls and returns to build hierarchical call trees,
providing insight into program flow and call patterns.
"""

import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Set

from ..core.config import CollectorConfig
from .base import BaseCollector


class CallCollector(BaseCollector):
    """
    Collector for call tree profiling.

    Uses Python's trace mechanism to monitor function calls and returns,
    building a hierarchical representation of program execution flow.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        """
        Initialize the call collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)
        self._original_trace_func = None
        self._call_stack: List[Dict[str, Any]] = []
        self._call_tree: Dict[str, Dict[str, Any]] = {}
        self._function_stats: Dict[str, Dict[str, Any]] = {}
        self._excluded_files: Set[str] = set()
        self._max_stack_depth = config.max_stack_depth if config else 100

        # Add default exclusions
        self._excluded_files.update(
            ["pycroscope", "site-packages", "<built-in>", "<frozen"]
        )

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "call"

    def _install_hooks(self) -> None:
        """Install call tracing hooks."""
        # Store original trace function
        self._original_trace_func = sys.gettrace()

        # Install our trace function
        sys.settrace(self._trace_calls)

    def _uninstall_hooks(self) -> None:
        """Remove call tracing hooks."""
        # Restore original trace function
        sys.settrace(self._original_trace_func)
        self._original_trace_func = None

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """
        Collect call tree events.

        This method is called periodically to retrieve buffered call events.
        """
        # Return any buffered call events
        # In a real implementation, this would process the call tree
        yield from []

    def _trace_calls(self, frame, event: str, arg):
        """
        Main trace function for call profiling.

        Args:
            frame: Current execution frame
            event: Trace event type ('call', 'return', 'exception')
            arg: Event-specific argument

        Returns:
            Trace function for continued tracing
        """
        if not self._is_running:
            return None

        # Skip if sampling says no
        if not self._should_sample():
            return self._trace_calls

        # Prevent stack overflow
        if len(self._call_stack) >= self._max_stack_depth:
            return self._trace_calls

        # Get frame information
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name

        # Skip excluded files
        if self._should_exclude_file(filename):
            return self._trace_calls

        # Handle different event types
        if event == "call":
            self._handle_call_event(frame, filename, line_number, function_name)
        elif event == "return":
            self._handle_return_event(frame, filename, line_number, function_name, arg)
        elif event == "exception":
            self._handle_exception_event(
                frame, filename, line_number, function_name, arg
            )

        return self._trace_calls

    def _handle_call_event(
        self, frame, filename: str, line_number: int, function_name: str
    ) -> None:
        """
        Handle function call events.

        Args:
            frame: Execution frame
            filename: Source filename
            line_number: Line number
            function_name: Function name
        """
        call_time = time.perf_counter_ns()
        thread_id = threading.get_ident()

        # Create function identifier
        func_id = f"{filename}:{function_name}:{line_number}"

        # Create call record
        call_record = {
            "func_id": func_id,
            "filename": filename,
            "function_name": function_name,
            "line_number": line_number,
            "call_time": call_time,
            "thread_id": thread_id,
            "depth": len(self._call_stack),
        }

        # Add to call stack
        self._call_stack.append(call_record)

        # Initialize function stats if needed
        if func_id not in self._function_stats:
            self._function_stats[func_id] = {
                "call_count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
                "callers": set(),
                "callees": set(),
            }

        # Update call count
        self._function_stats[func_id]["call_count"] += 1

        # Track caller-callee relationship
        if len(self._call_stack) > 1:
            caller_record = self._call_stack[-2]
            caller_id = caller_record["func_id"]

            # Add to caller's callees
            if caller_id in self._function_stats:
                self._function_stats[caller_id]["callees"].add(func_id)

            # Add to current function's callers
            self._function_stats[func_id]["callers"].add(caller_id)

        # Create call event
        event = self._create_base_event(
            "call",
            func_id=func_id,
            function_name=function_name,
            filename=filename,
            line_number=line_number,
            call_depth=len(self._call_stack),
            caller=(
                self._call_stack[-2]["func_id"] if len(self._call_stack) > 1 else None
            ),
        )

        # Add to buffer
        self._add_to_buffer(event)

    def _handle_return_event(
        self, frame, filename: str, line_number: int, function_name: str, return_value
    ) -> None:
        """
        Handle function return events.

        Args:
            frame: Execution frame
            filename: Source filename
            line_number: Line number
            function_name: Function name
            return_value: Function return value
        """
        return_time = time.perf_counter_ns()

        # Pop from call stack if it matches
        if self._call_stack:
            call_record = self._call_stack[-1]

            # Verify this is the right function (allow some flexibility for line numbers)
            if (
                call_record["function_name"] == function_name
                and call_record["filename"] == filename
            ):
                # Remove from stack
                self._call_stack.pop()

                # Calculate execution time
                execution_time = return_time - call_record["call_time"]
                func_id = call_record["func_id"]

                # Update function statistics
                if func_id in self._function_stats:
                    stats = self._function_stats[func_id]
                    stats["total_time"] += execution_time
                    stats["min_time"] = min(stats["min_time"], execution_time)
                    stats["max_time"] = max(stats["max_time"], execution_time)

                # Create return event
                event = self._create_base_event(
                    "return",
                    func_id=func_id,
                    function_name=function_name,
                    filename=filename,
                    line_number=line_number,
                    execution_time=execution_time,
                    call_depth=len(self._call_stack),
                    return_value_type=(
                        type(return_value).__name__
                        if return_value is not None
                        else "None"
                    ),
                )

                # Add to buffer
                self._add_to_buffer(event)

    def _handle_exception_event(
        self, frame, filename: str, line_number: int, function_name: str, exception_info
    ) -> None:
        """
        Handle exception events during function execution.

        Args:
            frame: Execution frame
            filename: Source filename
            line_number: Line number
            function_name: Function name
            exception_info: Exception information
        """
        exception_time = time.perf_counter_ns()

        # Extract exception details
        exception_type = type(exception_info).__name__ if exception_info else "Unknown"
        exception_message = str(exception_info) if exception_info else ""

        # Find matching call in stack
        func_id = None
        if self._call_stack:
            for call_record in reversed(self._call_stack):
                if (
                    call_record["function_name"] == function_name
                    and call_record["filename"] == filename
                ):
                    func_id = call_record["func_id"]
                    break

        # Create exception event
        event = self._create_base_event(
            "exception",
            func_id=func_id,
            function_name=function_name,
            filename=filename,
            line_number=line_number,
            call_depth=len(self._call_stack),
            exception_type=exception_type,
            exception_message=exception_message[:500],  # Truncate long messages
        )

        # Add to buffer
        self._add_to_buffer(event)

    def _should_exclude_file(self, filename: str) -> bool:
        """
        Check if a file should be excluded from profiling.

        Args:
            filename: File path to check

        Returns:
            True if file should be excluded
        """
        # Check against excluded patterns
        for excluded in self._excluded_files:
            if excluded in filename:
                return True

        # Check configuration patterns
        for pattern in self._config.exclude_patterns:
            if pattern in filename:
                return True

        return False

    def get_call_summary(self) -> Dict[str, Any]:
        """
        Get summary of call profiling data.

        Returns:
            Dictionary with call profiling statistics
        """
        if not self._function_stats:
            return {"error": "No call data available"}

        # Calculate overall statistics
        total_functions = len(self._function_stats)
        total_calls = sum(
            stats["call_count"] for stats in self._function_stats.values()
        )
        total_time = sum(stats["total_time"] for stats in self._function_stats.values())

        # Find hottest functions
        hottest_functions = []
        for func_id, stats in self._function_stats.items():
            if stats["call_count"] > 0:
                avg_time = stats["total_time"] / stats["call_count"]
                hottest_functions.append(
                    {
                        "func_id": func_id,
                        "call_count": stats["call_count"],
                        "total_time": stats["total_time"],
                        "average_time": avg_time,
                        "min_time": (
                            stats["min_time"]
                            if stats["min_time"] != float("inf")
                            else 0
                        ),
                        "max_time": stats["max_time"],
                        "caller_count": len(stats["callers"]),
                        "callee_count": len(stats["callees"]),
                    }
                )

        # Sort by total time descending
        hottest_functions.sort(key=lambda x: x["total_time"], reverse=True)

        return {
            "total_functions": total_functions,
            "total_calls": total_calls,
            "total_execution_time": total_time,
            "average_time_per_call": total_time / total_calls if total_calls > 0 else 0,
            "max_call_depth": max(
                len(self._call_stack),
                max((call["depth"] for call in self._call_stack), default=0),
            ),
            "hottest_functions": hottest_functions[:15],  # Top 15 functions
        }

    def get_call_graph(self) -> Dict[str, Any]:
        """
        Get the call graph structure.

        Returns:
            Dictionary representing the call graph
        """
        # Convert caller/callee relationships to a graph structure
        graph = {"nodes": [], "edges": []}

        # Add nodes (functions)
        for func_id, stats in self._function_stats.items():
            node = {
                "id": func_id,
                "call_count": stats["call_count"],
                "total_time": stats["total_time"],
                "average_time": (
                    stats["total_time"] / stats["call_count"]
                    if stats["call_count"] > 0
                    else 0
                ),
            }
            graph["nodes"].append(node)

        # Add edges (call relationships)
        edge_id = 0
        for func_id, stats in self._function_stats.items():
            for callee in stats["callees"]:
                edge = {
                    "id": edge_id,
                    "source": func_id,
                    "target": callee,
                    "relationship": "calls",
                }
                graph["edges"].append(edge)
                edge_id += 1

        return graph

    def clear_data(self) -> None:
        """Clear all collected call profiling data."""
        self._call_stack.clear()
        self._call_tree.clear()
        self._function_stats.clear()

    def export_call_data(self) -> Dict[str, Any]:
        """
        Export all call profiling data for analysis.

        Returns:
            Complete call profiling dataset
        """
        return {
            "function_stats": {
                func_id: {
                    **stats,
                    "callers": list(stats["callers"]),
                    "callees": list(stats["callees"]),
                }
                for func_id, stats in self._function_stats.items()
            },
            "call_graph": self.get_call_graph(),
            "collector_stats": self.get_stats(),
            "summary": self.get_call_summary(),
        }
