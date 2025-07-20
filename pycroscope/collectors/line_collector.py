"""
Line-level profiling collector.

Tracks execution time and call frequency for individual lines of code,
providing detailed insight into where time is spent in the application.
"""

import sys
import time
from typing import Any, Dict, Iterator, Optional, Set, List
from .base import BaseCollector
from ..core.config import CollectorConfig


class LineCollector(BaseCollector):
    """
    Collector for line-level execution profiling.

    Uses Python's trace mechanism to monitor every line execution,
    collecting timing and frequency data for detailed analysis.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        """
        Initialize the line collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)
        self._original_trace_func = None
        self._line_data: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._call_stack: List[tuple] = []
        self._excluded_files: Set[str] = set()

        # Add default exclusions for Pycroscope itself
        self._excluded_files.update(
            ["pycroscope", "site-packages", "<built-in>", "<frozen"]
        )

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "line"

    def _install_hooks(self) -> None:
        """Install line tracing hooks."""
        # Store original trace function
        self._original_trace_func = sys.gettrace()

        # Install our trace function
        sys.settrace(self._trace_calls)

    def _uninstall_hooks(self) -> None:
        """Remove line tracing hooks."""
        # Restore original trace function
        sys.settrace(self._original_trace_func)
        self._original_trace_func = None

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """
        Collect line execution events.

        This method is called periodically by the base collector
        to retrieve buffered events.
        """
        # Return any buffered line events
        # In a real implementation, this would be more sophisticated
        yield from []

    def _trace_calls(self, frame, event: str, arg):
        """
        Main trace function for line-level profiling.

        Args:
            frame: Current execution frame
            event: Trace event type ('call', 'line', 'return', 'exception')
            arg: Event-specific argument

        Returns:
            Trace function for continued tracing
        """
        if not self._is_running:
            return None

        # Skip if sampling says no
        if not self._should_sample():
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
        elif event == "line":
            self._handle_line_event(frame, filename, line_number, function_name)
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

        # Create call event
        event = self._create_base_event(
            "call",
            source_location=f"{filename}:{line_number}",
            function_name=function_name,
            filename=filename,
            line_number=line_number,
            call_time=call_time,
        )

        # Add to buffer
        self._add_to_buffer(event)

        # Track call stack for timing calculations
        self._call_stack.append((filename, line_number, function_name, call_time))

    def _handle_line_event(
        self, frame, filename: str, line_number: int, function_name: str
    ) -> None:
        """
        Handle line execution events.

        Args:
            frame: Execution frame
            filename: Source filename
            line_number: Line number
            function_name: Function name
        """
        execution_time = time.perf_counter_ns()

        # Initialize file data if needed
        if filename not in self._line_data:
            self._line_data[filename] = {}

        # Initialize line data if needed
        if line_number not in self._line_data[filename]:
            self._line_data[filename][line_number] = {
                "hit_count": 0,
                "total_time": 0,
                "function_name": function_name,
                "last_execution": 0,
            }

        # Calculate time since last line (approximate)
        line_info = self._line_data[filename][line_number]
        if line_info["last_execution"] > 0:
            line_time = execution_time - line_info["last_execution"]
        else:
            line_time = 0

        # Update statistics
        line_info["hit_count"] += 1
        line_info["total_time"] += line_time
        line_info["last_execution"] = execution_time

        # Create line execution event
        event = self._create_base_event(
            "line",
            source_location=f"{filename}:{line_number}",
            function_name=function_name,
            filename=filename,
            line_number=line_number,
            execution_time=line_time,
            hit_count=line_info["hit_count"],
            cumulative_time=line_info["total_time"],
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

        # Calculate function execution time
        function_time = 0
        if self._call_stack:
            call_filename, call_line, call_func, call_time = self._call_stack.pop()
            if call_func == function_name:
                function_time = return_time - call_time

        # Create return event
        event = self._create_base_event(
            "return",
            source_location=f"{filename}:{line_number}",
            function_name=function_name,
            filename=filename,
            line_number=line_number,
            return_time=return_time,
            function_duration=function_time,
            return_value_type=(
                type(return_value).__name__ if return_value is not None else "None"
            ),
        )

        # Add to buffer
        self._add_to_buffer(event)

    def _handle_exception_event(
        self, frame, filename: str, line_number: int, function_name: str, exception_info
    ) -> None:
        """
        Handle exception events.

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

        # Create exception event
        event = self._create_base_event(
            "exception",
            source_location=f"{filename}:{line_number}",
            function_name=function_name,
            filename=filename,
            line_number=line_number,
            exception_time=exception_time,
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

    def get_line_summary(self) -> Dict[str, Any]:
        """
        Get summary of line profiling data.

        Returns:
            Dictionary with line profiling statistics
        """
        total_lines = 0
        total_hits = 0
        total_time = 0
        hottest_lines = []

        for filename, lines in self._line_data.items():
            for line_number, line_info in lines.items():
                total_lines += 1
                total_hits += line_info["hit_count"]
                total_time += line_info["total_time"]

                hottest_lines.append(
                    {
                        "filename": filename,
                        "line_number": line_number,
                        "function_name": line_info["function_name"],
                        "hit_count": line_info["hit_count"],
                        "total_time": line_info["total_time"],
                        "avg_time": (
                            line_info["total_time"] / line_info["hit_count"]
                            if line_info["hit_count"] > 0
                            else 0
                        ),
                    }
                )

        # Sort by total time descending
        hottest_lines.sort(key=lambda x: x["total_time"], reverse=True)

        return {
            "total_lines_profiled": total_lines,
            "total_line_hits": total_hits,
            "total_execution_time": total_time,
            "average_time_per_hit": total_time / total_hits if total_hits > 0 else 0,
            "hottest_lines": hottest_lines[:10],  # Top 10 hottest lines
        }

    def clear_data(self) -> None:
        """Clear all collected line profiling data."""
        self._line_data.clear()
        self._call_stack.clear()

    def export_line_data(self) -> Dict[str, Any]:
        """
        Export all line profiling data for analysis.

        Returns:
            Complete line profiling dataset
        """
        return {
            "line_data": self._line_data.copy(),
            "collector_stats": self.get_stats(),
            "summary": self.get_line_summary(),
        }
