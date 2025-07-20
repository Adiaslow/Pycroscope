"""
Exception handling profiler for Pycroscope.

Monitors exception creation, handling, and propagation to identify
performance costs associated with exception-based control flow.
"""

import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from ..core.models import ExecutionEvent, FrameInfo, SourceLocation
from .base import BaseCollector


class ExceptionCollector(BaseCollector):
    """
    Collector for exception handling profiling.

    Tracks exception creation, handling, and propagation to measure
    the performance impact of exception-based control flow and identify
    potential optimization opportunities.
    """

    def __init__(self, config=None):
        """
        Initialize the exception collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)
        self._original_excepthook = None
        self._original_unraisablehook = None
        self._exception_stats = {
            "total_exceptions": 0,
            "exceptions_by_type": {},
            "exceptions_by_location": {},
            "exception_chains": [],
        }
        self._current_exception_chain = []
        self._chain_lock = threading.Lock()

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "exception"

    def _install_hooks(self) -> None:
        """Install exception monitoring hooks."""
        # Save original hooks
        self._original_excepthook = sys.excepthook
        self._original_unraisablehook = getattr(sys, "unraisablehook", None)

        # Install our hooks
        sys.excepthook = self._exception_hook
        if hasattr(sys, "unraisablehook"):
            sys.unraisablehook = self._unraisable_hook

        # Install trace function for exception tracking
        self._original_trace = sys.gettrace()
        sys.settrace(self._trace_exceptions)

    def _uninstall_hooks(self) -> None:
        """Remove exception monitoring hooks."""
        # Restore original hooks
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook

        if self._original_unraisablehook and hasattr(sys, "unraisablehook"):
            sys.unraisablehook = self._original_unraisablehook

        # Restore original trace function
        if self._original_trace:
            sys.settrace(self._original_trace)
        else:
            sys.settrace(None)

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """Collect exception events from buffer."""
        # Return buffered events - BaseCollector handles the actual buffering
        yield from []

    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        try:
            self._record_exception(exc_type, exc_value, exc_traceback, is_uncaught=True)
        except Exception:
            # Don't let our profiling break the application
            pass
        finally:
            # Always call the original exception hook
            if self._original_excepthook:
                self._original_excepthook(exc_type, exc_value, exc_traceback)

    def _unraisable_hook(self, unraisable):
        """Handle unraisable exceptions."""
        try:
            exc_type = type(unraisable.exc_value) if unraisable.exc_value else Exception
            self._record_exception(
                exc_type,
                unraisable.exc_value,
                unraisable.exc_traceback,
                is_unraisable=True,
                context_object=unraisable.object,
            )
        except Exception:
            pass
        finally:
            if self._original_unraisablehook:
                self._original_unraisablehook(unraisable)

    def _trace_exceptions(self, frame, event, arg):
        """Trace function to catch exception events."""
        if event == "exception":
            try:
                exc_type, exc_value, exc_traceback = arg
                self._record_exception(exc_type, exc_value, exc_traceback, frame=frame)
            except Exception:
                pass

        # Continue with original trace function if it exists
        if self._original_trace:
            return self._original_trace(frame, event, arg)

        return self._trace_exceptions

    def _record_exception(
        self,
        exc_type,
        exc_value,
        exc_traceback,
        frame=None,
        is_uncaught=False,
        is_unraisable=False,
        context_object=None,
    ):
        """Record exception occurrence and details."""
        current_time = time.perf_counter_ns()

        # Extract source location
        if frame:
            source_location = SourceLocation(
                filename=frame.f_code.co_filename,
                line_number=frame.f_lineno,
                function_name=frame.f_code.co_name,
            )
        elif exc_traceback:
            tb_frame = exc_traceback.tb_frame
            source_location = SourceLocation(
                filename=tb_frame.f_code.co_filename,
                line_number=exc_traceback.tb_lineno,
                function_name=tb_frame.f_code.co_name,
            )
        else:
            source_location = SourceLocation("unknown", 0, "unknown")

        # Create frame info
        frame_info = FrameInfo(
            source_location=source_location,
            local_variables={} if not frame else self._extract_safe_locals(frame),
        )

        # Calculate exception processing time (rough estimate)
        exception_time = current_time - self._start_time

        # Create execution event
        from ..core.models import EventType

        event = ExecutionEvent(
            timestamp=current_time,
            event_type=EventType.EXCEPTION,
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            execution_time=exception_time,
            call_stack=self._extract_call_stack(exc_traceback),
            event_data={
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value) if exc_value else "",
                "is_uncaught": is_uncaught,
                "is_unraisable": is_unraisable,
                "exception_hierarchy": [cls.__name__ for cls in exc_type.__mro__],
                "has_cause": exc_value.__cause__ is not None if exc_value else False,
                "has_context": (
                    exc_value.__context__ is not None if exc_value else False
                ),
                "context_object": str(context_object) if context_object else None,
                "traceback_depth": self._get_traceback_depth(exc_traceback),
            },
        )

        # Buffer the event
        self._add_to_buffer(event.__dict__)

        # Update statistics
        self._update_exception_stats(exc_type, source_location, exc_value)

        # Track exception chains
        self._track_exception_chain(exc_type, exc_value, source_location)

    def _extract_call_stack(self, exc_traceback) -> List[str]:
        """Extract call stack from exception traceback."""
        if not exc_traceback:
            return []

        stack = []
        tb = exc_traceback

        while tb:
            frame = tb.tb_frame
            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = tb.tb_lineno

            # Create a readable stack frame representation
            stack.append(f"{filename}:{line_number} in {function_name}")
            tb = tb.tb_next

        return stack

    def _extract_safe_locals(self, frame) -> Dict[str, Any]:
        """Safely extract local variables from frame."""
        safe_locals = {}

        try:
            for name, value in frame.f_locals.items():
                try:
                    # Only include simple, serializable values
                    if isinstance(value, (str, int, float, bool, type(None))):
                        safe_locals[name] = value
                    elif (
                        isinstance(value, (list, tuple, dict)) and len(str(value)) < 200
                    ):
                        safe_locals[name] = str(value)[:200]
                    else:
                        safe_locals[name] = f"<{type(value).__name__}>"
                except Exception:
                    safe_locals[name] = "<unavailable>"
        except Exception:
            pass

        return safe_locals

    def _get_traceback_depth(self, exc_traceback) -> int:
        """Calculate the depth of the exception traceback."""
        if not exc_traceback:
            return 0

        depth = 0
        tb = exc_traceback
        while tb:
            depth += 1
            tb = tb.tb_next

        return depth

    def _update_exception_stats(self, exc_type, source_location, exc_value):
        """Update exception statistics."""
        with self._buffer_lock:
            self._exception_stats["total_exceptions"] += 1

            # Track by exception type
            exc_name = exc_type.__name__
            if exc_name not in self._exception_stats["exceptions_by_type"]:
                self._exception_stats["exceptions_by_type"][exc_name] = {
                    "count": 0,
                    "locations": set(),
                    "recent_messages": [],
                }

            self._exception_stats["exceptions_by_type"][exc_name]["count"] += 1
            self._exception_stats["exceptions_by_type"][exc_name]["locations"].add(
                f"{source_location.filename}:{source_location.line_number}"
            )

            # Keep recent messages (limit to avoid memory bloat)
            messages = self._exception_stats["exceptions_by_type"][exc_name][
                "recent_messages"
            ]
            if len(messages) < 10:
                messages.append(str(exc_value) if exc_value else "")

            # Track by location
            location_key = f"{source_location.filename}:{source_location.line_number}"
            if location_key not in self._exception_stats["exceptions_by_location"]:
                self._exception_stats["exceptions_by_location"][location_key] = {
                    "count": 0,
                    "exception_types": set(),
                }

            self._exception_stats["exceptions_by_location"][location_key]["count"] += 1
            self._exception_stats["exceptions_by_location"][location_key][
                "exception_types"
            ].add(exc_name)

    def _track_exception_chain(self, exc_type, exc_value, source_location):
        """Track exception chaining patterns."""
        with self._chain_lock:
            chain_entry = {
                "timestamp": time.perf_counter_ns(),
                "exception_type": exc_type.__name__,
                "location": f"{source_location.filename}:{source_location.line_number}",
                "has_cause": exc_value.__cause__ is not None if exc_value else False,
                "has_context": (
                    exc_value.__context__ is not None if exc_value else False
                ),
            }

            self._current_exception_chain.append(chain_entry)

            # If this completes a chain (no cause/context), store it
            if not (chain_entry["has_cause"] or chain_entry["has_context"]):
                if len(self._current_exception_chain) > 1:
                    self._exception_stats["exception_chains"].append(
                        self._current_exception_chain.copy()
                    )
                self._current_exception_chain.clear()

            # Prevent memory bloat
            if len(self._current_exception_chain) > 50:
                self._current_exception_chain = self._current_exception_chain[-25:]

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics including exception patterns."""
        base_stats = super().get_stats()

        with self._buffer_lock:
            # Convert sets to lists for JSON serialization
            serializable_stats = {
                "total_exceptions": self._exception_stats["total_exceptions"],
                "exceptions_by_type": {},
                "exceptions_by_location": {},
                "exception_chains": len(self._exception_stats["exception_chains"]),
                "most_common_exceptions": self._get_most_common_exceptions(),
                "hotspot_locations": self._get_exception_hotspots(),
            }

            # Convert exception type stats
            for exc_type, stats in self._exception_stats["exceptions_by_type"].items():
                serializable_stats["exceptions_by_type"][exc_type] = {
                    "count": stats["count"],
                    "locations": list(stats["locations"]),
                    "recent_messages": stats["recent_messages"],
                }

            # Convert location stats
            for location, stats in self._exception_stats[
                "exceptions_by_location"
            ].items():
                serializable_stats["exceptions_by_location"][location] = {
                    "count": stats["count"],
                    "exception_types": list(stats["exception_types"]),
                }

            base_stats["exception_statistics"] = serializable_stats

        return base_stats

    def _get_most_common_exceptions(self) -> List[Dict[str, Any]]:
        """Get the most frequently occurring exceptions."""
        exception_counts = [
            {"type": exc_type, "count": stats["count"]}
            for exc_type, stats in self._exception_stats["exceptions_by_type"].items()
        ]

        return sorted(exception_counts, key=lambda x: x["count"], reverse=True)[:10]

    def _get_exception_hotspots(self) -> List[Dict[str, Any]]:
        """Get locations with the highest exception frequency."""
        location_counts = [
            {
                "location": location,
                "count": stats["count"],
                "types": len(stats["exception_types"]),
            }
            for location, stats in self._exception_stats[
                "exceptions_by_location"
            ].items()
        ]

        return sorted(location_counts, key=lambda x: x["count"], reverse=True)[:10]
