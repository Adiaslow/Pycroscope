"""
Trace multiplexer for coordinating multiple profilers.

Allows cProfile and line_profiler to coexist by managing sys.settrace() properly.
This eliminates profiler conflicts and enables robust multi-profiler sessions.
"""

import sys
from typing import Dict, Callable, Any, Optional
from contextlib import contextmanager


class TraceMultiplexer:
    """
    Global coordinator for ALL profilers that need sys.settrace().

    Enables nested profiling scenarios (e.g., Pycroscope profiling Pycroscope)
    by being the single owner of sys.settrace() and distributing events to all
    registered profilers across all levels of nesting.
    """

    def __init__(self):
        self._profilers: Dict[str, Callable] = {}
        self._original_trace: Optional[Callable] = None
        self._is_multiplexer_active = False
        self._registration_count = 0

    def register_profiler(self, name: str, trace_func: Callable):
        """Register a profiler's trace function."""
        print(f"[TRACE] Trace multiplexer: Registering {name} profiler")

        # Handle duplicate registrations (e.g., nested Pycroscope)
        registration_key = f"{name}_{self._registration_count}"
        self._profilers[registration_key] = trace_func
        self._registration_count += 1

        # Ensure multiplexer is active
        self._ensure_active()

        print(f"   [INFO] Total registered profilers: {len(self._profilers)}")

    def unregister_profiler(self, name: str):
        """Unregister a profiler's trace function."""
        # Remove all profilers with this name prefix
        to_remove = [
            key for key in self._profilers.keys() if key.startswith(f"{name}_")
        ]
        for key in to_remove:
            self._profilers.pop(key, None)
            print(f"[TRACE] Trace multiplexer: Unregistered {key}")

        # Deactivate if no profilers remain
        if not self._profilers:
            self._deactivate()

    def _ensure_active(self):
        """Ensure the multiplexer owns sys.settrace()."""
        if not self._is_multiplexer_active:
            current_trace = sys.gettrace()

            # If there's already a trace function and it's not ours, we have a conflict
            if current_trace is not None and current_trace != self._multiplexed_trace:
                print(
                    f"[WARNING] Trace multiplexer: Detected existing trace function, taking control"
                )
                self._original_trace = current_trace

            sys.settrace(self._multiplexed_trace)
            self._is_multiplexer_active = True
            print(f"[OK] Trace multiplexer: Active and controlling sys.settrace()")

    def _deactivate(self):
        """Deactivate the multiplexer and restore original trace function."""
        if self._is_multiplexer_active:
            sys.settrace(self._original_trace)
            self._is_multiplexer_active = False
            print(f"[TRACE] Trace multiplexer: Deactivated, restored original trace")

    def _multiplexed_trace(self, frame, event, arg):
        """Multiplexed trace function that calls all registered profilers."""
        # Call all registered profiler trace functions - fail fast on any error
        for profiler_key, trace_func in self._profilers.items():
            trace_func(frame, event, arg)

        # Return ourselves to continue tracing
        return self._multiplexed_trace

    @contextmanager
    def activate(self):
        """Context manager to activate the trace multiplexer."""
        if self._is_multiplexer_active:
            yield self
            return

        # Activate the multiplexer
        self._ensure_active()

        try:
            yield self
        finally:
            # Deactivate and clear all profilers
            self._deactivate()
            self._profilers.clear()
            self._registration_count = 0


# Global instance for coordinating all profilers
_trace_multiplexer = TraceMultiplexer()


@contextmanager
def cooperative_tracing():
    """
    Context manager for cooperative tracing between profilers.

    Use this to enable multiple profilers that need sys.settrace() to work together.
    """
    with _trace_multiplexer.activate() as multiplexer:
        yield multiplexer


def register_trace_function(profiler_name: str, trace_func: Callable):
    """Register a trace function with the global multiplexer."""
    _trace_multiplexer.register_profiler(profiler_name, trace_func)


def unregister_trace_function(profiler_name: str):
    """Unregister a trace function from the global multiplexer."""
    _trace_multiplexer.unregister_profiler(profiler_name)
