"""
Garbage collection profiler for Pycroscope.

Monitors garbage collection events to identify memory management
performance issues and optimization opportunities.
"""

import gc
import sys
import threading
import time
import weakref
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..core.models import EventType, ExecutionEvent, FrameInfo, SourceLocation
from .base import BaseCollector


class GCCollector(BaseCollector):
    """
    Collector for garbage collection profiling.

    Tracks garbage collection events, timing, and memory patterns
    to identify memory management performance issues.
    """

    def __init__(self, config=None):
        """
        Initialize the GC collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)

        # GC callbacks and statistics
        self._gc_callbacks = []
        self._gc_stats = {
            "total_collections": 0,
            "collections_by_generation": [0, 0, 0],
            "total_gc_time": 0,
            "objects_collected": 0,
            "collections_with_uncollectable": 0,
            "expensive_collections": [],
            "generation_stats": defaultdict(list),
        }

        # Object tracking
        self._tracked_objects = weakref.WeakSet()
        self._object_creation_times = weakref.WeakKeyDictionary()
        self._object_types_count = defaultdict(int)

        # GC thresholds and settings
        self._expensive_gc_threshold_ms = 10  # 10ms
        self._very_expensive_gc_threshold_ms = 100  # 100ms

        # Original GC settings (to restore later)
        self._original_gc_callbacks = None
        self._original_gc_thresholds = None
        self._original_gc_enabled = None

        # Thread safety
        self._gc_lock = threading.RLock()

        # Track GC state
        self._gc_in_progress = False
        self._current_collection_start = None
        self._current_generation = None

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "gc"

    def _install_hooks(self) -> None:
        """Install garbage collection monitoring hooks."""
        # Save original GC state
        self._original_gc_enabled = gc.isenabled()
        self._original_gc_thresholds = gc.get_threshold()
        self._original_gc_callbacks = gc.callbacks.copy()

        # Install our GC callback
        gc.callbacks.append(self._gc_callback)

        # Enable automatic garbage collection if disabled
        if not gc.isenabled():
            gc.enable()

        # Install object creation tracking
        self._install_object_tracking()

    def _uninstall_hooks(self) -> None:
        """Remove garbage collection monitoring hooks."""
        # Remove our callback
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)

        # Restore original GC state
        if self._original_gc_enabled is not None:
            if self._original_gc_enabled:
                gc.enable()
            else:
                gc.disable()

        if self._original_gc_thresholds is not None:
            gc.set_threshold(*self._original_gc_thresholds)

        # Remove object tracking
        self._uninstall_object_tracking()

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """Collect GC events from buffer."""
        # Return buffered events - BaseCollector handles the actual buffering
        yield from []

    def _install_object_tracking(self):
        """Install hooks for tracking object creation and lifetime."""
        # This is a simplified implementation
        # In a full implementation, we might hook into object allocation
        # For now, we'll track objects during GC cycles
        pass

    def _uninstall_object_tracking(self):
        """Remove object tracking hooks."""
        pass

    def _gc_callback(self, phase, info):
        """Callback function called during garbage collection."""
        current_time = time.perf_counter_ns()

        try:
            if phase == "start":
                self._handle_gc_start(info, current_time)
            elif phase == "stop":
                self._handle_gc_stop(info, current_time)
        except Exception:
            # Don't let our profiling interfere with GC
            pass

    def _handle_gc_start(self, info, timestamp):
        """Handle start of garbage collection."""
        with self._gc_lock:
            self._gc_in_progress = True
            self._current_collection_start = timestamp
            self._current_generation = info.get("generation", -1)

            # Get current frame info
            frame = sys._getframe(2)  # Skip our callback frames
            caller_location = SourceLocation(
                filename=frame.f_code.co_filename,
                line_number=frame.f_lineno,
                function_name=frame.f_code.co_name,
            )

            # Create GC start event
            self._record_gc_event(
                "gc_start",
                timestamp,
                caller_location,
                {
                    "generation": self._current_generation,
                    "total_objects_before": len(gc.get_objects()),
                    "gc_counts": gc.get_count(),
                    "gc_stats": gc.get_stats(),
                },
            )

    def _handle_gc_stop(self, info, timestamp):
        """Handle end of garbage collection."""
        with self._gc_lock:
            if not self._gc_in_progress or self._current_collection_start is None:
                return

            # Calculate collection duration
            collection_duration = timestamp - self._current_collection_start

            # Get collection results
            collected_objects = info.get("collected", 0)
            uncollectable_objects = info.get("uncollectable", 0)

            # Get current frame info
            frame = sys._getframe(2)  # Skip our callback frames
            caller_location = SourceLocation(
                filename=frame.f_code.co_filename,
                line_number=frame.f_lineno,
                function_name=frame.f_code.co_name,
            )

            # Create GC stop event
            self._record_gc_event(
                "gc_stop",
                timestamp,
                caller_location,
                {
                    "generation": self._current_generation,
                    "duration_ms": collection_duration / 1_000_000,
                    "collected_objects": collected_objects,
                    "uncollectable_objects": uncollectable_objects,
                    "total_objects_after": len(gc.get_objects()),
                    "gc_counts_after": gc.get_count(),
                    "is_expensive": collection_duration
                    > self._expensive_gc_threshold_ms * 1_000_000,
                    "is_very_expensive": collection_duration
                    > self._very_expensive_gc_threshold_ms * 1_000_000,
                },
            )

            # Update statistics
            self._update_gc_stats(
                collection_duration, collected_objects, uncollectable_objects
            )

            # Reset state
            self._gc_in_progress = False
            self._current_collection_start = None
            self._current_generation = None

    def _record_gc_event(self, event_type, timestamp, caller_location, event_data):
        """Record a garbage collection event."""
        # Create frame info
        frame_info = FrameInfo(source_location=caller_location, local_variables={})

        # Map event types to EventType enum
        event_type_mapping = {
            "gc_start": EventType.GC_START,
            "gc_stop": EventType.GC_END,
        }

        # Create execution event
        event = ExecutionEvent(
            timestamp=timestamp,
            event_type=event_type_mapping.get(event_type, EventType.GC_START),
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            execution_time=(
                event_data.get("duration_ms", 0) * 1_000_000
                if "duration_ms" in event_data
                else None
            ),
            event_data=event_data,
        )

        # Buffer the event
        self._add_to_buffer(event.__dict__)

    def _update_gc_stats(self, duration, collected_objects, uncollectable_objects):
        """Update garbage collection statistics."""
        with self._gc_lock:
            self._gc_stats["total_collections"] += 1
            self._gc_stats["total_gc_time"] += duration
            self._gc_stats["objects_collected"] += collected_objects

            if (
                self._current_generation is not None
                and 0 <= self._current_generation <= 2
            ):
                self._gc_stats["collections_by_generation"][
                    self._current_generation
                ] += 1
                self._gc_stats["generation_stats"][self._current_generation].append(
                    {
                        "duration_ms": duration / 1_000_000,
                        "collected_objects": collected_objects,
                        "uncollectable_objects": uncollectable_objects,
                        "timestamp": time.perf_counter_ns(),
                    }
                )

            if uncollectable_objects > 0:
                self._gc_stats["collections_with_uncollectable"] += 1

            # Track expensive collections
            duration_ms = duration / 1_000_000
            if duration_ms > self._expensive_gc_threshold_ms:
                expensive_info = {
                    "generation": self._current_generation,
                    "duration_ms": duration_ms,
                    "collected_objects": collected_objects,
                    "uncollectable_objects": uncollectable_objects,
                    "timestamp": time.perf_counter_ns(),
                    "severity": (
                        "very_expensive"
                        if duration_ms > self._very_expensive_gc_threshold_ms
                        else "expensive"
                    ),
                }
                self._gc_stats["expensive_collections"].append(expensive_info)

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics including GC analysis."""
        base_stats = super().get_stats()

        with self._gc_lock:
            # Calculate average GC time
            avg_gc_time = 0
            if self._gc_stats["total_collections"] > 0:
                avg_gc_time = (
                    self._gc_stats["total_gc_time"]
                    / self._gc_stats["total_collections"]
                )

            # Get generation analysis
            generation_analysis = self._analyze_generations()

            # Get current GC state
            current_gc_state = {
                "gc_enabled": gc.isenabled(),
                "gc_counts": gc.get_count(),
                "gc_thresholds": gc.get_threshold(),
                "total_objects": len(gc.get_objects()),
                "gc_stats": gc.get_stats(),
            }

            # Get object type distribution
            object_type_distribution = self._get_object_type_distribution()

            gc_analysis = {
                "total_collections": self._gc_stats["total_collections"],
                "collections_by_generation": self._gc_stats[
                    "collections_by_generation"
                ],
                "total_gc_time_ms": self._gc_stats["total_gc_time"] / 1_000_000,
                "average_gc_time_ms": avg_gc_time / 1_000_000,
                "objects_collected": self._gc_stats["objects_collected"],
                "collections_with_uncollectable": self._gc_stats[
                    "collections_with_uncollectable"
                ],
                "expensive_collections_count": len(
                    self._gc_stats["expensive_collections"]
                ),
                "generation_analysis": generation_analysis,
                "current_gc_state": current_gc_state,
                "object_type_distribution": object_type_distribution,
                "expensive_collections": self._gc_stats["expensive_collections"][
                    -10:
                ],  # Last 10
                "gc_efficiency": self._calculate_gc_efficiency(),
            }

            base_stats["gc_analysis"] = gc_analysis

        return base_stats

    def _analyze_generations(self) -> Dict[str, Any]:
        """Analyze garbage collection patterns by generation."""
        analysis = {}

        for generation in range(3):
            stats_list = self._gc_stats["generation_stats"][generation]
            if not stats_list:
                analysis[f"generation_{generation}"] = {
                    "collection_count": 0,
                    "average_duration_ms": 0,
                    "total_objects_collected": 0,
                    "average_objects_collected": 0,
                }
                continue

            durations = [stat["duration_ms"] for stat in stats_list]
            collected_counts = [stat["collected_objects"] for stat in stats_list]

            analysis[f"generation_{generation}"] = {
                "collection_count": len(stats_list),
                "average_duration_ms": sum(durations) / len(durations),
                "max_duration_ms": max(durations),
                "min_duration_ms": min(durations),
                "total_objects_collected": sum(collected_counts),
                "average_objects_collected": sum(collected_counts)
                / len(collected_counts),
                "max_objects_collected": max(collected_counts),
                "efficiency_score": self._calculate_generation_efficiency(
                    generation, stats_list
                ),
            }

        return analysis

    def _calculate_generation_efficiency(self, generation, stats_list) -> float:
        """Calculate efficiency score for a generation."""
        if not stats_list:
            return 0.0

        # Efficiency = objects collected per millisecond of GC time
        total_duration = sum(stat["duration_ms"] for stat in stats_list)
        total_collected = sum(stat["collected_objects"] for stat in stats_list)

        if total_duration == 0:
            return 0.0

        return total_collected / total_duration

    def _get_object_type_distribution(self) -> Dict[str, int]:
        """Get distribution of object types in memory."""
        try:
            objects = gc.get_objects()
            type_counts = defaultdict(int)

            for obj in objects:
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1

            # Return top 20 most common types
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_types[:20])

        except Exception:
            return {}

    def _calculate_gc_efficiency(self) -> float:
        """Calculate overall garbage collection efficiency."""
        if (
            self._gc_stats["total_collections"] == 0
            or self._gc_stats["total_gc_time"] == 0
        ):
            return 0.0

        # Efficiency = objects collected per millisecond of total GC time
        total_time_ms = self._gc_stats["total_gc_time"] / 1_000_000
        return self._gc_stats["objects_collected"] / total_time_ms

    def force_gc_analysis(self) -> Dict[str, Any]:
        """Force a garbage collection and analyze the results."""
        with self._gc_lock:
            # Get state before GC
            objects_before = len(gc.get_objects())
            counts_before = gc.get_count()

            # Force garbage collection
            start_time = time.perf_counter_ns()
            collected = gc.collect()
            end_time = time.perf_counter_ns()

            # Get state after GC
            objects_after = len(gc.get_objects())
            counts_after = gc.get_count()

            duration = end_time - start_time

            analysis = {
                "forced_collection": True,
                "duration_ms": duration / 1_000_000,
                "objects_before": objects_before,
                "objects_after": objects_after,
                "objects_collected": collected,
                "objects_freed": objects_before - objects_after,
                "counts_before": counts_before,
                "counts_after": counts_after,
                "memory_efficiency": (
                    (objects_before - objects_after) / objects_before
                    if objects_before > 0
                    else 0
                ),
            }

            return analysis
