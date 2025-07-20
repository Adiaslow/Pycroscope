"""
Memory profiling collector.

Tracks memory allocations, deallocations, and memory usage patterns
to help identify memory leaks and optimization opportunities.
"""

import gc
import sys
import threading
import time
import tracemalloc
from typing import Any, Dict, Iterator, Optional, List, Tuple
from .base import BaseCollector
from ..core.config import CollectorConfig


class MemoryCollector(BaseCollector):
    """
    Collector for memory allocation and usage profiling.

    Uses Python's tracemalloc and garbage collector to monitor
    memory allocations, deallocations, and usage patterns.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        """
        Initialize the memory collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)
        self._tracemalloc_started = False
        self._gc_callbacks_installed = False
        self._memory_snapshots: List[Dict[str, Any]] = []
        self._allocation_tracking: Dict[int, Dict[str, Any]] = {}
        self._peak_memory = 0
        self._last_snapshot_time = 0
        self._snapshot_interval = 1.0  # seconds

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "memory"

    def _install_hooks(self) -> None:
        """Install memory tracking hooks."""
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames in tracebacks
            self._tracemalloc_started = True

        # Install garbage collection callbacks
        if not self._gc_callbacks_installed:
            gc.callbacks.append(self._gc_callback)
            self._gc_callbacks_installed = True

    def _uninstall_hooks(self) -> None:
        """Remove memory tracking hooks."""
        # Stop tracemalloc if we started it
        if self._tracemalloc_started and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._tracemalloc_started = False

        # Remove garbage collection callbacks
        if self._gc_callbacks_installed:
            try:
                gc.callbacks.remove(self._gc_callback)
            except ValueError:
                pass  # Callback not in list
            self._gc_callbacks_installed = False

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """
        Collect memory-related events.

        This method is called periodically to collect memory snapshots
        and generate memory events.
        """
        current_time = time.time()

        # Take periodic memory snapshots
        if current_time - self._last_snapshot_time >= self._snapshot_interval:
            self._take_memory_snapshot()
            self._last_snapshot_time = current_time

        # Yield any buffered memory events
        yield from []

    def _take_memory_snapshot(self) -> None:
        """Take a snapshot of current memory usage."""
        if not tracemalloc.is_tracing():
            return

        current_time = time.perf_counter_ns()

        # Get current memory statistics
        current, peak = tracemalloc.get_traced_memory()

        # Update peak memory if necessary
        if peak > self._peak_memory:
            self._peak_memory = peak

        # Get top memory allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Create memory snapshot event
        snapshot_data = {
            "current_memory": current,
            "peak_memory": peak,
            "allocated_blocks": len(snapshot.traces),
            "top_allocations": [],
        }

        # Add top 10 memory allocations
        for index, stat in enumerate(top_stats[:10]):
            allocation_info = {
                "filename": (
                    stat.traceback.format()[0] if stat.traceback.format() else "unknown"
                ),
                "size_bytes": stat.size,
                "count": stat.count,
                "average_size": stat.size / stat.count if stat.count > 0 else 0,
            }
            snapshot_data["top_allocations"].append(allocation_info)

        # Create snapshot event
        event = self._create_base_event(
            "memory_snapshot", **snapshot_data, gc_collections=self._get_gc_stats()
        )

        # Add to buffer and internal tracking
        self._add_to_buffer(event)
        self._memory_snapshots.append(snapshot_data)

        # Limit number of stored snapshots
        if len(self._memory_snapshots) > 1000:
            self._memory_snapshots = self._memory_snapshots[-500:]

    def _gc_callback(self, phase: str, info: Dict[str, Any]) -> None:
        """
        Callback for garbage collection events.

        Args:
            phase: GC phase ('start' or 'stop')
            info: GC information dictionary
        """
        if not self._is_running:
            return

        # Create garbage collection event
        event = self._create_base_event(
            "gc_event",
            gc_phase=phase,
            generation=info.get("generation", -1),
            collected=info.get("collected", 0),
            connections=info.get("connections", 0),
            uncollectable=info.get("uncollectable", 0),
        )

        # Add to buffer
        self._add_to_buffer(event)

    def _get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collector statistics."""
        # Get GC stats for each generation
        gc_stats = gc.get_stats()

        return {
            "generation_0": gc_stats[0] if len(gc_stats) > 0 else {},
            "generation_1": gc_stats[1] if len(gc_stats) > 1 else {},
            "generation_2": gc_stats[2] if len(gc_stats) > 2 else {},
            "total_objects": len(gc.get_objects()),
            "garbage_objects": len(gc.garbage),
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory profiling data.

        Returns:
            Dictionary with memory profiling statistics
        """
        if not self._memory_snapshots:
            return {"error": "No memory snapshots available"}

        # Calculate memory trends
        snapshots = self._memory_snapshots
        current_memory = snapshots[-1]["current_memory"] if snapshots else 0

        # Memory growth calculation
        if len(snapshots) >= 2:
            initial_memory = snapshots[0]["current_memory"]
            memory_growth = current_memory - initial_memory
            growth_rate = memory_growth / len(snapshots) if len(snapshots) > 0 else 0
        else:
            memory_growth = 0
            growth_rate = 0

        # Find memory peaks
        peak_memory = max(snapshot["current_memory"] for snapshot in snapshots)

        # Allocation patterns
        total_allocations = sum(
            snapshot.get("allocated_blocks", 0) for snapshot in snapshots
        )
        avg_allocations = total_allocations / len(snapshots) if snapshots else 0

        return {
            "current_memory_bytes": current_memory,
            "peak_memory_bytes": peak_memory,
            "memory_growth_bytes": memory_growth,
            "growth_rate_per_snapshot": growth_rate,
            "total_snapshots": len(snapshots),
            "average_allocations_per_snapshot": avg_allocations,
            "gc_stats": self._get_gc_stats(),
            "memory_usage_mb": current_memory / (1024 * 1024),
            "peak_usage_mb": peak_memory / (1024 * 1024),
        }

    def get_allocation_hotspots(self) -> List[Dict[str, Any]]:
        """
        Get the top memory allocation hotspots.

        Returns:
            List of allocation hotspots with file and size information
        """
        if not self._memory_snapshots:
            return []

        # Aggregate allocations across all snapshots
        allocation_totals: Dict[str, Dict[str, Any]] = {}

        for snapshot in self._memory_snapshots:
            for allocation in snapshot.get("top_allocations", []):
                filename = allocation["filename"]
                if filename not in allocation_totals:
                    allocation_totals[filename] = {
                        "filename": filename,
                        "total_size": 0,
                        "total_count": 0,
                        "occurrences": 0,
                    }

                allocation_totals[filename]["total_size"] += allocation["size_bytes"]
                allocation_totals[filename]["total_count"] += allocation["count"]
                allocation_totals[filename]["occurrences"] += 1

        # Sort by total size and return top allocators
        hotspots = list(allocation_totals.values())
        hotspots.sort(key=lambda x: x["total_size"], reverse=True)

        # Add percentage of total memory
        total_memory = sum(spot["total_size"] for spot in hotspots)
        for spot in hotspots:
            spot["percentage"] = (
                (spot["total_size"] / total_memory * 100) if total_memory > 0 else 0
            )
            spot["average_size"] = (
                spot["total_size"] / spot["total_count"]
                if spot["total_count"] > 0
                else 0
            )

        return hotspots[:20]  # Top 20 hotspots

    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """
        Detect potential memory leaks based on growth patterns.

        Returns:
            List of potential memory leak indicators
        """
        if len(self._memory_snapshots) < 10:
            return [{"warning": "Not enough data to detect memory leaks"}]

        leaks = []
        snapshots = self._memory_snapshots

        # Check for consistent memory growth
        recent_snapshots = snapshots[-10:]
        growth_trend = []

        for i in range(1, len(recent_snapshots)):
            growth = (
                recent_snapshots[i]["current_memory"]
                - recent_snapshots[i - 1]["current_memory"]
            )
            growth_trend.append(growth)

        # If memory consistently grows, potential leak
        positive_growth = sum(1 for growth in growth_trend if growth > 0)
        if positive_growth >= len(growth_trend) * 0.8:  # 80% of snapshots show growth
            avg_growth = sum(growth_trend) / len(growth_trend)
            leaks.append(
                {
                    "type": "consistent_growth",
                    "severity": (
                        "high" if avg_growth > 1024 * 1024 else "medium"
                    ),  # 1MB threshold
                    "description": f"Memory consistently growing by avg {avg_growth/1024:.1f}KB per snapshot",
                    "average_growth_bytes": avg_growth,
                    "growth_snapshots": positive_growth,
                    "total_snapshots": len(growth_trend),
                }
            )

        # Check for objects that never get collected
        gc_stats = self._get_gc_stats()
        if gc_stats["garbage_objects"] > 0:
            leaks.append(
                {
                    "type": "uncollectable_objects",
                    "severity": "high",
                    "description": f'{gc_stats["garbage_objects"]} objects cannot be garbage collected',
                    "uncollectable_count": gc_stats["garbage_objects"],
                }
            )

        return leaks

    def clear_data(self) -> None:
        """Clear all collected memory profiling data."""
        self._memory_snapshots.clear()
        self._allocation_tracking.clear()
        self._peak_memory = 0

    def export_memory_data(self) -> Dict[str, Any]:
        """
        Export all memory profiling data for analysis.

        Returns:
            Complete memory profiling dataset
        """
        return {
            "memory_snapshots": self._memory_snapshots.copy(),
            "collector_stats": self.get_stats(),
            "summary": self.get_memory_summary(),
            "allocation_hotspots": self.get_allocation_hotspots(),
            "potential_leaks": self.detect_memory_leaks(),
        }
