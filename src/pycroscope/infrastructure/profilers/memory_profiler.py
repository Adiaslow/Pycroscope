"""
Memory profiler implementation for Pycroscope.

Wraps psutil for comprehensive memory analysis.
All dependencies are required - no fallbacks.
"""

import time
from typing import Dict, Any, List, Optional
import psutil  # Required dependency
import threading

from .base import BaseProfiler
from ...core.config import ProfileConfig
from ...core.exceptions import ProfilerConflictError


class MemoryProfiler(BaseProfiler):
    """
    Wrapper around psutil for memory usage analysis.

    Provides memory profiling capabilities.
    psutil is a required dependency - no fallbacks.
    """

    def __init__(self, config):
        """Initialize MemoryProfiler with configuration."""
        from ...core.config import ProfileConfig

        # Handle both ProfileConfig objects and dicts
        if isinstance(config, ProfileConfig):
            super().__init__(config)
            self.precision = config.memory_precision
        elif isinstance(config, dict):
            # Create minimal ProfileConfig for base class
            from pathlib import Path
            import tempfile

            profile_config = ProfileConfig(
                output_dir=Path(tempfile.mkdtemp(prefix="pycroscope_")),
                profiler_prefix=config.get("prefix", "pycroscope"),
            )
            super().__init__(profile_config)
            self.precision = config.get("memory_precision", 1)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        self._process = psutil.Process()
        self._initial_memory = None
        self._peak_memory = None
        self._memory_samples = []
        self._monitoring = False
        self._monitor_thread = None
        self.sampling_interval = 0.01  # 10ms sampling

    @property
    def profiler_type(self) -> str:
        """Get the type of profiler."""
        return "memory"

    def start(self) -> None:
        """Start memory profiling."""
        self._check_conflicts()
        self._mark_start()

        # Get initial memory reading
        memory_info = self._process.memory_info()
        self._initial_memory = memory_info.rss
        self._peak_memory = self._initial_memory

        # Initialize samples with starting point
        start_time = time.time()
        self._memory_samples = [
            {
                "timestamp": start_time,
                "rss_mb": self._initial_memory / (1024 * 1024),  # Convert to MB
                "vms_mb": (
                    memory_info.vms / (1024 * 1024)
                    if hasattr(memory_info, "vms")
                    else 0
                ),
                "percent": self._process.memory_percent(),
            }
        ]

        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory, daemon=True
        )
        self._monitor_thread.start()

    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        start_time = self._memory_samples[0]["timestamp"]

        while self._monitoring:
            memory_info = self._process.memory_info()
            current_rss = memory_info.rss
            current_time = time.time()

            # Update peak memory
            if current_rss > self._peak_memory:
                self._peak_memory = current_rss

            # Add sample
            sample = {
                "timestamp": current_time,
                "rss_mb": current_rss / (1024 * 1024),  # Convert to MB
                "vms_mb": (
                    memory_info.vms / (1024 * 1024)
                    if hasattr(memory_info, "vms")
                    else 0
                ),
                "percent": self._process.memory_percent(),
            }
            self._memory_samples.append(sample)

            time.sleep(self.sampling_interval)

    def stop(self) -> Dict[str, Any]:
        """Stop memory profiling and return results."""
        if self._initial_memory is None:
            return {"samples": [], "error": "Memory profiling was not started"}

        # Stop monitoring
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

        self._mark_end()

        # Get final memory reading
        memory_info = self._process.memory_info()
        final_memory = memory_info.rss
        final_time = time.time()

        # Add final sample
        final_sample = {
            "timestamp": final_time,
            "rss_mb": final_memory / (1024 * 1024),
            "vms_mb": (
                memory_info.vms / (1024 * 1024) if hasattr(memory_info, "vms") else 0
            ),
            "percent": self._process.memory_percent(),
        }
        self._memory_samples.append(final_sample)

        # Calculate statistics in MB
        initial_mb = self._initial_memory / (1024 * 1024)
        final_mb = final_memory / (1024 * 1024)
        memory_delta_mb = final_mb - initial_mb

        # Calculate average and peak memory
        if self._memory_samples:
            memory_values = [sample["rss_mb"] for sample in self._memory_samples]
            peak_memory_mb = max(memory_values)
            avg_memory_mb = sum(memory_values) / len(memory_values)
        else:
            peak_memory_mb = final_mb
            avg_memory_mb = final_mb

        return {
            "samples": self._memory_samples,
            "peak_memory_mb": peak_memory_mb,
            "avg_memory_mb": avg_memory_mb,
            "memory_delta_mb": memory_delta_mb,
            "sample_count": len(self._memory_samples),
            "initial_memory_mb": initial_mb,
            "final_memory_mb": final_mb,
            "duration": self.duration if self.duration else 0,
            "metadata": {
                "profiler": "psutil",
                "precision": self.precision,
                "status": "completed",
                "sampling_rate": f"{1/self.sampling_interval:.0f} Hz",
            },
        }

    def _check_conflicts(self) -> None:
        """Check for memory profiling conflicts."""
        # Memory profiling using psutil typically doesn't conflict with other profilers
        pass
