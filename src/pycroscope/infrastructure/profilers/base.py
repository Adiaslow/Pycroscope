"""
Base profiler implementation for Pycroscope.

Provides common interface and utilities for all profiler wrappers,
following the Template Method and Strategy patterns.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

from ...core.config import ProfileConfig
from ...core.exceptions import ProfilerConflictError, ProfilerError


class BaseProfiler(ABC):
    """
    Abstract base class for all profiler implementations.

    Provides common interface and lifecycle management for profiler wrappers
    following the Template Method pattern.
    """

    def __init__(self, config: ProfileConfig):
        self.config = config
        self.is_active = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._profiler_instance = None

    @property
    def name(self) -> str:
        """Get profiler name."""
        return self.__class__.__name__.replace("Profiler", "").lower()

    @property
    @abstractmethod
    def profiler_type(self) -> str:
        """Get the type of profiler."""
        pass

    @property
    def duration(self) -> Optional[float]:
        """Get profiling duration."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @abstractmethod
    def start(self) -> None:
        """
        Start profiling.

        Raises:
            ProfilerError: If profiling cannot be started
        """
        pass

    @abstractmethod
    def stop(self) -> Dict[str, Any]:
        """
        Stop profiling and return collected data.

        Returns:
            Dictionary containing profiling results
        """
        pass

    def check_conflicts(self, other_profilers: List["BaseProfiler"]) -> None:
        """
        Check for conflicts with other profilers.

        Args:
            other_profilers: List of other profilers to check against

        Raises:
            ProfilerConflictError: If conflicts are detected
        """
        # Default implementation - override if profiler has specific conflicts
        pass

    @contextmanager
    def profile(self):
        """Context manager for profiling operations."""
        try:
            self.start()
            yield self
        finally:
            if self.is_active:
                self.stop()

    def _mark_start(self) -> None:
        """Mark profiling start time."""
        self.start_time = time.perf_counter()
        self.is_active = True

    def _mark_end(self) -> None:
        """Mark profiling end time."""
        self.end_time = time.perf_counter()
        self.is_active = False
