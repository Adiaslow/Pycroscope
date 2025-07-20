"""
Base collector implementation providing common functionality.

All concrete collectors inherit from BaseCollector to ensure consistent
behavior and reduce code duplication.
"""

import threading
import time
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, Iterator, List, Optional
from ..core.interfaces import Collector, Configurable, Lifecycle
from ..core.config import CollectorConfig


class BaseCollector(Collector, Configurable, Lifecycle):
    """
    Abstract base implementation for all profiling data collectors.

    Provides common functionality including event buffering, sampling,
    lifecycle management, and configuration handling.
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        """
        Initialize the base collector.

        Args:
            config: Optional collector configuration
        """
        self._config = config or CollectorConfig()
        self._buffer: deque = deque(maxlen=self._config.buffer_size)
        self._buffer_lock = threading.Lock()
        self._is_installed = False
        self._is_running = False
        self._sample_counter = 0
        self._start_time = time.perf_counter_ns()

        # Apply configuration
        self.configure(self._config.__dict__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this collector type."""
        pass

    @property
    def is_installed(self) -> bool:
        """Whether collection hooks are currently active."""
        return self._is_installed

    @property
    def is_running(self) -> bool:
        """Whether collector is actively running."""
        return self._is_running

    @property
    def configuration(self) -> CollectorConfig:
        """Current collector configuration."""
        return self._config

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration settings to this collector.

        Args:
            config: Configuration dictionary
        """
        # Update configuration from dictionary
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def install(self) -> None:
        """Install collection hooks into the Python runtime."""
        if self._is_installed:
            return

        try:
            self._install_hooks()
            self._is_installed = True
            self.start()
        except Exception as e:
            raise RuntimeError(f"Failed to install {self.name} collector: {e}")

    def uninstall(self) -> None:
        """Remove collection hooks and clean up resources."""
        if not self._is_installed:
            return

        try:
            self.stop()
            self._uninstall_hooks()
            self._is_installed = False
        except Exception as e:
            # Log error but don't raise to avoid cascading failures
            pass

    def start(self) -> None:
        """Initialize collector for active use."""
        if self._is_running:
            return

        self._is_running = True
        self._start_time = time.perf_counter_ns()
        self._sample_counter = 0

    def stop(self) -> None:
        """Shutdown collector and release resources."""
        if not self._is_running:
            return

        self._is_running = False

    def collect(self) -> Iterator[Dict[str, Any]]:
        """
        Yield collected profiling events as they occur.

        This base implementation handles sampling and buffering.
        Subclasses should override _collect_events for actual data collection.
        """
        if not self._is_running:
            return

        # Apply sampling
        if not self._should_sample():
            return

        # Collect events from subclass implementation
        for event in self._collect_events():
            if self._should_include_event(event):
                self._add_to_buffer(event)
                yield event

    def flush(self) -> List[Dict[str, Any]]:
        """Return all buffered events and clear internal state."""
        with self._buffer_lock:
            events = list(self._buffer)
            self._buffer.clear()
            return events

    @abstractmethod
    def _install_hooks(self) -> None:
        """Install collector-specific hooks. Implemented by subclasses."""
        pass

    @abstractmethod
    def _uninstall_hooks(self) -> None:
        """Remove collector-specific hooks. Implemented by subclasses."""
        pass

    @abstractmethod
    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """Collect events from instrumentation. Implemented by subclasses."""
        pass

    def _should_sample(self) -> bool:
        """Determine if this collection cycle should be sampled."""
        if self._config.sampling_rate >= 1.0:
            return True

        self._sample_counter += 1
        sample_threshold = 1.0 / self._config.sampling_rate

        if self._sample_counter >= sample_threshold:
            self._sample_counter = 0
            return True

        return False

    def _should_include_event(self, event: Dict[str, Any]) -> bool:
        """
        Determine if an event should be included based on filtering rules.

        Args:
            event: Event data to check

        Returns:
            True if event should be included
        """
        # Check exclude patterns
        source_location = event.get("source_location", "")

        for pattern in self._config.exclude_patterns:
            if pattern in source_location:
                return False

        # Check include patterns (if any specified)
        if self._config.include_patterns:
            for pattern in self._config.include_patterns:
                if pattern in source_location:
                    return True
            return False  # No include patterns matched

        return True  # No include patterns specified, include by default

    def _add_to_buffer(self, event: Dict[str, Any]) -> None:
        """
        Add event to internal buffer with thread safety.

        Args:
            event: Event data to buffer
        """
        with self._buffer_lock:
            # Add timestamp if not present
            if "timestamp" not in event:
                event["timestamp"] = time.perf_counter_ns()

            # Add collector identification
            event["collector"] = self.name

            # Add to buffer (automatically handles overflow due to maxlen)
            self._buffer.append(event)

    def _create_base_event(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """
        Create a base event structure with common fields.

        Args:
            event_type: Type of event being created
            **kwargs: Additional event-specific data

        Returns:
            Event dictionary with base structure
        """
        event = {
            "event_type": event_type,
            "timestamp": time.perf_counter_ns(),
            "collector": self.name,
            "thread_id": threading.get_ident(),
        }
        event.update(kwargs)
        return event

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collector statistics.

        Returns:
            Dictionary with collector runtime statistics
        """
        current_time = time.perf_counter_ns()
        runtime = (
            current_time - self._start_time
        ) / 1_000_000_000  # Convert to seconds

        with self._buffer_lock:
            buffer_size = len(self._buffer)

        return {
            "name": self.name,
            "is_installed": self._is_installed,
            "is_running": self._is_running,
            "runtime_seconds": runtime,
            "events_buffered": buffer_size,
            "buffer_capacity": self._config.buffer_size,
            "sampling_rate": self._config.sampling_rate,
            "sample_counter": self._sample_counter,
        }

    def __enter__(self):
        """Context manager entry."""
        self.install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.uninstall()

    def __repr__(self) -> str:
        """String representation."""
        status = "installed" if self._is_installed else "not installed"
        return f"{self.__class__.__name__}(name='{self.name}', status='{status}')"
