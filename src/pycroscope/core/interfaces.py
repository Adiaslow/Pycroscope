"""
Core interfaces and protocols for Pycroscope.

Defines contracts and abstractions to follow SOLID principles,
particularly the Dependency Inversion Principle (DIP).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Protocol, TypeVar, runtime_checkable
from contextlib import AbstractContextManager


T = TypeVar("T")


@runtime_checkable
class Profiler(Protocol):
    """Protocol defining the contract for all profilers."""

    @property
    def is_active(self) -> bool:
        """Check if profiler is currently active."""
        ...

    @property
    def profiler_type(self) -> str:
        """Get the type of profiler."""
        ...

    def start(self) -> None:
        """Start profiling."""
        ...

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        ...


@runtime_checkable
class ProfilerFactory(Protocol):
    """Protocol for profiler factories."""

    def create(self, config: Dict[str, Any]) -> Profiler:
        """Create a profiler instance with given configuration."""
        ...

    def supports(self, profiler_type: str) -> bool:
        """Check if factory supports creating given profiler type."""
        ...


@runtime_checkable
class ConfigurationProvider(Protocol):
    """Protocol for configuration providers."""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...

    def validate(self) -> bool:
        """Validate configuration."""
        ...


@runtime_checkable
class SessionManager(Protocol):
    """Protocol for session management."""

    def create_session(self, config: Any) -> Any:
        """Create a new profiling session."""
        ...

    def get_session(self, session_id: str) -> Optional[Any]:
        """Get existing session by ID."""
        ...

    def save_session(self, session: Any) -> str:
        """Save session and return path."""
        ...


@runtime_checkable
class ProfilerOrchestrator(Protocol):
    """Protocol for profiler orchestration."""

    def start_profiling(self) -> Any:
        """Start comprehensive profiling."""
        ...

    def stop_profiling(self) -> Any:
        """Stop profiling and return session."""
        ...

    def is_profiling_active(self) -> bool:
        """Check if any profiling is active."""
        ...


class ConflictDetector(ABC):
    """Abstract base for conflict detection strategies."""

    @abstractmethod
    def check_conflicts(self, profiler_type: str) -> bool:
        """Check for conflicts before starting profiler."""
        pass

    @abstractmethod
    def register_active_profiler(self, profiler_type: str, identifier: str) -> None:
        """Register an active profiler."""
        pass

    @abstractmethod
    def unregister_profiler(self, profiler_type: str, identifier: str) -> None:
        """Unregister a profiler."""
        pass


class ErrorHandler(ABC):
    """Abstract base for error handling strategies."""

    @abstractmethod
    def handle_profiler_error(self, profiler_type: str, error: Exception) -> bool:
        """Handle profiler error. Return True if error was handled."""
        pass

    @abstractmethod
    def handle_configuration_error(self, error: Exception) -> bool:
        """Handle configuration error. Return True if error was handled."""
        pass


# Type aliases for better readability
ProfilerConfig = Dict[str, Any]
ProfilerResults = Dict[str, Any]
SessionId = str
