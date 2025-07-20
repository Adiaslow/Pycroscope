"""
Core interfaces for the Pycroscope profiling system.

Defines abstract base classes that establish contracts for all major components,
ensuring clean separation of concerns and adherence to SOLID principles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


class Collector(ABC):
    """
    Abstract base class for all profiling data collectors.

    Each collector is responsible for gathering a specific type of profiling data
    (e.g., execution time, memory usage, call relationships) without knowing
    about other collectors or the overall system architecture.
    """

    @abstractmethod
    def install(self) -> None:
        """Install collection hooks into the Python runtime."""
        pass

    @abstractmethod
    def uninstall(self) -> None:
        """Remove collection hooks and clean up resources."""
        pass

    @abstractmethod
    def collect(self) -> Iterator[Dict[str, Any]]:
        """Yield collected profiling events as they occur."""
        pass

    @abstractmethod
    def flush(self) -> List[Dict[str, Any]]:
        """Return all buffered events and clear internal state."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this collector type."""
        pass

    @property
    @abstractmethod
    def is_installed(self) -> bool:
        """Whether collection hooks are currently active."""
        pass


class Analyzer(ABC):
    """
    Abstract base class for analysis engines.

    Analyzers process collected profiling data to extract insights,
    detect patterns, and generate optimization recommendations.
    """

    @abstractmethod
    def analyze(self, profile_data: "ProfileSession") -> "AnalysisResult":
        """
        Process profiling data and return analysis results.

        Args:
            profile_data: Complete profiling session data

        Returns:
            Structured analysis results with insights and recommendations
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this analyzer type."""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """List of collector names required by this analyzer."""
        pass


class DataStore(ABC):
    """
    Abstract base class for profiling data persistence.

    Handles storage and retrieval of profiling sessions, enabling
    historical analysis and comparison between optimization attempts.
    """

    @abstractmethod
    def store_session(self, session: "ProfileSession") -> str:
        """
        Persist a complete profiling session.

        Args:
            session: Complete profiling session data

        Returns:
            Unique session identifier for later retrieval
        """
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> Optional["ProfileSession"]:
        """
        Retrieve a stored profiling session.

        Args:
            session_id: Unique identifier from store_session

        Returns:
            Complete session data or None if not found
        """
        pass

    @abstractmethod
    def list_sessions(
        self, package_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """
        List available profiling sessions.

        Args:
            package_name: Filter by target package name
            limit: Maximum number of sessions to return

        Returns:
            List of session identifiers
        """
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """
        Remove a stored profiling session.

        Args:
            session_id: Unique identifier to delete

        Returns:
            True if deletion successful, False if session not found
        """
        pass


class Visualizer(ABC):
    """
    Abstract base class for profiling data visualization.

    Generates interactive dashboards and reports from analysis results,
    providing multiple perspectives on performance characteristics.
    """

    @abstractmethod
    def create_dashboard(self, analysis_result: "AnalysisResult") -> "Dashboard":
        """
        Generate interactive dashboard from analysis results.

        Args:
            analysis_result: Complete analysis data

        Returns:
            Interactive dashboard with linked visualizations
        """
        pass

    @abstractmethod
    def export_report(
        self, analysis_result: "AnalysisResult", format: str = "html"
    ) -> str:
        """
        Export static report in specified format.

        Args:
            analysis_result: Complete analysis data
            format: Output format ('html', 'pdf', 'json')

        Returns:
            Path to generated report file
        """
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported export formats."""
        pass


# Marker interfaces for type safety and extensibility


class Configurable(ABC):
    """Marker interface for components that accept configuration."""

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings to this component."""
        pass


class Lifecycle(ABC):
    """Marker interface for components with explicit lifecycle management."""

    @abstractmethod
    def start(self) -> None:
        """Initialize component for active use."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Shutdown component and release resources."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether component is currently active."""
        pass


# Component registry interface for dependency injection


class Registry(ABC):
    """
    Abstract component registry for dependency injection.

    Enables loose coupling between components by providing a centralized
    location for component discovery and instantiation.
    """

    @abstractmethod
    def register(
        self, interface_type: type, implementation: type, name: Optional[str] = None
    ) -> None:
        """Register an implementation for an interface."""
        pass

    @abstractmethod
    def get(self, interface_type: type, name: Optional[str] = None) -> Any:
        """Retrieve implementation instance for interface."""
        pass

    @abstractmethod
    def get_all(self, interface_type: type) -> List[Any]:
        """Retrieve all implementations for interface."""
        pass
