"""
Component registry for dependency injection and component discovery.

Provides a centralized location for registering and retrieving component
implementations, enabling loose coupling and extensibility.
"""

from typing import Dict, List, Type, Any, Optional, TypeVar, cast
from .interfaces import Collector, Analyzer, DataStore, Visualizer, Registry
from .config import CollectorType, AnalysisType, StorageType

T = TypeVar("T")


class ComponentRegistry(Registry):
    """
    Central registry for component implementations.

    Enables dependency injection by providing a mapping from interface
    types to concrete implementations.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._collectors: Dict[CollectorType, Type[Collector]] = {}
        self._analyzers: Dict[AnalysisType, Type[Analyzer]] = {}
        self._data_stores: Dict[StorageType, Type[DataStore]] = {}
        self._visualizers: List[Type[Visualizer]] = []

        # Generic registry for custom components
        self._components: Dict[str, Dict[str, Type]] = {}

    def register(
        self, interface_type: type, implementation: type, name: Optional[str] = None
    ) -> None:
        """
        Register an implementation for an interface.

        Args:
            interface_type: The interface/abstract base class
            implementation: Concrete implementation class
            name: Optional name for the implementation
        """
        if not name:
            name = implementation.__name__

        if interface_type not in self._components:
            self._components[interface_type.__name__] = {}

        self._components[interface_type.__name__][name] = implementation

    def get(self, interface_type: type, name: Optional[str] = None) -> Any:
        """
        Retrieve implementation instance for interface.

        Args:
            interface_type: The interface type to get implementation for
            name: Optional specific implementation name

        Returns:
            Instance of the implementation
        """
        interface_name = interface_type.__name__

        if interface_name not in self._components:
            return None

        implementations = self._components[interface_name]

        if name:
            if name in implementations:
                return implementations[name]()
            return None

        # Return first available implementation
        if implementations:
            return list(implementations.values())[0]()

        return None

    def get_all(self, interface_type: type) -> List[Any]:
        """
        Retrieve all implementations for interface.

        Args:
            interface_type: The interface type

        Returns:
            List of implementation instances
        """
        interface_name = interface_type.__name__

        if interface_name not in self._components:
            return []

        implementations = self._components[interface_name]
        return [impl() for impl in implementations.values()]

    # Specialized methods for core component types

    def register_collector(
        self, collector_type: CollectorType, implementation: Type[Collector]
    ) -> None:
        """Register a collector implementation."""
        self._collectors[collector_type] = implementation

    def get_collector_class(
        self, collector_type: CollectorType
    ) -> Optional[Type[Collector]]:
        """Get collector class for type."""
        return self._collectors.get(collector_type)

    def register_analyzer(
        self, analyzer_type: AnalysisType, implementation: Type[Analyzer]
    ) -> None:
        """Register an analyzer implementation."""
        self._analyzers[analyzer_type] = implementation

    def get_analyzer_class(
        self, analyzer_type: AnalysisType
    ) -> Optional[Type[Analyzer]]:
        """Get analyzer class for type."""
        return self._analyzers.get(analyzer_type)

    def register_data_store(
        self, storage_type: StorageType, implementation: Type[DataStore]
    ) -> None:
        """Register a data store implementation."""
        self._data_stores[storage_type] = implementation

    def get_data_store_class(
        self, storage_type: StorageType
    ) -> Optional[Type[DataStore]]:
        """Get data store class for storage type."""
        return self._data_stores.get(storage_type)

    def register_visualizer(self, implementation: Type[Visualizer]) -> None:
        """Register a visualizer implementation."""
        self._visualizers.append(implementation)

    def get_visualizer_class(self) -> Optional[Type[Visualizer]]:
        """Get first available visualizer class."""
        return self._visualizers[0] if self._visualizers else None

    def list_collectors(self) -> List[CollectorType]:
        """List all registered collector types."""
        return list(self._collectors.keys())

    def list_analyzers(self) -> List[AnalysisType]:
        """List all registered analyzer types."""
        return list(self._analyzers.keys())

    def list_data_stores(self) -> List[StorageType]:
        """List all registered data store types."""
        return list(self._data_stores.keys())

    def clear(self) -> None:
        """Clear all registrations."""
        self._collectors.clear()
        self._analyzers.clear()
        self._data_stores.clear()
        self._visualizers.clear()
        self._components.clear()


# Global registry instance
_global_registry = ComponentRegistry()


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = ComponentRegistry()
