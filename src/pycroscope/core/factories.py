"""
Factory patterns for Pycroscope components.

Implements Factory Method and Abstract Factory patterns to create
profiler instances while maintaining loose coupling and extensibility.
"""

from typing import Dict, Any, Type, Optional, List
from abc import ABC, abstractmethod
import importlib
import platform

from .interfaces import Profiler, ProfilerFactory
from .exceptions import FactoryError, ProfilerNotAvailableError
from .constants import ProfilerType


class BaseProfilerFactory(ProfilerFactory):
    """
    Base implementation of profiler factory.

    Provides common functionality for all profiler factories
    following the Factory Method pattern.
    """

    def __init__(self, profiler_class: Type[Profiler]):
        self.profiler_class = profiler_class
        self._supported_types = self._get_supported_types()

    def create(self, config: Dict[str, Any]) -> Profiler:
        """Create profiler instance with given configuration."""
        try:
            # Extract relevant configuration for this profiler
            profiler_config = self._extract_config(config)

            # Validate configuration before creation
            self._validate_config(profiler_config)

            # Create instance with dependency injection
            return self._create_instance(profiler_config)

        except Exception as e:
            raise FactoryError(
                f"Failed to create {self.profiler_class.__name__}: {str(e)}",
                factory_type=self.profiler_class.__name__,
                cause=e,
            )

    def supports(self, profiler_type: str) -> bool:
        """Check if factory supports creating given profiler type."""
        return profiler_type in self._supported_types

    @abstractmethod
    def _get_supported_types(self) -> List[str]:
        """Get list of profiler types this factory supports."""
        pass

    def _extract_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant configuration for this profiler."""
        # Base implementation returns all config
        # Subclasses can override for specific extraction logic
        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration for this profiler."""
        # Base implementation does no validation
        # Subclasses can override for specific validation
        pass

    def _create_instance(self, config: Dict[str, Any]) -> Profiler:
        """Create the actual profiler instance."""
        return self.profiler_class(config)


class CallProfilerFactory(BaseProfilerFactory):
    """Factory for creating call profilers."""

    def _get_supported_types(self) -> List[str]:
        return [ProfilerType.CALL.value]

    def _extract_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract call profiler specific configuration."""
        return {
            "prefix": config.get("profiler_prefix", "pycroscope"),
            "use_thread_isolation": config.get("use_thread_isolation", True),
            "max_depth": config.get("max_call_depth", 50),
        }


class LineProfilerFactory(BaseProfilerFactory):
    """Factory for creating line profilers."""

    def _get_supported_types(self) -> List[str]:
        return [ProfilerType.LINE.value]

    def _extract_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract line profiler specific configuration."""
        return {
            "prefix": config.get("profiler_prefix", "pycroscope"),
            "use_thread_isolation": config.get("use_thread_isolation", True),
        }


class MemoryProfilerFactory(BaseProfilerFactory):
    """Factory for creating memory profilers."""

    def _get_supported_types(self) -> List[str]:
        return [ProfilerType.MEMORY.value]

    def _extract_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory profiler specific configuration."""
        return {
            "prefix": config.get("profiler_prefix", "pycroscope"),
            "use_thread_isolation": config.get("use_thread_isolation", True),
            "precision": config.get("memory_precision", 3),
        }


class ProfilerFactoryRegistry:
    """
    Registry of profiler factories implementing Abstract Factory pattern.

    Maintains a registry of factories for different profiler types
    and provides unified interface for profiler creation.
    """

    def __init__(self):
        self._factories: Dict[str, ProfilerFactory] = {}
        self._register_default_factories()

    def register_factory(self, profiler_type: str, factory: ProfilerFactory) -> None:
        """Register a factory for a profiler type."""
        self._factories[profiler_type] = factory

    def get_factory_for_type(self, profiler_type: str) -> ProfilerFactory:
        """Get the factory for a specific profiler type."""
        factory = self._factories.get(profiler_type)

        if factory is None:
            raise FactoryError(
                f"No factory registered for profiler type: {profiler_type}",
                factory_type=profiler_type,
            )

        return factory

    def create_profiler(self, profiler_type: str, config: Dict[str, Any]) -> Profiler:
        """Create a profiler instance of the specified type."""
        factory = self._factories.get(profiler_type)

        if factory is None:
            raise FactoryError(
                f"No factory registered for profiler type: {profiler_type}",
                factory_type=profiler_type,
            )

        if not factory.supports(profiler_type):
            raise FactoryError(
                f"Factory does not support profiler type: {profiler_type}",
                factory_type=profiler_type,
            )

        return factory.create(config)

    def get_supported_types(self) -> List[str]:
        """Get list of all supported profiler types."""
        supported = set()
        for profiler_type, factory in self._factories.items():
            if factory.supports(profiler_type):
                supported.add(profiler_type)
        return list(supported)

    def is_supported(self, profiler_type: str) -> bool:
        """Check if profiler type is supported."""
        return any(
            factory.supports(profiler_type) for factory in self._factories.values()
        )

    def _register_default_factories(self) -> None:
        """Register profiler factories."""
        # Factories will be registered by bootstrap module
        pass


# Global factory registry instance
_factory_registry: Optional[ProfilerFactoryRegistry] = None


def get_profiler_factory() -> ProfilerFactoryRegistry:
    """Get the global profiler factory registry."""
    global _factory_registry

    if _factory_registry is None:
        _factory_registry = ProfilerFactoryRegistry()

    return _factory_registry


def create_profiler(profiler_type: str, config: Dict[str, Any]) -> Profiler:
    """Convenience function to create profiler using global registry."""
    return get_profiler_factory().create_profiler(profiler_type, config)
