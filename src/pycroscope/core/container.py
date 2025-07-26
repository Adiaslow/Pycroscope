"""
Dependency injection container for Pycroscope.

Implements the Dependency Inversion Principle (DIP) by providing
a centralized container for managing dependencies and their lifecycles.
"""

from typing import Dict, Any, TypeVar, Generic, Callable, Optional, Type, Union
from abc import ABC, abstractmethod
import threading
from functools import lru_cache

from .interfaces import Profiler, ProfilerFactory, ConfigurationProvider
from .exceptions import DependencyInjectionError, FactoryError


T = TypeVar("T")


class LifecycleScope:
    """Enumeration of dependency lifecycle scopes."""

    TRANSIENT = "transient"  # New instance every time
    SINGLETON = "singleton"  # One instance for lifetime of container
    SCOPED = "scoped"  # One instance per scope/session


class DependencyRegistration(Generic[T]):
    """Registration information for a dependency."""

    def __init__(
        self,
        factory: Callable[[], T],
        scope: str = LifecycleScope.TRANSIENT,
        interface: Optional[Type] = None,
    ):
        self.factory = factory
        self.scope = scope
        self.interface = interface
        self.instance: Optional[T] = None
        self._lock = threading.Lock()

    def get_instance(self) -> T:
        """Get instance according to lifecycle scope."""
        if self.scope == LifecycleScope.SINGLETON:
            if self.instance is None:
                with self._lock:
                    if self.instance is None:  # Double-check locking
                        self.instance = self.factory()
            return self.instance

        elif self.scope == LifecycleScope.TRANSIENT:
            return self.factory()

        else:
            # For scoped, we'd need a scope context - defaulting to transient
            return self.factory()


class DependencyContainer:
    """
    Dependency injection container implementing the Inversion of Control pattern.

    Provides centralized dependency management with support for:
    - Multiple lifecycle scopes (singleton, transient, scoped)
    - Factory-based registration
    - Type-safe resolution
    - Circular dependency detection
    """

    def __init__(self):
        self._registrations: Dict[str, DependencyRegistration] = {}
        self._resolution_stack: set = set()
        self._lock = threading.Lock()

    def register_singleton(
        self, interface: Type[T], implementation: Union[Type[T], Callable[[], T]]
    ) -> "DependencyContainer":
        """Register a singleton implementation for an interface."""
        key = self._get_key(interface)

        if callable(implementation) and not isinstance(implementation, type):
            factory = implementation
        else:
            factory = lambda: implementation()

        registration = DependencyRegistration(
            factory=factory, scope=LifecycleScope.SINGLETON, interface=interface
        )

        with self._lock:
            self._registrations[key] = registration

        return self

    def register_transient(
        self, interface: Type[T], implementation: Union[Type[T], Callable[[], T]]
    ) -> "DependencyContainer":
        """Register a transient implementation for an interface."""
        key = self._get_key(interface)

        if callable(implementation) and not isinstance(implementation, type):
            factory = implementation
        else:
            factory = lambda: implementation()

        registration = DependencyRegistration(
            factory=factory, scope=LifecycleScope.TRANSIENT, interface=interface
        )

        with self._lock:
            self._registrations[key] = registration

        return self

    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[[], T],
        scope: str = LifecycleScope.TRANSIENT,
    ) -> "DependencyContainer":
        """Register a factory function for an interface."""
        key = self._get_key(interface)

        registration = DependencyRegistration(
            factory=factory, scope=scope, interface=interface
        )

        with self._lock:
            self._registrations[key] = registration

        return self

    def resolve(self, interface: Type[T]) -> T:
        """Resolve an instance of the specified interface."""
        key = self._get_key(interface)

        # Check for circular dependencies
        if key in self._resolution_stack:
            raise DependencyInjectionError(
                f"Circular dependency detected for {interface.__name__}",
                interface_name=interface.__name__,
            )

        registration = self._registrations.get(key)
        if registration is None:
            raise DependencyInjectionError(
                f"No registration found for interface {interface.__name__}",
                interface_name=interface.__name__,
            )

        self._resolution_stack.add(key)
        try:
            return registration.get_instance()
        except Exception as e:
            raise DependencyInjectionError(
                f"Failed to resolve {interface.__name__}: {str(e)}",
                interface_name=interface.__name__,
                cause=e,
            )
        finally:
            self._resolution_stack.discard(key)

    def is_registered(self, interface: Type[T]) -> bool:
        """Check if an interface is registered."""
        key = self._get_key(interface)
        return key in self._registrations

    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._registrations.clear()

    def get_registrations(self) -> Dict[str, DependencyRegistration]:
        """Get all current registrations (for debugging)."""
        return dict(self._registrations)

    @staticmethod
    def _get_key(interface: Type) -> str:
        """Get string key for interface type."""
        return f"{interface.__module__}.{interface.__name__}"


# Global container instance following the Singleton pattern
_container: Optional[DependencyContainer] = None
_container_lock = threading.Lock()


def get_container() -> DependencyContainer:
    """Get the global dependency container instance."""
    global _container

    if _container is None:
        with _container_lock:
            if _container is None:  # Double-check locking
                _container = DependencyContainer()

    return _container


def reset_container() -> None:
    """Reset the global container (mainly for testing)."""
    global _container

    with _container_lock:
        if _container is not None:
            _container.clear()
        _container = None


# Convenience functions for common operations
def register_singleton(
    interface: Type[T], implementation: Union[Type[T], Callable[[], T]]
) -> None:
    """Register a singleton in the global container."""
    get_container().register_singleton(interface, implementation)


def register_transient(
    interface: Type[T], implementation: Union[Type[T], Callable[[], T]]
) -> None:
    """Register a transient in the global container."""
    get_container().register_transient(interface, implementation)


def resolve(interface: Type[T]) -> T:
    """Resolve from the global container."""
    return get_container().resolve(interface)
