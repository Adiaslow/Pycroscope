"""
Tests for dependency injection container.

Tests the core DI container functionality including lifecycle management,
registration patterns, and thread safety.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pycroscope.core.container import (
    DependencyContainer,
    DependencyRegistration,
    LifecycleScope,
    get_container,
    reset_container,
)
from pycroscope.core.exceptions import DependencyInjectionError, FactoryError
from pycroscope.core.interfaces import Profiler, ProfilerFactory


class MockService:
    """Mock service for testing DI container."""

    def __init__(self, value: str = "default"):
        self.value = value
        self.creation_time = time.time()

    def get_value(self) -> str:
        return self.value


class MockProfiler:
    """Mock profiler for testing."""

    def __init__(self):
        self.is_active = False
        self.profiler_type = "mock"

    def start(self) -> None:
        self.is_active = True

    def stop(self) -> dict:
        self.is_active = False
        return {"mock": "data"}


class MockProfilerFactory:
    """Mock profiler factory for testing."""

    def create(self, config: dict) -> MockProfiler:
        return MockProfiler()

    def supports(self, profiler_type: str) -> bool:
        return profiler_type == "mock"


class TestDependencyRegistration:
    """Test DependencyRegistration lifecycle management."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_transient_registration_creates_new_instances(self):
        """Test transient scope creates new instances each time."""
        # Arrange
        factory = lambda: MockService("transient")
        registration = DependencyRegistration(factory, LifecycleScope.TRANSIENT)

        # Act
        instance1 = registration.get_instance()
        instance2 = registration.get_instance()

        # Assert
        assert instance1 is not instance2
        assert instance1.value == "transient"
        assert instance2.value == "transient"

    @pytest.mark.unit
    @pytest.mark.core
    def test_singleton_registration_returns_same_instance(self):
        """Test singleton scope returns same instance."""
        # Arrange
        factory = lambda: MockService("singleton")
        registration = DependencyRegistration(factory, LifecycleScope.SINGLETON)

        # Act
        instance1 = registration.get_instance()
        instance2 = registration.get_instance()

        # Assert
        assert instance1 is instance2
        assert instance1.value == "singleton"

    @pytest.mark.unit
    @pytest.mark.core
    def test_singleton_thread_safety(self):
        """Test singleton creation is thread-safe."""
        # Arrange
        creation_count = 0

        def factory():
            nonlocal creation_count
            creation_count += 1
            time.sleep(0.01)  # Simulate work
            return MockService(f"instance_{creation_count}")

        registration = DependencyRegistration(factory, LifecycleScope.SINGLETON)
        instances = []

        def get_instance():
            instances.append(registration.get_instance())

        # Act - Create instances from multiple threads
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Assert
        assert creation_count == 1  # Only one instance created
        assert all(instance is instances[0] for instance in instances)

    @pytest.mark.unit
    @pytest.mark.core
    def test_scoped_registration_defaults_to_transient(self):
        """Test scoped registration defaults to transient behavior."""
        # Arrange
        factory = lambda: MockService("scoped")
        registration = DependencyRegistration(factory, LifecycleScope.SCOPED)

        # Act
        instance1 = registration.get_instance()
        instance2 = registration.get_instance()

        # Assert
        assert instance1 is not instance2
        assert instance1.value == "scoped"


class TestDependencyContainer:
    """Test DependencyContainer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.container = DependencyContainer()

    @pytest.mark.unit
    @pytest.mark.core
    def test_register_singleton_with_class(self):
        """Test registering singleton with class."""
        # Act
        result = self.container.register_singleton(MockService, MockService)

        # Assert
        assert result is self.container  # Fluent interface
        key = f"{MockService.__module__}.{MockService.__name__}"
        assert key in self.container._registrations

    @pytest.mark.unit
    @pytest.mark.core
    def test_register_singleton_with_factory(self):
        """Test registering singleton with factory function."""
        # Arrange
        factory = lambda: MockService("factory_created")

        # Act
        self.container.register_singleton(MockService, factory)

        # Assert
        key = f"{MockService.__module__}.{MockService.__name__}"
        assert key in self.container._registrations
        registration = self.container._registrations[key]
        assert registration.scope == LifecycleScope.SINGLETON

    @pytest.mark.unit
    @pytest.mark.core
    def test_register_transient_with_class(self):
        """Test registering transient with class."""
        # Act
        result = self.container.register_transient(MockService, MockService)

        # Assert
        assert result is self.container  # Fluent interface
        key = f"{MockService.__module__}.{MockService.__name__}"
        assert key in self.container._registrations

    @pytest.mark.unit
    @pytest.mark.core
    def test_register_transient_with_factory(self):
        """Test registering transient with factory function."""
        # Arrange
        factory = lambda: MockService("transient_factory")

        # Act
        self.container.register_transient(MockService, factory)

        # Assert
        key = f"{MockService.__module__}.{MockService.__name__}"
        registration = self.container._registrations[key]
        assert registration.scope == LifecycleScope.TRANSIENT

    @pytest.mark.unit
    @pytest.mark.core
    def test_resolve_singleton_service(self):
        """Test resolving singleton service."""
        # Arrange
        self.container.register_singleton(MockService, MockService)

        # Act
        instance1 = self.container.resolve(MockService)
        instance2 = self.container.resolve(MockService)

        # Assert
        assert instance1 is instance2
        assert isinstance(instance1, MockService)

    @pytest.mark.unit
    @pytest.mark.core
    def test_resolve_transient_service(self):
        """Test resolving transient service."""
        # Arrange
        self.container.register_transient(MockService, MockService)

        # Act
        instance1 = self.container.resolve(MockService)
        instance2 = self.container.resolve(MockService)

        # Assert
        assert instance1 is not instance2
        assert isinstance(instance1, MockService)
        assert isinstance(instance2, MockService)

    @pytest.mark.unit
    @pytest.mark.core
    def test_resolve_unregistered_service_raises_error(self):
        """Test resolving unregistered service raises error."""
        # Act & Assert
        with pytest.raises(
            DependencyInjectionError, match="No registration found for interface"
        ):
            self.container.resolve(MockService)

    @pytest.mark.unit
    @pytest.mark.core
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        # This would require more complex setup, but we can test the stack mechanism
        # Arrange
        key = f"{MockService.__module__}.{MockService.__name__}"
        self.container._resolution_stack.add(key)

        # Act & Assert
        with pytest.raises(
            DependencyInjectionError, match="Circular dependency detected"
        ):
            self.container.resolve(MockService)

    @pytest.mark.unit
    @pytest.mark.core
    def test_clear_registrations(self):
        """Test clearing all registrations."""
        # Arrange
        self.container.register_singleton(MockService, MockService)
        assert len(self.container._registrations) > 0

        # Act
        self.container.clear()

        # Assert
        assert len(self.container._registrations) == 0

    @pytest.mark.unit
    @pytest.mark.core
    def test_is_registered(self):
        """Test checking if service is registered."""
        # Arrange
        assert not self.container.is_registered(MockService)

        # Act
        self.container.register_singleton(MockService, MockService)

        # Assert
        assert self.container.is_registered(MockService)

    @pytest.mark.unit
    @pytest.mark.core
    def test_thread_safety_of_registration(self):
        """Test thread safety of container registration."""

        # Arrange
        def register_service(suffix):
            factory = lambda: MockService(f"service_{suffix}")
            self.container.register_singleton(MockService, factory)

        # Act - Register from multiple threads
        threads = [
            threading.Thread(target=register_service, args=(i,)) for i in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Assert - Should not crash and should have one registration
        assert self.container.is_registered(MockService)


class TestContainerGlobalState:
    """Test global container management."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_container_returns_singleton(self):
        """Test get_container returns singleton instance."""
        # Act
        container1 = get_container()
        container2 = get_container()

        # Assert
        assert container1 is container2
        assert isinstance(container1, DependencyContainer)

    @pytest.mark.unit
    @pytest.mark.core
    def test_reset_container_creates_new_instance(self):
        """Test reset_container creates new instance."""
        # Arrange
        container1 = get_container()
        container1.register_singleton(MockService, MockService)

        # Act
        reset_container()
        container2 = get_container()

        # Assert
        assert container1 is not container2
        assert not container2.is_registered(MockService)

    @pytest.mark.unit
    @pytest.mark.core
    def test_container_state_isolation_after_reset(self):
        """Test container state is isolated after reset."""
        # Arrange
        container1 = get_container()
        container1.register_singleton(MockService, MockService)
        instance1 = container1.resolve(MockService)

        # Act
        reset_container()
        container2 = get_container()

        # Assert
        assert not container2.is_registered(MockService)
        with pytest.raises(DependencyInjectionError):
            container2.resolve(MockService)


class TestContainerErrorHandling:
    """Test container error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.container = DependencyContainer()

    @pytest.mark.unit
    @pytest.mark.core
    def test_factory_error_is_wrapped(self):
        """Test factory errors are wrapped in DependencyInjectionError."""

        # Arrange
        def failing_factory():
            raise ValueError("Factory failed")

        self.container.register_singleton(MockService, failing_factory)

        # Act & Assert
        with pytest.raises(DependencyInjectionError) as exc_info:
            self.container.resolve(MockService)

        assert "Factory failed" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, ValueError)

    @pytest.mark.unit
    @pytest.mark.core
    def test_invalid_key_handling(self):
        """Test handling of invalid registration keys."""

        # This tests the _get_key method's robustness for unregistered types
        # Arrange & Act & Assert
        class UnregisteredService:
            pass

        with pytest.raises(
            DependencyInjectionError, match="No registration found for interface"
        ):
            self.container.resolve(UnregisteredService)
