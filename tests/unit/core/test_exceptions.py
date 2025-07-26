"""
Tests for custom exception hierarchy.

Tests the exception system for proper error handling, context information,
and fail-fast behavior following clean architecture principles.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pycroscope.core.exceptions import (
    PycroscopeError,
    ConfigurationError,
    ProfilerError,
    ProfilerConflictError,
    ProfilerNotAvailableError,
    ProfilerInitializationError,
    SessionError,
    SessionNotFoundError,
    SessionAlreadyActiveError,
    SessionStateError,
    DependencyInjectionError,
    FactoryError,
    ValidationError,
    ResourceError,
)


class TestPycroscopeError:
    """Test base exception class."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic exception construction."""
        # Arrange & Act
        error = PycroscopeError("Test error message")

        # Assert
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.cause is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_error_code(self):
        """Test exception construction with error code."""
        # Arrange & Act
        error = PycroscopeError("Test error", error_code="TEST_ERROR")

        # Assert
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_cause(self):
        """Test exception construction with cause."""
        # Arrange
        cause = ValueError("Original error")

        # Act
        error = PycroscopeError("Wrapped error", cause=cause)

        # Assert
        assert error.message == "Wrapped error"
        assert error.cause is cause

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_exception(self):
        """Test that PycroscopeError inherits from Exception."""
        # Arrange & Act
        error = PycroscopeError("Test")

        # Assert
        assert isinstance(error, Exception)


class TestConfigurationError:
    """Test configuration-related errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic configuration error construction."""
        # Arrange & Act
        error = ConfigurationError("Invalid configuration")

        # Assert
        assert str(error) == "Invalid configuration"
        assert error.error_code == "CONFIG_ERROR"
        assert error.config_key is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_config_key(self):
        """Test construction with specific config key."""
        # Arrange & Act
        error = ConfigurationError("Invalid value", config_key="output_dir")

        # Assert
        assert error.message == "Invalid value"
        assert error.config_key == "output_dir"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = ConfigurationError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestProfilerError:
    """Test profiler-related errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic profiler error construction."""
        # Arrange & Act
        error = ProfilerError("Profiler failed")

        # Assert
        assert str(error) == "Profiler failed"
        assert error.profiler_type is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_profiler_type(self):
        """Test construction with profiler type."""
        # Arrange & Act
        error = ProfilerError("Call profiler failed", profiler_type="call")

        # Assert
        assert error.message == "Call profiler failed"
        assert error.profiler_type == "call"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = ProfilerError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestProfilerConflictError:
    """Test profiler conflict errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic profiler conflict error construction."""
        # Arrange & Act
        error = ProfilerConflictError("Profiler conflict detected")

        # Assert
        assert str(error) == "Profiler conflict detected"
        assert error.error_code == "PROFILER_CONFLICT"
        assert error.conflicting_profilers == []

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_conflicting_profilers(self):
        """Test construction with conflicting profilers list."""
        # Arrange
        conflicts = ["line", "call"]

        # Act
        error = ProfilerConflictError(
            "Multiple conflicts", conflicting_profilers=conflicts
        )

        # Assert
        assert error.conflicting_profilers == conflicts

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_profiler_error(self):
        """Test inheritance from ProfilerError."""
        # Arrange & Act
        error = ProfilerConflictError("Test")

        # Assert
        assert isinstance(error, ProfilerError)


class TestProfilerNotAvailableError:
    """Test profiler availability errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic profiler not available error construction."""
        # Arrange & Act
        error = ProfilerNotAvailableError("Profiler not available")

        # Assert
        assert str(error) == "Profiler not available"
        assert error.error_code == "PROFILER_UNAVAILABLE"
        assert error.missing_dependencies == []

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_missing_dependencies(self):
        """Test construction with missing dependencies list."""
        # Arrange
        deps = ["line_profiler", "psutil"]

        # Act
        error = ProfilerNotAvailableError(
            "Missing dependencies", missing_dependencies=deps
        )

        # Assert
        assert error.missing_dependencies == deps

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_profiler_error(self):
        """Test inheritance from ProfilerError."""
        # Arrange & Act
        error = ProfilerNotAvailableError("Test")

        # Assert
        assert isinstance(error, ProfilerError)


class TestProfilerInitializationError:
    """Test profiler initialization errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic profiler initialization error construction."""
        # Arrange & Act
        error = ProfilerInitializationError("Failed to initialize")

        # Assert
        assert str(error) == "Failed to initialize"
        assert error.error_code == "PROFILER_INIT_FAILED"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_profiler_error(self):
        """Test inheritance from ProfilerError."""
        # Arrange & Act
        error = ProfilerInitializationError("Test")

        # Assert
        assert isinstance(error, ProfilerError)


class TestSessionError:
    """Test session-related errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic session error construction."""
        # Arrange & Act
        error = SessionError("Session error")

        # Assert
        assert str(error) == "Session error"
        assert error.session_id is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_session_id(self):
        """Test construction with session ID."""
        # Arrange & Act
        error = SessionError("Session failed", session_id="test-session-123")

        # Assert
        assert error.message == "Session failed"
        assert error.session_id == "test-session-123"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = SessionError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestSessionNotFoundError:
    """Test session not found errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_session_id(self):
        """Test construction with session ID."""
        # Arrange & Act
        error = SessionNotFoundError("missing-session-456")

        # Assert
        assert "Session not found: missing-session-456" in str(error)
        assert error.session_id == "missing-session-456"
        assert error.error_code == "SESSION_NOT_FOUND"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_session_error(self):
        """Test inheritance from SessionError."""
        # Arrange & Act
        error = SessionNotFoundError("test-id")

        # Assert
        assert isinstance(error, SessionError)


class TestSessionAlreadyActiveError:
    """Test session already active errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_session_id(self):
        """Test construction with session ID."""
        # Arrange & Act
        error = SessionAlreadyActiveError("active-session-789")

        # Assert
        assert "Session already active: active-session-789" in str(error)
        assert error.session_id == "active-session-789"
        assert error.error_code == "SESSION_ALREADY_ACTIVE"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_session_error(self):
        """Test inheritance from SessionError."""
        # Arrange & Act
        error = SessionAlreadyActiveError("test-id")

        # Assert
        assert isinstance(error, SessionError)


class TestDependencyInjectionError:
    """Test dependency injection errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic dependency injection error construction."""
        # Arrange & Act
        error = DependencyInjectionError("DI error")

        # Assert
        assert str(error) == "DI error"
        assert error.error_code == "DI_ERROR"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = DependencyInjectionError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestFactoryError:
    """Test factory-related errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic factory error construction."""
        # Arrange & Act
        error = FactoryError("Factory failed")

        # Assert
        assert str(error) == "Factory failed"
        assert error.factory_type is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_factory_type(self):
        """Test construction with factory type."""
        # Arrange & Act
        error = FactoryError("Factory failed", factory_type="CallProfilerFactory")

        # Assert
        assert error.message == "Factory failed"
        assert error.factory_type == "CallProfilerFactory"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = FactoryError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestValidationError:
    """Test validation errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic validation error construction."""
        # Arrange & Act
        error = ValidationError("Validation failed")

        # Assert
        assert str(error) == "Validation failed"
        assert error.error_code == "VALIDATION_ERROR"

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_validation_errors(self):
        """Test construction with validation errors list."""
        # Arrange & Act
        validation_errors = [
            "Field 'output_dir' is required",
            "Field 'name' is too long",
        ]
        error = ValidationError(
            "Validation failed", validation_errors=validation_errors
        )

        # Assert
        assert error.message == "Validation failed"
        assert error.validation_errors == validation_errors
        assert "Field 'output_dir' is required" in str(error)

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = ValidationError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestResourceError:
    """Test resource-related errors."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_basic_construction(self):
        """Test basic resource error construction."""
        # Arrange & Act
        error = ResourceError("Resource error")

        # Assert
        assert str(error) == "Resource error"
        assert error.error_code == "RESOURCE_ERROR"

    @pytest.mark.unit
    @pytest.mark.core
    def test_construction_with_resource_type(self):
        """Test construction with resource type."""
        # Arrange & Act
        error = ResourceError("File not found", resource_type="file")

        # Assert
        assert error.message == "File not found"
        assert error.resource_type == "file"

    @pytest.mark.unit
    @pytest.mark.core
    def test_inheritance_from_pycroscope_error(self):
        """Test inheritance from PycroscopeError."""
        # Arrange & Act
        error = ResourceError("Test")

        # Assert
        assert isinstance(error, PycroscopeError)


class TestExceptionContext:
    """Test exception context and chaining."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_exception_chaining(self):
        """Test proper exception chaining with cause."""
        # Arrange
        original_error = ValueError("Original problem")

        # Act
        wrapped_error = ProfilerError("Profiler failed", cause=original_error)

        # Assert
        assert wrapped_error.cause is original_error
        assert isinstance(wrapped_error.cause, ValueError)

    @pytest.mark.unit
    @pytest.mark.core
    def test_exception_context_preservation(self):
        """Test that exception context is preserved."""
        # Arrange
        session_id = "test-session-123"
        profiler_type = "line"

        # Act
        error = ProfilerError(
            "Line profiler failed for session",
            profiler_type=profiler_type,
            error_code="CUSTOM_ERROR",
        )

        # Assert
        assert error.profiler_type == profiler_type
        assert error.error_code == "CUSTOM_ERROR"
        assert "Line profiler failed for session" in str(error)

    @pytest.mark.unit
    @pytest.mark.core
    def test_error_hierarchy_inheritance(self):
        """Test proper inheritance hierarchy."""
        # Arrange & Act
        config_error = ConfigurationError("Config error")
        profiler_error = ProfilerError("Profiler error")
        session_error = SessionError("Session error")

        # Assert
        assert isinstance(config_error, PycroscopeError)
        assert isinstance(profiler_error, PycroscopeError)
        assert isinstance(session_error, PycroscopeError)

        # Test specific inheritance
        conflict_error = ProfilerConflictError("Conflict")
        assert isinstance(conflict_error, ProfilerError)
        assert isinstance(conflict_error, PycroscopeError)

    @pytest.mark.unit
    @pytest.mark.core
    def test_fail_fast_behavior(self):
        """Test that exceptions support fail-fast behavior."""
        # This test demonstrates that exceptions can be raised immediately
        # without being caught and wrapped, supporting fail-fast principles

        # Act & Assert
        with pytest.raises(ProfilerError) as exc_info:
            raise ProfilerError("Immediate failure", profiler_type="test")

        assert exc_info.value.profiler_type == "test"
        assert "Immediate failure" in str(exc_info.value)
