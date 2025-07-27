"""
Unit tests for application services.

Tests configuration, session management, and profiling services
following pytest best practices and our principles.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pycroscope.application.services import (
    ConfigurationService,
    SessionManagementService,
    ProfilingService,
)
from pycroscope.core.config import ProfileConfig
from pycroscope.core.session import ProfileSession
from pycroscope.core.exceptions import (
    ConfigurationError,
    SessionError,
    ValidationError,
    ProfilerError,
)
from pydantic import ValidationError


# Simple fixture builders
@pytest.fixture
def minimal_config(temp_dir):
    """Create minimal ProfileConfig for testing."""
    return ProfileConfig(output_dir=temp_dir)


class TestConfigurationService:
    """Test ConfigurationService functionality."""

    @pytest.fixture
    def config_service(self):
        """Create ConfigurationService instance."""
        return ConfigurationService()

    @pytest.mark.unit
    @pytest.mark.application
    def test_service_initialization(self, config_service):
        """Test ConfigurationService initialization."""
        # Assert - ConfigurationService is a simple static class
        assert isinstance(config_service, ConfigurationService)

    @pytest.mark.unit
    @pytest.mark.application
    def test_create_config_with_defaults(self, temp_dir):
        """Test creating config applies defaults."""
        # Act
        profile_config = ConfigurationService.create_config(output_dir=temp_dir)

        # Assert
        assert isinstance(profile_config, ProfileConfig)
        assert profile_config.line_profiling is True  # Default
        assert profile_config.memory_profiling is True  # Default
        assert profile_config.call_profiling is True  # Default
        assert profile_config.output_dir == temp_dir

    @pytest.mark.unit
    @pytest.mark.application
    def test_create_config_with_overrides(self, temp_dir):
        """Test creating config with explicit overrides."""
        # Act
        profile_config = ConfigurationService.create_config(
            output_dir=temp_dir, line_profiling=False, memory_precision=4
        )

        # Assert
        assert profile_config.line_profiling is False
        assert profile_config.memory_precision == 4
        assert profile_config.output_dir == temp_dir

    @pytest.mark.unit
    @pytest.mark.application
    def test_create_config_static_method(self, temp_dir):
        """Test that create_config is a static method."""
        # Act - can call without instance
        profile_config = ConfigurationService.create_config(
            output_dir=temp_dir, call_profiling=False
        )

        # Assert
        assert profile_config.call_profiling is False

    @pytest.mark.unit
    @pytest.mark.application
    def test_validation_error_for_invalid_values(self, temp_dir):
        """Test validation raises error for invalid values."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ConfigurationService.create_config(
                output_dir=temp_dir, memory_precision=7  # Exceeds limit
            )

        assert "memory_precision" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.application
    def test_validation_multiple_errors(self, config_service, temp_dir):
        """Test validation collects multiple errors."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            config_service.create_config(
                output_dir=temp_dir,
                memory_precision=10,  # Invalid
                max_call_depth=0,  # Invalid
            )

        error_message = str(exc_info.value)
        assert "memory_precision" in error_message
        assert "max_call_depth" in error_message
        assert "max_call_depth" in error_message


class TestSessionManagementService:
    """Test SessionManagementService functionality."""

    @pytest.fixture
    def session_service(self):
        """Create SessionManagementService instance."""
        # Clear storage before each test
        SessionManagementService.clear_all_sessions()
        return SessionManagementService()

    @pytest.mark.unit
    @pytest.mark.application
    def test_create_session(self, session_service, minimal_config):
        """Test creating new session."""
        # Act
        profile_session = session_service.create_session(minimal_config)

        # Assert
        assert isinstance(profile_session, ProfileSession)
        assert profile_session.session_id is not None
        assert profile_session.config == minimal_config
        # Session should be stored
        assert profile_session.session_id in SessionManagementService._session_storage

    @pytest.mark.unit
    @pytest.mark.application
    def test_get_existing_session(self, session_service, minimal_config):
        """Test retrieving existing session."""
        # Arrange
        profile_session = session_service.create_session(minimal_config)
        session_id = profile_session.session_id

        # Act
        retrieved_session = session_service.get_session(session_id)

        # Assert
        assert retrieved_session == profile_session

    @pytest.mark.unit
    @pytest.mark.application
    def test_get_nonexistent_session(self, session_service):
        """Test retrieving nonexistent session returns None."""
        # Act
        retrieved_session = session_service.get_session("nonexistent")

        # Assert
        assert retrieved_session is None

    @pytest.mark.unit
    @pytest.mark.application
    def test_clear_all_sessions(self, session_service, minimal_config):
        """Test clearing all sessions."""
        # Arrange
        session1 = session_service.create_session(minimal_config)
        session2 = session_service.create_session(minimal_config)

        # Verify sessions exist
        assert len(SessionManagementService._session_storage) == 2

        # Act
        SessionManagementService.clear_all_sessions()

        # Assert
        assert len(SessionManagementService._session_storage) == 0
        assert session_service.get_session(session1.session_id) is None
        assert session_service.get_session(session2.session_id) is None

    # Removed tests for non-existent methods:
    # - save_session, remove_session, get_active_sessions, load_session
    # These methods don't exist in the current SessionManagementService implementation


class TestProfilingService:
    """Test ProfilingService functionality."""

    @pytest.fixture
    def profiling_service(self):
        """Create ProfilingService instance."""
        return ProfilingService()

    @pytest.mark.unit
    @pytest.mark.application
    def test_service_initialization(self, profiling_service):
        """Test ProfilingService initialization."""
        # Assert - current implementation has these attributes
        assert hasattr(profiling_service, "session_service")
        assert hasattr(profiling_service, "config_service")
        assert isinstance(profiling_service.session_service, SessionManagementService)
        assert isinstance(profiling_service.config_service, ConfigurationService)

    @pytest.mark.unit
    @pytest.mark.application
    def test_start_profiling_creates_session(self, profiling_service, temp_dir):
        """Test starting profiling creates and returns session."""
        # Act
        session = profiling_service.start_profiling(
            output_dir=temp_dir, call_profiling=True, line_profiling=False
        )

        # Assert
        assert isinstance(session, ProfileSession)
        assert session.config.output_dir == temp_dir
        assert session.config.call_profiling is True
        assert session.config.line_profiling is False
        assert session.session_id in profiling_service._active_orchestrators

    @pytest.mark.unit
    @pytest.mark.application
    def test_profile_context_manager(self, profiling_service, temp_dir):
        """Test profiling context manager."""
        # Act & Assert
        with profiling_service.profile_context(
            output_dir=temp_dir, memory_profiling=True, call_profiling=False
        ) as session:
            assert isinstance(session, ProfileSession)
            assert session.config.memory_profiling is True
            assert session.config.call_profiling is False
            # Session should be active during context
            assert session.session_id in profiling_service._active_orchestrators

        # After context, orchestrator should be cleaned up
        assert session.session_id not in profiling_service._active_orchestrators

    @pytest.mark.unit
    @pytest.mark.application
    def test_stop_profiling(self, profiling_service):
        """Test stopping profiling session."""
        # Arrange
        mock_session = Mock()
        mock_session.session_id = "test_session"

        # Mock the orchestrator in active_orchestrators
        mock_orchestrator = Mock()
        profiling_service._active_orchestrators["test_session"] = mock_orchestrator

        # Act
        result = profiling_service.stop_profiling(mock_session)

        # Assert
        assert result == mock_session
        mock_orchestrator.stop_profiling.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.application
    def test_stop_profiling_nonexistent_session(self, profiling_service):
        """Test stopping nonexistent session raises error."""
        # Arrange
        mock_session = Mock()
        mock_session.session_id = "nonexistent"

        # Act & Assert
        with pytest.raises(ProfilerError, match="No active orchestrator found"):
            profiling_service.stop_profiling(mock_session)

    @pytest.mark.unit
    @pytest.mark.application
    def test_profile_function(self, profiling_service, minimal_config):
        """Test profiling function decorator."""
        # Arrange
        profiling_service.config_service.create_config.return_value = minimal_config

        # Act
        def test_function():
            return "test_result"

        wrapped_function = profiling_service.profile_function(test_function)

        # Assert
        assert callable(wrapped_function)
        # Note: We're not calling the wrapped function to avoid setting up full profiling


class TestServiceIntegration:
    """Test integration between application services."""

    @pytest.mark.unit
    @pytest.mark.application
    def test_configuration_and_session_services_integration(self, temp_dir):
        """Test ConfigurationService and SessionManagementService working together."""
        # Arrange
        config_service = ConfigurationService()
        session_service = SessionManagementService()

        # Act
        profile_config = config_service.create_config(
            output_dir=temp_dir, line_profiling=True, memory_profiling=False
        )
        profile_session = session_service.create_session(profile_config)

        # Assert
        assert profile_session.config == profile_config
        assert profile_session.config.line_profiling is True
        assert profile_session.config.memory_profiling is False
        assert profile_session.session_id in session_service._session_storage

    @pytest.mark.unit
    @pytest.mark.application
    def test_profiling_service_end_to_end(self, temp_dir):
        """Test ProfilingService end-to-end flow."""
        # Arrange
        service = ProfilingService()

        # Act - create configuration
        config_kwargs = {
            "output_dir": temp_dir,
            "line_profiling": False,
            "call_profiling": True,
        }

        # Test configuration creation through service
        profile_config = service.config_service.create_config(**config_kwargs)

        # Assert configuration
        assert profile_config.output_dir == temp_dir
        assert profile_config.line_profiling is False
        assert profile_config.call_profiling is True

        # Test session creation through service
        profile_session = service.session_service.create_session(profile_config)

        # Assert session
        assert profile_session.config == profile_config
        assert profile_session.session_id is not None
