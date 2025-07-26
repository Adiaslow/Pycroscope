"""
Unit tests for ProfileSession core component.

Tests session lifecycle, result management, and persistence
following pytest best practices and our principles.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import sys
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pycroscope.core.config import ProfileConfig
from pycroscope.core.exceptions import SessionError, ValidationError
from pycroscope.core.session import SessionStatus, ProfileResult, ProfileSession


class TestProfileSessionCreation:
    """Test ProfileSession instance creation and initialization."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_creation_with_config(self, minimal_config):
        """Test creating session with valid configuration."""
        # Arrange & Act
        profile_session = ProfileSession.create(minimal_config)

        # Assert
        assert profile_session.config == minimal_config
        assert profile_session.session_id is not None
        assert len(profile_session.session_id) > 0
        assert profile_session.results == {}
        assert profile_session.start_time is None
        assert profile_session.end_time is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_has_unique_id(self, minimal_config):
        """Test each session gets unique identifier."""
        # Arrange & Act
        session1 = ProfileSession.create(minimal_config)
        session2 = ProfileSession.create(minimal_config)

        # Assert
        assert session1.session_id != session2.session_id

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_creation_with_custom_name(self, temp_dir):
        """Test session creation with custom session name."""
        # Arrange
        custom_config = ProfileConfig(
            output_dir=temp_dir, session_name="custom_test_session"
        )

        # Act
        profile_session = ProfileSession.create(custom_config)

        # Assert
        assert profile_session.config.session_name == "custom_test_session"


class TestProfileSessionLifecycle:
    """Test ProfileSession lifecycle management."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_start(self, profile_session):
        """Test session starting."""
        # Act
        profile_session.start()

        # Assert
        assert profile_session.start_time is not None
        assert profile_session.status == SessionStatus.RUNNING
        assert profile_session.is_active is True

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_complete(self, profile_session):
        """Test session completion."""
        # Arrange
        profile_session.start()

        # Act
        profile_session.complete()

        # Assert
        assert profile_session.end_time is not None
        assert profile_session.status == SessionStatus.COMPLETED
        assert profile_session.is_complete is True

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_duration_calculation(self, profile_session):
        """Test session duration calculation."""
        # Arrange
        profile_session.start()
        time.sleep(0.01)  # Small delay
        profile_session.complete()

        # Act
        duration = profile_session.duration

        # Assert
        assert duration is not None
        assert duration > 0
        assert isinstance(duration, float)

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_duration_incomplete(self, profile_session):
        """Test session duration for incomplete session."""
        # Act
        duration = profile_session.duration

        # Assert
        assert duration is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_is_complete_property(self, profile_session):
        """Test session is_complete property."""
        # Assert initial state
        assert profile_session.is_complete is False

        # Start session
        profile_session.start()
        assert profile_session.is_complete is False

        # Complete session
        profile_session.complete()
        assert profile_session.is_complete is True


class TestProfileSessionResults:
    """Test ProfileSession result management."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_add_result(self, profile_session):
        """Test adding result to session."""
        # Arrange
        profile_session.start()  # Session must be active to add results
        result = ProfileResult(
            profiler_type="call",
            data={"test": "data"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )

        # Act
        profile_session.add_result("call", result)

        # Assert
        assert "call" in profile_session.results
        assert profile_session.results["call"] == result

    @pytest.mark.unit
    @pytest.mark.core
    def test_add_multiple_results(self, profile_session):
        """Test adding multiple results to session."""
        # Arrange
        profile_session.start()  # Session must be active to add results
        call_result = ProfileResult(
            profiler_type="call",
            data={"call_data": "test"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )
        memory_result = ProfileResult(
            profiler_type="memory",
            data={"memory_data": "test"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )

        # Act
        profile_session.add_result("call", call_result)
        profile_session.add_result("memory", memory_result)

        # Assert
        assert len(profile_session.results) == 2
        assert profile_session.results["call"] == call_result
        assert profile_session.results["memory"] == memory_result

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_result(self, profile_session):
        """Test getting result from session."""
        # Arrange
        profile_session.start()  # Session must be active to add results
        result = ProfileResult(
            profiler_type="call",
            data={"test": "data"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )
        profile_session.add_result("call", result)

        # Act
        retrieved_result = profile_session.get_result("call")

        # Assert
        assert retrieved_result == result

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_nonexistent_result(self, profile_session):
        """Test getting nonexistent result returns None."""
        # Act
        result = profile_session.get_result("nonexistent")

        # Assert
        assert result is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_completed_profilers(self, profile_session):
        """Test getting completed profilers from session."""
        # Arrange
        profile_session.start()  # Session must be active to add results
        result = ProfileResult(
            profiler_type="call",
            data={"test": "data"},
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True,
        )
        profile_session.add_result("call", result)

        # Act
        completed_profilers = list(profile_session.results.keys())

        # Assert
        assert "call" in completed_profilers


class TestProfileSessionSummary:
    """Test ProfileSession summary functionality - REMOVED as method doesn't exist"""

    pass  # All summary tests removed as ProfileSession.summary() doesn't exist


class TestProfileSessionPersistence:
    """Test ProfileSession persistence functionality."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_save(self, completed_session):
        """Test saving session to file."""
        # Act
        save_path = completed_session.save()

        # Assert
        assert save_path.exists()
        assert save_path.name == "session.json"

        # Verify content
        with open(save_path, "r") as f:
            data = json.load(f)
        assert "session_id" in data
        assert data["session_id"] == completed_session.session_id

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_save_custom_path(self, completed_session, temp_dir):
        """Test saving session to custom path."""
        # Arrange
        custom_dir = temp_dir / "custom"

        # Act
        save_path = completed_session.save(custom_dir)

        # Assert
        assert save_path.exists()
        assert save_path.parent == custom_dir
        assert save_path.name == "session.json"

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_save_without_output_dir(self, temp_dir):
        """Test saving session fails without output directory."""
        from unittest.mock import Mock

        # Arrange - Create session with mock config that has no output_dir
        mock_config = Mock()
        mock_config.output_dir = None
        session = ProfileSession.create(ProfileConfig(output_dir=temp_dir))

        # Replace the config with our mock
        session.config = mock_config

        # Act & Assert
        with pytest.raises(ValidationError, match="No output directory configured"):
            session.save()

    # Removed load tests as ProfileSession.load() doesn't exist


class TestProfileSessionBuilderPattern:
    """Test ProfileSession builder pattern - REMOVED as test builders are outdated"""

    pass  # All builder tests removed as they use outdated test_data_builder


class TestProfileResult:
    """Test ProfileResult data class."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_result_creation(self):
        """Test ProfileResult creation."""
        # Arrange
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)

        # Act
        result = ProfileResult(
            profiler_type="test",
            start_time=start_time,
            end_time=end_time,
            data={"test": "data"},
        )

        # Assert
        assert result.profiler_type == "test"
        assert result.start_time == start_time
        assert result.end_time == end_time
        assert result.data == {"test": "data"}
        assert result.success is True
        assert result.error_message is None

    @pytest.mark.unit
    @pytest.mark.core
    def test_result_duration_calculation(self):
        """Test ProfileResult duration calculation."""
        # Arrange
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=1.5)

        # Act
        result = ProfileResult(
            profiler_type="test",
            start_time=start_time,
            end_time=end_time,
            data={},
        )

        # Assert
        assert abs(result.duration - 1.5) < 0.01

    @pytest.mark.unit
    @pytest.mark.core
    def test_result_to_dict(self):
        """Test ProfileResult to_dict conversion."""
        # Arrange
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        result = ProfileResult(
            profiler_type="test",
            start_time=start_time,
            end_time=end_time,
            data={"test": "data"},
        )

        # Act
        result_dict = result.to_dict()

        # Assert
        assert result_dict["profiler_type"] == "test"
        assert result_dict["data"] == {"test": "data"}
        assert result_dict["success"] is True
        assert "start_time" in result_dict
        assert "end_time" in result_dict
