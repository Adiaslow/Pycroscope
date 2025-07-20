"""
Unit tests for CLI commands.

Tests command creation, argument parsing, and basic execution logic.
"""

import pytest
from argparse import ArgumentParser, Namespace
from unittest.mock import Mock, patch, MagicMock

from pycroscope.cli.commands import (
    BaseCommand,
    ProfileCommand,
    ListCommand,
    CompareCommand,
    StatusCommand,
    ExportCommand,
    AnalyzeCommand,
    DeleteCommand,
    CleanupCommand,
    ConfigCommand,
)
from pycroscope.cli.formatters import ResultFormatter
from pycroscope.core.config import ProfileConfig


class TestBaseCommand:
    """Test BaseCommand abstract base class."""

    def test_creation(self):
        """Test BaseCommand creation with config and formatter."""
        config = ProfileConfig()
        formatter = ResultFormatter()

        # Cannot instantiate abstract base class directly
        with pytest.raises(TypeError):
            BaseCommand(config, formatter)


class TestProfileCommand:
    """Test ProfileCommand implementation."""

    @pytest.fixture
    def command(self, sample_config):
        """ProfileCommand instance for testing."""
        formatter = ResultFormatter()
        return ProfileCommand(sample_config, formatter)

    def test_add_arguments(self):
        """Test ProfileCommand argument parser setup."""
        parser = ArgumentParser()
        ProfileCommand.add_arguments(parser)

        # Should have target argument
        args = parser.parse_args(["test_script.py"])
        assert args.target == "test_script.py"

    def test_add_arguments_with_options(self):
        """Test ProfileCommand with optional arguments."""
        parser = ArgumentParser()
        ProfileCommand.add_arguments(parser)

        args = parser.parse_args(
            [
                "test_script.py",
                "--session-id",
                "custom_session",
                "--timeout",
                "120",
                "--analyze",
            ]
        )

        assert args.target == "test_script.py"
        assert args.session_id == "custom_session"
        assert args.timeout == 120
        assert args.analyze is True

    def test_execute_file_not_found(self, command):
        """Test ProfileCommand execution with non-existent file."""
        args = Namespace(
            target="nonexistent_script.py",
            args=None,
            session_id=None,
            timeout=300,
            output_dir=None,
            analyze=False,
            no_analysis=False,
        )

        result = command.execute(args)
        assert result == 1  # Error due to file not found

    @patch("pycroscope.cli.commands.subprocess.Popen")
    @patch("pycroscope.cli.commands.ProfilerSuite")
    @patch("pycroscope.cli.commands.Path")
    def test_execute_basic(self, mock_path, mock_profiler_suite, mock_popen, command):
        """Test basic ProfileCommand execution with successful profiling."""
        # Mock file existence check
        mock_target_path = Mock()
        mock_target_path.exists.return_value = True
        mock_target_path.__str__ = Mock(return_value="test_script.py")
        mock_path.return_value = mock_target_path

        # Mock subprocess execution - return successful execution
        mock_process = Mock()
        mock_process.communicate.return_value = ("stdout output", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Mock profiler suite with proper context management and string-formattable values
        mock_profiler = MagicMock()
        mock_profiler_suite.return_value = mock_profiler

        # Mock the context manager methods
        mock_profiler.__enter__.return_value = mock_profiler
        mock_profiler.__exit__.return_value = None

        # Mock current_session with values that can be formatted in strings and used in calculations
        mock_session = Mock()
        mock_session.session_id = "test_session_123"  # String value
        mock_session.total_events = 42  # Integer value
        mock_session.peak_memory = 1048576  # Integer value (1 MB in bytes)
        mock_profiler.current_session = mock_session

        # Create mock args
        args = Namespace(
            target="test_script.py",
            args=None,
            session_id=None,
            timeout=300,
            output_dir=None,
            analyze=False,
            no_analysis=False,
        )

        # Execute command
        result = command.execute(args)

        # Verify basic command execution succeeded
        assert result == 0
        mock_path.assert_called_once_with("test_script.py")
        mock_profiler_suite.assert_called_once_with(command.config)


class TestListCommand:
    """Test ListCommand implementation."""

    @pytest.fixture
    def command(self, sample_config):
        """ListCommand instance for testing."""
        formatter = ResultFormatter()
        return ListCommand(sample_config, formatter)

    def test_add_arguments(self):
        """Test ListCommand argument parser setup."""
        parser = ArgumentParser()
        ListCommand.add_arguments(parser)

        # Test with package filter
        args = parser.parse_args(["--package", "mypackage", "--limit", "10"])
        assert args.package == "mypackage"
        assert args.limit == 10

    @patch("pycroscope.cli.commands.FileDataStore")
    def test_execute_empty_list(self, mock_store_class, command):
        """Test ListCommand with no sessions."""
        # Mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.list_sessions.return_value = []

        args = Namespace(package=None, limit=20, storage_path=None)

        result = command.execute(args)
        assert result == 0


class TestStatusCommand:
    """Test StatusCommand implementation."""

    @pytest.fixture
    def command(self, sample_config):
        """StatusCommand instance for testing."""
        formatter = ResultFormatter()
        return StatusCommand(sample_config, formatter)

    def test_add_arguments(self):
        """Test StatusCommand argument parser setup."""
        parser = ArgumentParser()
        StatusCommand.add_arguments(parser)

        args = parser.parse_args(["--storage-path", "/custom/path"])
        assert args.storage_path == "/custom/path"

    @patch("pycroscope.cli.commands.FileDataStore")
    def test_execute(self, mock_store_class, command):
        """Test StatusCommand execution."""
        # Mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.get_storage_stats.return_value = {
            "storage_path": "/test/path",
            "total_sessions": 5,
            "total_size_mb": 10.5,
            "compression_enabled": True,
        }

        args = Namespace(storage_path=None)

        result = command.execute(args)
        assert result == 0


class TestConfigCommand:
    """Test ConfigCommand implementation."""

    @pytest.fixture
    def command(self, sample_config):
        """ConfigCommand instance for testing."""
        formatter = ResultFormatter()
        return ConfigCommand(sample_config, formatter)

    def test_add_arguments(self):
        """Test ConfigCommand argument parser setup."""
        parser = ArgumentParser()
        ConfigCommand.add_arguments(parser)

        # Test show subcommand
        args = parser.parse_args(["show", "--section", "analysis"])
        assert args.config_action == "show"
        assert args.section == "analysis"

    def test_execute_show_config(self, command):
        """Test ConfigCommand show action."""
        args = Namespace(config_action="show", section=None)

        result = command.execute(args)
        assert result == 0

    def test_execute_unknown_action(self, command):
        """Test ConfigCommand with unknown action."""
        args = Namespace(config_action=None)

        result = command.execute(args)
        assert result == 1


class TestDeleteCommand:
    """Test DeleteCommand implementation."""

    @pytest.fixture
    def command(self, sample_config):
        """DeleteCommand instance for testing."""
        formatter = ResultFormatter()
        return DeleteCommand(sample_config, formatter)

    def test_add_arguments(self):
        """Test DeleteCommand argument parser setup."""
        parser = ArgumentParser()
        DeleteCommand.add_arguments(parser)

        args = parser.parse_args(["session1", "session2", "--force", "--dry-run"])

        assert args.session_ids == ["session1", "session2"]
        assert args.force is True
        assert args.dry_run is True

    @patch("pycroscope.cli.commands.FileDataStore")
    def test_execute_with_force(self, mock_store_class, command):
        """Test DeleteCommand with force flag."""
        # Mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.get_session_metadata.return_value = {"session_id": "test_session"}
        mock_store.delete_session.return_value = True

        args = Namespace(
            session_ids=["test_session"], storage_path=None, force=True, dry_run=False
        )

        result = command.execute(args)
        assert result == 0


class TestCleanupCommand:
    """Test CleanupCommand implementation."""

    @pytest.fixture
    def command(self, sample_config):
        """CleanupCommand instance for testing."""
        formatter = ResultFormatter()
        return CleanupCommand(sample_config, formatter)

    def test_add_arguments(self):
        """Test CleanupCommand argument parser setup."""
        parser = ArgumentParser()
        CleanupCommand.add_arguments(parser)

        args = parser.parse_args(
            ["--max-age-days", "30", "--max-sessions", "50", "--dry-run", "--force"]
        )

        assert args.max_age_days == 30
        assert args.max_sessions == 50
        assert args.dry_run is True
        assert args.force is True

    @patch("pycroscope.cli.commands.FileDataStore")
    def test_execute_with_force(self, mock_store_class, command):
        """Test CleanupCommand with force flag."""
        # Mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.cleanup_old_sessions.return_value = 3
        mock_store.list_sessions.return_value = ["session1", "session2"]
        # Add missing mock for get_storage_stats with proper numeric values
        mock_store.get_storage_stats.return_value = {
            "storage_path": "/test/path",
            "total_sessions": 5,
            "total_size_mb": 10.5,
            "compression_enabled": True,
        }

        args = Namespace(
            max_age_days=30,
            max_sessions=None,
            storage_path=None,
            dry_run=False,
            force=True,
        )

        result = command.execute(args)
        assert result == 0
