"""
Unit tests for CLI module.

Tests command-line interface functionality including profiling scripts
and session listing using click testing utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pycroscope.cli import main, profile, list_sessions
from pycroscope.core.config import ProfileConfig
from pycroscope.core.session import ProfileSession


class TestCLIMain:
    """Test main CLI group functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.unit
    def test_main_command_help(self, runner):
        """Test main command shows help."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Pycroscope: Python performance analysis" in result.output
        assert "profile" in result.output
        assert "list-sessions" in result.output

    @pytest.mark.unit
    def test_main_command_version(self, runner):
        """Test version option."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestCLIProfile:
    """Test CLI profile command functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def test_script(self, tmp_path):
        """Create a simple test script to profile."""
        script = tmp_path / "test_script.py"
        script.write_text(
            """
def simple_function():
    total = 0
    for i in range(100):
        total += i * 2
    return total

if __name__ == "__main__":
    result = simple_function()
    print(f"Result: {result}")
"""
        )
        return script

    @pytest.mark.unit
    def test_profile_help(self, runner):
        """Test profile command help."""
        result = runner.invoke(main, ["profile", "--help"])
        assert result.exit_code == 0
        assert "Profile a Python script" in result.output
        assert "--line / --no-line" in result.output
        assert "--memory / --no-memory" in result.output
        assert "--call / --no-call" in result.output

    @pytest.mark.unit
    @patch("pycroscope.cli.ProfilerOrchestra")
    @patch("pycroscope.cli.ProfileSession")
    def test_profile_basic_usage(
        self, mock_session_class, mock_orchestra_class, runner, test_script, tmp_path
    ):
        """Test basic profile command usage."""
        # Setup mocks
        mock_session = Mock()
        mock_session.session_id = "test-session-123"
        mock_session.duration = 1.234
        mock_session.save.return_value = tmp_path / "session.json"
        mock_session_class.create.return_value = mock_session

        mock_orchestra = Mock()
        mock_orchestra.start_profiling.return_value = ["call", "memory"]
        mock_orchestra.stop_profiling.return_value = {
            "call": {"test": "data"},
            "memory": {"test": "data"},
        }
        mock_orchestra_class.return_value = mock_orchestra

        # Run command
        result = runner.invoke(main, ["profile", str(test_script)])

        # Verify
        assert result.exit_code == 0
        assert "Profiling" in result.output
        assert "test_script.py" in result.output
        assert "Profiling complete!" in result.output
        assert "Session ID: test-session-123" in result.output
        assert "Duration: 1.234s" in result.output
        assert "Profilers used: call, memory" in result.output

        # Verify mocks were called correctly
        mock_session_class.create.assert_called_once()
        mock_orchestra_class.assert_called_once_with(mock_session)
        mock_orchestra.start_profiling.assert_called_once()
        mock_orchestra.stop_profiling.assert_called_once()
        mock_session.complete.assert_called_once()

    @pytest.mark.unit
    @patch("pycroscope.cli.ProfilerOrchestra")
    @patch("pycroscope.cli.ProfileSession")
    def test_profile_with_options(
        self, mock_session_class, mock_orchestra_class, runner, test_script, tmp_path
    ):
        """Test profile command with various options."""
        # Setup mocks
        mock_session = Mock()
        mock_session.session_id = "test-session-456"
        mock_session.duration = 0.567
        mock_session.save.return_value = tmp_path / "session.json"
        mock_session_class.create.return_value = mock_session

        mock_orchestra = Mock()
        mock_orchestra.start_profiling.return_value = ["call"]
        mock_orchestra.stop_profiling.return_value = {"call": {"test": "data"}}
        mock_orchestra_class.return_value = mock_orchestra

        output_dir = tmp_path / "custom_output"

        # Run command with options
        result = runner.invoke(
            main,
            [
                "profile",
                str(test_script),
                "--no-line",
                "--no-memory",
                "--call",
                "--output-dir",
                str(output_dir),
            ],
        )

        # Verify
        assert result.exit_code == 0

        # Check that ProfileConfig was created with correct options
        call_args = mock_session_class.create.call_args[0][0]
        assert isinstance(call_args, ProfileConfig)
        assert call_args.line_profiling is False
        assert call_args.memory_profiling is False
        assert call_args.call_profiling is True
        assert call_args.output_dir == output_dir

    @pytest.mark.unit
    @patch("pycroscope.cli.ProfilerOrchestra")
    @patch("pycroscope.cli.ProfileSession")
    def test_profile_minimal_mode(
        self, mock_session_class, mock_orchestra_class, runner, test_script
    ):
        """Test profile command with minimal flag."""
        # Setup mocks
        mock_session = Mock()
        mock_session.session_id = "test-session-minimal"
        mock_session.duration = 0.123
        mock_session.save.return_value = Path("session.json")
        mock_session_class.create.return_value = mock_session

        mock_orchestra = Mock()
        mock_orchestra.start_profiling.return_value = ["call"]
        mock_orchestra.stop_profiling.return_value = {"call": {"test": "data"}}
        mock_orchestra_class.return_value = mock_orchestra

        # Run command with minimal flag
        result = runner.invoke(main, ["profile", str(test_script), "--minimal"])

        # Verify
        assert result.exit_code == 0

        # Check that config was modified for minimal mode
        call_args = mock_session_class.create.call_args[0][0]
        assert call_args.line_profiling is False
        assert call_args.create_visualizations is False
        assert call_args.analyze_patterns is False

    @pytest.mark.unit
    def test_profile_nonexistent_script(self, runner):
        """Test profile command with nonexistent script."""
        result = runner.invoke(main, ["profile", "nonexistent_script.py"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    @pytest.mark.unit
    @patch("pycroscope.cli.ProfilerOrchestra")
    @patch("pycroscope.cli.ProfileSession")
    def test_profile_no_profilers_started(
        self, mock_session_class, mock_orchestra_class, runner, test_script
    ):
        """Test profile command when no profilers start."""
        # Setup mocks
        mock_session = Mock()
        mock_session_class.create.return_value = mock_session

        mock_orchestra = Mock()
        mock_orchestra.start_profiling.return_value = []  # No profilers started
        mock_orchestra_class.return_value = mock_orchestra

        # Run command
        result = runner.invoke(main, ["profile", str(test_script)])

        # Should still exit cleanly even if no profilers start
        assert result.exit_code == 0
        assert "Profiling" in result.output


class TestCLIListSessions:
    """Test CLI list-sessions command functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.unit
    def test_list_sessions_help(self, runner):
        """Test list-sessions command help."""
        result = runner.invoke(main, ["list-sessions", "--help"])
        assert result.exit_code == 0
        assert "List saved profiling sessions" in result.output
        assert "--sessions-dir" in result.output

    @pytest.mark.unit
    def test_list_sessions_no_directory(self, runner):
        """Test list-sessions without specifying directory."""
        result = runner.invoke(main, ["list-sessions"])
        assert result.exit_code == 0
        assert "Please specify --sessions-dir" in result.output

    @pytest.mark.unit
    def test_list_sessions_empty_directory(self, runner, tmp_path):
        """Test list-sessions with empty directory."""
        result = runner.invoke(main, ["list-sessions", "--sessions-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No sessions found" in result.output

    @pytest.mark.unit
    def test_list_sessions_with_sessions(self, runner, tmp_path):
        """Test list-sessions with actual session files."""
        # Create sample session files
        session1_file = tmp_path / "profiling_data.json"
        session1_data = {
            "session_id": "session-1",
            "status": "completed",
            "duration": 1.234,
            "results": {"call": {"test": "data"}, "memory": {"test": "data"}},
        }
        session1_file.write_text(json.dumps(session1_data))

        session2_file = tmp_path / "profiling_data.json"
        session2_data = {
            "session_id": "session-2",
            "status": "running",
            "results": {"call": {"test": "data"}},
        }
        session2_file.write_text(json.dumps(session2_data))

        # Run command
        result = runner.invoke(main, ["list-sessions", "--sessions-dir", str(tmp_path)])

        # Verify
        assert result.exit_code == 0
        assert (
            "Found 1 sessions" in result.output
        )  # Only one file due to name collision
        assert "profiling_data.json" in result.output
        assert "Status: running" in result.output  # Last written data
        assert "Profilers: call" in result.output

    @pytest.mark.unit
    def test_list_sessions_with_invalid_json(self, runner, tmp_path):
        """Test list-sessions with invalid JSON file."""
        # Create invalid session file
        invalid_file = tmp_path / "profiling_data.json"
        invalid_file.write_text("invalid json content")

        # Run command
        result = runner.invoke(main, ["list-sessions", "--sessions-dir", str(tmp_path)])

        # Verify
        assert result.exit_code == 0
        assert "Found 1 sessions" in result.output
        assert "Error reading:" in result.output

    @pytest.mark.unit
    def test_list_sessions_nonexistent_directory(self, runner):
        """Test list-sessions with nonexistent directory."""
        result = runner.invoke(
            main, ["list-sessions", "--sessions-dir", "/nonexistent/path"]
        )
        assert result.exit_code != 0


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.mark.unit
    def test_cli_main_as_module(self):
        """Test that CLI main can be called as module."""
        # This tests the if __name__ == "__main__": main() part
        from pycroscope import cli

        # Mock main to avoid actually running it
        with patch.object(cli, "main") as mock_main:
            # Simulate running as main module
            old_name = cli.__name__
            try:
                cli.__name__ = "__main__"
                # Import and run the module's main block
                exec("if __name__ == '__main__': main()", cli.__dict__)
                mock_main.assert_called_once()
            finally:
                cli.__name__ = old_name
