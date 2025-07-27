"""
Output management for Pycroscope profiling sessions.

Handles file output, directory management, and result serialization
following clean architecture principles.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from ...core.session import ProfileSession
from ...core.exceptions import ConfigurationError, ResourceError


class OutputManager:
    """
    Manages output operations for profiling sessions.

    Handles file creation, directory management, and result serialization
    with no fallbacks.
    """

    def __init__(self, session: ProfileSession):
        self.session = session
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate output configuration."""
        if self.session.config.output_dir is None:
            raise ConfigurationError(
                "Output directory must be configured", config_key="output_dir"
            )

    def get_output_dir(self) -> Path:
        """Get validated output directory."""
        # Rely on _validate_configuration() which is called in __init__
        return self.session.config.output_dir

    def create_session_directory(self) -> Path:
        """Create session output directory."""
        output_dir = self.get_output_dir()
        session_dir = output_dir / f"session_{self.session.session_id}"

        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir
        except OSError as e:
            raise ResourceError(
                f"Failed to create session directory: {session_dir}",
                resource_type="directory",
                resource_path=str(session_dir),
                cause=e,
            )

    def save_session_results(self) -> Path:
        """Save session results to file."""
        session_dir = self.create_session_directory()
        results_file = session_dir / "results.json"

        try:
            with open(results_file, "w") as f:
                json.dump(self.session.to_dict(), f, indent=2, default=str)
            return results_file
        except (OSError, ValueError) as e:
            raise ResourceError(
                f"Failed to save session results: {results_file}",
                resource_type="file",
                resource_path=str(results_file),
                cause=e,
            )

    def save_profiler_output(self, profiler_type: str, data: Dict[str, Any]) -> Path:
        """Save profiler output."""
        session_dir = self.create_session_directory()
        output_file = session_dir / f"{profiler_type}_output.json"

        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            return output_file
        except (OSError, ValueError) as e:
            raise ResourceError(
                f"Failed to save {profiler_type} output: {output_file}",
                resource_type="file",
                resource_path=str(output_file),
                cause=e,
            )

    def create_summary_report(self) -> Path:
        """Create session summary report."""
        session_dir = self.create_session_directory()
        summary_file = session_dir / "summary.json"

        summary_data = {
            "session_id": self.session.session_id,
            "status": self.session.status,
            "duration": self.session.duration,
            "profilers": list(self.session.results.keys()),
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "line_profiling": self.session.config.line_profiling,
                "memory_profiling": self.session.config.memory_profiling,
                "call_profiling": self.session.config.call_profiling,
            },
        }

        try:
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)
            return summary_file
        except (OSError, ValueError) as e:
            raise ResourceError(
                f"Failed to create summary report: {summary_file}",
                resource_type="file",
                resource_path=str(summary_file),
                cause=e,
            )
