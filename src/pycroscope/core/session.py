"""
Profile session management and data models.

Manages profiling sessions with proper lifecycle, validation, and storage
following clean architecture principles.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import uuid

from .config import ProfileConfig
from .constants import SessionStatus
from .exceptions import SessionError, ValidationError


@dataclass
class ProfileResult:
    """
    Result from a single profiler execution.

    Contains profiling data with metadata following validation constraints.
    """

    profiler_type: str
    data: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    success: bool = True
    error_message: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "profiler_type": self.profiler_type,
            "data": self.data,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class ProfileSession:
    """
    Complete profiling session with results and metadata.

    Manages session lifecycle with strict validation and fail-fast behavior.
    """

    session_id: str
    config: ProfileConfig
    results: Dict[str, ProfileResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: SessionStatus = SessionStatus.PENDING

    def __post_init__(self):
        """Initialize session with output directory."""
        if self.config.output_dir:
            # Ensure output directory exists but don't create timestamped subdirectories
            # Single directory per profiler run as requested
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def duration(self) -> Optional[float]:
        """Get total session duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def is_complete(self) -> bool:
        """Check if session is complete."""
        return self.status == SessionStatus.COMPLETED

    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status == SessionStatus.RUNNING

    def add_result(self, profiler_type: str, result: ProfileResult) -> None:
        """
        Add profiler result to session.

        Args:
            profiler_type: Type of profiler (call, memory, line, etc.)
            result: ProfileResult instance

        Raises:
            SessionError: If session is not active
            ValidationError: If result is invalid
        """
        if not self.is_active:
            raise SessionError(f"Cannot add result to inactive session: {self.status}")

        if not isinstance(result, ProfileResult):
            raise ValidationError("Result must be a ProfileResult instance")

        self.results[profiler_type] = result

    def get_result(self, profiler_type: str) -> Optional[ProfileResult]:
        """Get result for specific profiler type."""
        return self.results.get(profiler_type)

    def start(self) -> None:
        """Mark session as started."""
        if self.status != SessionStatus.PENDING:
            raise SessionError(f"Cannot start session in status: {self.status}")

        self.start_time = datetime.now()
        self.status = SessionStatus.RUNNING

    def complete(self) -> None:
        """Mark session as completed."""
        if self.status != SessionStatus.RUNNING:
            raise SessionError(f"Cannot complete session in status: {self.status}")

        self.end_time = datetime.now()
        self.status = SessionStatus.COMPLETED

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save session to JSON file.

        Args:
            output_dir: Directory to save to (uses session config if None)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            if self.config.output_dir is None:
                raise ValidationError("No output directory configured for session save")
            output_dir = self.config.output_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use clean filename without session ID
        session_file = output_dir / "profiling_data.json"

        session_data = {
            "session_id": self.session_id,
            "config": self.config.model_dump(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "status": self.status.value,
            "results": {
                profiler_type: result.to_dict()
                for profiler_type, result in self.results.items()
            },
        }

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)

        return session_file

    @classmethod
    def create(cls, config: ProfileConfig) -> "ProfileSession":
        """
        Create new session with generated ID.

        Args:
            config: Profile configuration

        Returns:
            New ProfileSession instance
        """
        session_id = str(uuid.uuid4())
        return cls(session_id=session_id, config=config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "config": self.config.model_dump(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "status": self.status.value,
            "results": {
                profiler_type: result.to_dict()
                for profiler_type, result in self.results.items()
            },
        }
