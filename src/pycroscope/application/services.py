"""
Application services for Pycroscope.

High-level application services that coordinate domain logic and infrastructure.
"""

from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from ..core.config import ProfileConfig
from ..core.session import ProfileSession
from ..core.exceptions import ProfilerError
from ..infrastructure.profilers.orchestra import ProfilerOrchestra


class ConfigurationService:
    """Service for managing profiling configuration."""

    @staticmethod
    def create_config(**kwargs) -> ProfileConfig:
        """Create ProfileConfig with validation and defaults."""
        config_data = ConfigurationService._apply_defaults(kwargs)
        return ProfileConfig(**config_data)

    @staticmethod
    def _apply_defaults(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration."""
        defaults = {
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": True,
            "sampling_profiling": False,
            "output_dir": Path.cwd() / "pycroscope_output",
            "save_raw_data": True,
            "sampling_interval": 0.01,
            "memory_precision": 3,
            "max_call_depth": 50,
            "generate_reports": True,
            "create_visualizations": True,
            "analyze_patterns": True,
            "profiler_prefix": "pycroscope",
            "use_thread_isolation": True,
            "cleanup_on_exit": True,
        }

        # Start with defaults and override with provided values
        result = defaults.copy()
        result.update(config_data)

        return result


class SessionManagementService:
    """Service for managing profiling sessions."""

    # Class-level storage for sessions
    _session_storage: Dict[str, ProfileSession] = {}

    @classmethod
    def create_session(cls, config: ProfileConfig) -> ProfileSession:
        """Create and store a new profiling session."""
        session = ProfileSession.create(config)
        cls._session_storage[session.session_id] = session
        return session

    @classmethod
    def get_session(cls, session_id: str) -> Optional[ProfileSession]:
        """Retrieve a session by ID."""
        return cls._session_storage.get(session_id)

    @classmethod
    def clear_all_sessions(cls) -> None:
        """Clear all stored sessions (useful for testing)."""
        cls._session_storage.clear()


class ProfilingService:
    """
    High-level service for managing profiling sessions.

    Provides simple API for starting/stopping profiling sessions
    with comprehensive analysis and reporting.
    """

    def __init__(self):
        self.session_service = SessionManagementService()
        self.config_service = ConfigurationService()
        self._active_orchestrators: Dict[str, ProfilerOrchestra] = {}

    def start_profiling(self, **config_kwargs) -> ProfileSession:
        """Start a new profiling session."""
        config = self.config_service.create_config(**config_kwargs)
        session = self.session_service.create_session(config)

        orchestrator = ProfilerOrchestra(session)
        orchestrator.start_profiling()

        # Store orchestrator for this session
        self._active_orchestrators[session.session_id] = orchestrator

        return session

    def stop_profiling(self, session: ProfileSession) -> ProfileSession:
        """Stop profiling and get results."""
        orchestrator = self._active_orchestrators.get(session.session_id)
        if not orchestrator:
            raise ProfilerError(
                f"No active orchestrator found for session {session.session_id}"
            )

        orchestrator.stop_profiling()

        # Clean up orchestrator reference
        self._active_orchestrators.pop(session.session_id, None)

        # Save session if configured
        if session.config.save_raw_data:
            session.save()

        return session

    @contextmanager
    def profile_context(self, **config_kwargs):
        """Context manager for profiling code blocks."""
        session = self.start_profiling(**config_kwargs)
        try:
            yield session
        finally:
            self.stop_profiling(session)

    def profile_function(self, func, **config_kwargs):
        """Decorator/wrapper for profiling functions."""

        def wrapper(*args, **kwargs):
            with self.profile_context(**config_kwargs) as session:
                result = func(*args, **kwargs)
                return result

        return wrapper
