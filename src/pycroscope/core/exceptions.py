"""
Custom exception hierarchy for Pycroscope.

Provides specific exception types for different error scenarios,
following the Single Responsibility Principle for error handling.
"""

from typing import Optional, List


class PycroscopeError(Exception):
    """Base exception for all Pycroscope errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.cause = cause


class ConfigurationError(PycroscopeError):
    """Raised when configuration is invalid or incompatible."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class ProfilerError(PycroscopeError):
    """Base class for profiler-related errors."""

    def __init__(self, message: str, profiler_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.profiler_type = profiler_type


class ProfilerConflictError(ProfilerError):
    """Raised when profiler conflicts with existing profiling."""

    def __init__(
        self, message: str, conflicting_profilers: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(message, error_code="PROFILER_CONFLICT", **kwargs)
        if conflicting_profilers is None:
            conflicting_profilers = []
        self.conflicting_profilers = conflicting_profilers


class ProfilerNotAvailableError(ProfilerError):
    """Raised when requested profiler is not available."""

    def __init__(
        self, message: str, missing_dependencies: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(message, error_code="PROFILER_UNAVAILABLE", **kwargs)
        if missing_dependencies is None:
            missing_dependencies = []
        self.missing_dependencies = missing_dependencies


class ProfilerInitializationError(ProfilerError):
    """Raised when profiler fails to initialize."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PROFILER_INIT_FAILED", **kwargs)


class SessionError(PycroscopeError):
    """Base class for session-related errors."""

    def __init__(self, message: str, session_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.session_id = session_id


class SessionNotFoundError(SessionError):
    """Raised when session cannot be found."""

    def __init__(self, session_id: str, **kwargs):
        message = f"Session not found: {session_id}"
        super().__init__(
            message, session_id=session_id, error_code="SESSION_NOT_FOUND", **kwargs
        )


class SessionAlreadyActiveError(SessionError):
    """Raised when trying to start session that's already active."""

    def __init__(self, session_id: str, **kwargs):
        message = f"Session already active: {session_id}"
        super().__init__(
            message,
            session_id=session_id,
            error_code="SESSION_ALREADY_ACTIVE",
            **kwargs,
        )


class SessionStateError(SessionError):
    """Raised when session is in invalid state for operation."""

    def __init__(
        self,
        message: str,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="SESSION_INVALID_STATE", **kwargs)
        self.current_state = current_state
        self.expected_state = expected_state


class DependencyInjectionError(PycroscopeError):
    """Raised when dependency injection fails."""

    def __init__(self, message: str, interface_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DI_ERROR", **kwargs)
        self.interface_name = interface_name


class FactoryError(PycroscopeError):
    """Raised when factory fails to create instance."""

    def __init__(self, message: str, factory_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="FACTORY_ERROR", **kwargs)
        self.factory_type = factory_type


class ValidationError(PycroscopeError):
    """Raised when validation fails."""

    def __init__(
        self, message: str, validation_errors: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        if validation_errors is None:
            validation_errors = []
        self.validation_errors = validation_errors

    def __str__(self) -> str:
        """Return string representation including validation errors."""
        if self.validation_errors:
            errors_str = "; ".join(self.validation_errors)
            return f"{self.message}: {errors_str}"
        return self.message


class ResourceError(PycroscopeError):
    """Raised when resource operation fails."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        self.resource_path = resource_path


class PycroscopePermissionError(PycroscopeError):
    """Raised when operation lacks required permissions."""

    def __init__(
        self, message: str, required_permission: Optional[str] = None, **kwargs
    ):
        super().__init__(message, error_code="PERMISSION_ERROR", **kwargs)
        self.required_permission = required_permission
