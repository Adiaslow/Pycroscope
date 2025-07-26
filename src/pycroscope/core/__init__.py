"""
Core components for Pycroscope profiling framework.

Contains the fundamental abstractions and configuration for
orchestrating multiple profiling tools following SOLID principles.
"""

from .config import ProfileConfig
from .session import ProfileSession
from .interfaces import (
    Profiler,
    ProfilerFactory,
    ConfigurationProvider,
    SessionManager,
    ProfilerOrchestrator,
    ConflictDetector,
    ErrorHandler,
)
from .exceptions import (
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
    PycroscopePermissionError,
)
from .constants import (
    VERSION,
    PACKAGE_NAME,
    ProfilerType,
    SessionStatus,
    ConfigKeys,
    Defaults,
    Limits,
    FilePatterns,
    MetadataKeys,
    ErrorCodes,
    EnvVars,
)
from .container import (
    DependencyContainer,
    get_container,
    register_singleton,
    register_transient,
    resolve,
)
from .factories import ProfilerFactoryRegistry, get_profiler_factory, create_profiler
from .strategies import ProfilingStrategy, ProfilingStrategySelector

# Import bootstrap to initialize the system
from . import bootstrap

__all__ = [
    # Configuration and session
    "ProfileConfig",
    "ProfileSession",
    # Interfaces
    "Profiler",
    "ProfilerFactory",
    "ConfigurationProvider",
    "SessionManager",
    "ProfilerOrchestrator",
    "ConflictDetector",
    "ErrorHandler",
    # Exceptions
    "PycroscopeError",
    "ConfigurationError",
    "ProfilerError",
    "ProfilerConflictError",
    "ProfilerNotAvailableError",
    "ProfilerInitializationError",
    "SessionError",
    "SessionNotFoundError",
    "SessionAlreadyActiveError",
    "SessionStateError",
    "DependencyInjectionError",
    "FactoryError",
    "ValidationError",
    "ResourceError",
    "PycroscopePermissionError",
    # Constants
    "VERSION",
    "PACKAGE_NAME",
    "ProfilerType",
    "SessionStatus",
    "ConfigKeys",
    "Defaults",
    "Limits",
    "FilePatterns",
    "MetadataKeys",
    "ErrorCodes",
    "EnvVars",
    # Dependency injection
    "DependencyContainer",
    "get_container",
    "register_singleton",
    "register_transient",
    "resolve",
    # Factories
    "ProfilerFactoryRegistry",
    "get_profiler_factory",
    "create_profiler",
    # Strategies
    "ProfilingStrategy",
    "ProfilingStrategySelector",
]
