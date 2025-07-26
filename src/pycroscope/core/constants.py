"""
Constants and enums for Pycroscope.

Single Source of Truth (SSoT) for all configuration constants,
magic numbers, and enumerated values used throughout the system.
"""

from enum import Enum, auto
from typing import Final


# Version and metadata
VERSION: Final[str] = "2.0.0"
PACKAGE_NAME: Final[str] = "pycroscope"
DEFAULT_PREFIX: Final[str] = "pycroscope"


class ProfilerType(Enum):
    """Enumeration of supported profiler types."""

    CALL = "call"
    LINE = "line"
    MEMORY = "memory"
    SAMPLING = "sampling"


class SessionStatus(Enum):
    """Enumeration of session statuses."""

    PENDING = "pending"
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConfigKeys(Enum):
    """Configuration key constants."""

    # Core profiling toggles
    LINE_PROFILING = "line_profiling"
    MEMORY_PROFILING = "memory_profiling"
    CALL_PROFILING = "call_profiling"
    SAMPLING_PROFILING = "sampling_profiling"

    # Output configuration
    OUTPUT_DIR = "output_dir"
    SESSION_NAME = "session_name"
    SAVE_RAW_DATA = "save_raw_data"

    # Performance tuning
    SAMPLING_INTERVAL = "sampling_interval"
    MEMORY_PRECISION = "memory_precision"
    MAX_CALL_DEPTH = "max_call_depth"

    # Analysis configuration
    GENERATE_REPORTS = "generate_reports"
    CREATE_VISUALIZATIONS = "create_visualizations"
    ANALYZE_PATTERNS = "analyze_patterns"


# Default values
class Defaults:
    """Default configuration values."""

    # Profiling defaults
    LINE_PROFILING: Final[bool] = True
    MEMORY_PROFILING: Final[bool] = True
    CALL_PROFILING: Final[bool] = True
    SAMPLING_PROFILING: Final[bool] = False

    # Performance defaults
    SAMPLING_INTERVAL: Final[float] = 0.01
    MEMORY_PRECISION: Final[int] = 3
    MAX_CALL_DEPTH: Final[int] = 50

    # Analysis defaults
    GENERATE_REPORTS: Final[bool] = True
    CREATE_VISUALIZATIONS: Final[bool] = True
    ANALYZE_PATTERNS: Final[bool] = True

    # Threading and isolation
    USE_THREAD_ISOLATION: Final[bool] = True
    CLEANUP_ON_EXIT: Final[bool] = True

    # File handling
    SAVE_RAW_DATA: Final[bool] = True


# Limits and constraints
class Limits:
    """System limits and constraints."""

    MIN_SAMPLING_INTERVAL: Final[float] = 0.001
    MAX_SAMPLING_INTERVAL: Final[float] = 1.0

    MIN_MEMORY_PRECISION: Final[int] = 1
    MAX_MEMORY_PRECISION: Final[int] = 6

    MIN_CALL_DEPTH: Final[int] = 5
    MAX_CALL_DEPTH: Final[int] = 200

    MAX_SESSION_NAME_LENGTH: Final[int] = 255
    MAX_PROFILER_PREFIX_LENGTH: Final[int] = 50


class FilePatterns:
    """File name patterns and extensions."""

    SESSION_FILE_PATTERN: Final[str] = "session_{session_id}.json"
    PROFILER_OUTPUT_PATTERN: Final[str] = "{profiler_type}_{session_id}.{ext}"

    # File extensions
    JSON_EXT: Final[str] = "json"
    CSV_EXT: Final[str] = "csv"
    HTML_EXT: Final[str] = "html"
    PNG_EXT: Final[str] = "png"


class MetadataKeys:
    """Keys for metadata dictionaries."""

    # System metadata
    PYTHON_VERSION = "python_version"
    PLATFORM = "platform"
    HOSTNAME = "hostname"
    PROCESS_ID = "process_id"
    THREAD_ID = "thread_id"

    # Profiler metadata
    PROFILER_VERSION = "profiler_version"
    PROFILER_CONFIG = "profiler_config"
    START_TIME = "start_time"
    END_TIME = "end_time"
    DURATION = "duration"

    # Target metadata
    FUNCTION_NAME = "function_name"
    FUNCTION_MODULE = "function_module"
    FUNCTION_FILE = "function_file"
    SOURCE_LINES = "source_lines"


class ErrorCodes:
    """Error codes for different types of failures."""

    # Configuration errors
    CONFIG_ERROR = "CONFIG_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"

    # Profiler errors
    PROFILER_CONFLICT = "PROFILER_CONFLICT"
    PROFILER_UNAVAILABLE = "PROFILER_UNAVAILABLE"
    PROFILER_INIT_FAILED = "PROFILER_INIT_FAILED"

    # Session errors
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_ALREADY_ACTIVE = "SESSION_ALREADY_ACTIVE"
    SESSION_INVALID_STATE = "SESSION_INVALID_STATE"

    # System errors
    DI_ERROR = "DI_ERROR"
    FACTORY_ERROR = "FACTORY_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    PERMISSION_ERROR = "PERMISSION_ERROR"


# Environment variables
class EnvVars:
    """Environment variable names."""

    PYCROSCOPE_CONFIG_PATH = "PYCROSCOPE_CONFIG_PATH"
    PYCROSCOPE_OUTPUT_DIR = "PYCROSCOPE_OUTPUT_DIR"
    PYCROSCOPE_DEBUG = "PYCROSCOPE_DEBUG"
    PYCROSCOPE_LOG_LEVEL = "PYCROSCOPE_LOG_LEVEL"
