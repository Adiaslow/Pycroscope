"""
Configuration management for the Pycroscope profiling system.

Provides structured configuration with validation, defaults, and type safety.
Supports both programmatic configuration and file-based configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import json
import yaml
from enum import Enum


class CollectorType(Enum):
    """Available collector types."""

    LINE = "line"
    MEMORY = "memory"
    CALL = "call"
    IO = "io"
    CPU = "cpu"
    GC = "gc"
    IMPORT = "import"
    EXCEPTION = "exception"


class AnalysisType(Enum):
    """Available analysis types."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    CORRELATION = "correlation"
    OPTIMIZATION = "optimization"
    PATTERN = "pattern"
    COMPLEXITY = "complexity"


class StorageType(Enum):
    """Available storage backends."""

    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"


@dataclass
class CollectorConfig:
    """Configuration for individual collectors."""

    enabled: bool = True
    sampling_rate: float = 1.0  # 0.0 to 1.0
    buffer_size: int = 10000
    max_stack_depth: int = 100
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be between 0.0 and 1.0")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.max_stack_depth <= 0:
            raise ValueError("max_stack_depth must be positive")


@dataclass
class AnalysisConfig:
    """Configuration for analysis engines."""

    enabled_analyzers: Set[AnalysisType] = field(
        default_factory=lambda: {
            AnalysisType.STATIC,
            AnalysisType.DYNAMIC,
            AnalysisType.PATTERN,
        }
    )
    complexity_detection: bool = True
    pattern_detection: bool = True
    optimization_suggestions: bool = True
    confidence_threshold: float = 0.7
    impact_threshold: float = 0.1
    custom_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.impact_threshold <= 1.0:
            raise ValueError("impact_threshold must be between 0.0 and 1.0")


@dataclass
class StorageConfig:
    """Configuration for data storage."""

    storage_type: StorageType = StorageType.FILE
    storage_path: Optional[Path] = None
    max_sessions: int = 100
    compression: bool = True
    retention_days: int = 30
    auto_cleanup: bool = True
    connection_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults."""
        if self.storage_path is None:
            self.storage_path = Path.home() / ".pycroscope" / "data"
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if self.retention_days <= 0:
            raise ValueError("retention_days must be positive")


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""

    enabled: bool = True
    auto_open_dashboard: bool = False
    dashboard_port: int = 8080
    export_formats: List[str] = field(default_factory=lambda: ["html", "json"])
    chart_types: List[str] = field(
        default_factory=lambda: ["flame_graph", "timeline", "memory_flow", "call_graph"]
    )
    interactive: bool = True
    theme: str = "default"

    def __post_init__(self):
        """Validate configuration values."""
        if not 1024 <= self.dashboard_port <= 65535:
            raise ValueError("dashboard_port must be between 1024 and 65535")


@dataclass
class ProfileConfig:
    """
    Master configuration for the Pycroscope profiling system.

    Contains all configuration options for collectors, analyzers,
    storage, and visualization components.
    """

    # Target configuration
    target_package: Optional[str] = None
    working_directory: Optional[Path] = None

    # Collector configurations
    collectors: Dict[CollectorType, CollectorConfig] = field(
        default_factory=lambda: {
            collector_type: CollectorConfig() for collector_type in CollectorType
        }
    )

    # Analysis configuration
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Visualization configuration
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Global options
    debug_mode: bool = False
    verbose: bool = False
    parallel_collection: bool = True
    max_threads: int = 4
    timeout_seconds: Optional[int] = None

    def __post_init__(self):
        """Validate global configuration."""
        if self.working_directory is None:
            self.working_directory = Path.cwd()
        if self.max_threads <= 0:
            raise ValueError("max_threads must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def enable_collector(self, collector_type: CollectorType) -> None:
        """Enable a specific collector."""
        if collector_type in self.collectors:
            self.collectors[collector_type].enabled = True
        else:
            self.collectors[collector_type] = CollectorConfig(enabled=True)

    def disable_collector(self, collector_type: CollectorType) -> None:
        """Disable a specific collector."""
        if collector_type in self.collectors:
            self.collectors[collector_type].enabled = False

    def get_enabled_collectors(self) -> List[CollectorType]:
        """Get list of enabled collector types."""
        return [
            collector_type
            for collector_type, config in self.collectors.items()
            if config.enabled
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "target_package": self.target_package,
            "working_directory": (
                str(self.working_directory) if self.working_directory else None
            ),
            "collectors": {
                collector_type.value: {
                    "enabled": config.enabled,
                    "sampling_rate": config.sampling_rate,
                    "buffer_size": config.buffer_size,
                    "max_stack_depth": config.max_stack_depth,
                    "exclude_patterns": config.exclude_patterns,
                    "include_patterns": config.include_patterns,
                    "custom_options": config.custom_options,
                }
                for collector_type, config in self.collectors.items()
            },
            "analysis": {
                "enabled_analyzers": [a.value for a in self.analysis.enabled_analyzers],
                "complexity_detection": self.analysis.complexity_detection,
                "pattern_detection": self.analysis.pattern_detection,
                "optimization_suggestions": self.analysis.optimization_suggestions,
                "confidence_threshold": self.analysis.confidence_threshold,
                "impact_threshold": self.analysis.impact_threshold,
                "custom_patterns": self.analysis.custom_patterns,
            },
            "storage": {
                "storage_type": self.storage.storage_type.value,
                "storage_path": (
                    str(self.storage.storage_path)
                    if self.storage.storage_path
                    else None
                ),
                "max_sessions": self.storage.max_sessions,
                "compression": self.storage.compression,
                "retention_days": self.storage.retention_days,
                "auto_cleanup": self.storage.auto_cleanup,
                "connection_options": self.storage.connection_options,
            },
            "visualization": {
                "enabled": self.visualization.enabled,
                "auto_open_dashboard": self.visualization.auto_open_dashboard,
                "dashboard_port": self.visualization.dashboard_port,
                "export_formats": self.visualization.export_formats,
                "chart_types": self.visualization.chart_types,
                "interactive": self.visualization.interactive,
                "theme": self.visualization.theme,
            },
            "debug_mode": self.debug_mode,
            "verbose": self.verbose,
            "parallel_collection": self.parallel_collection,
            "max_threads": self.max_threads,
            "timeout_seconds": self.timeout_seconds,
        }


class ConfigLoader:
    """Utility class for loading configuration from various sources."""

    @staticmethod
    def from_file(config_path: Union[str, Path]) -> ProfileConfig:
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        return ConfigLoader.from_dict(data)

    @staticmethod
    def from_dict(config_data: Dict[str, Any]) -> ProfileConfig:
        """Create ProfileConfig from dictionary."""
        # Create collector configurations
        collectors = {}
        if "collectors" in config_data:
            for collector_name, collector_data in config_data["collectors"].items():
                try:
                    collector_type = CollectorType(collector_name)
                    collectors[collector_type] = CollectorConfig(**collector_data)
                except ValueError:
                    # Skip unknown collector types
                    continue

        # Create analysis configuration
        analysis_data = config_data.get("analysis", {})
        if "enabled_analyzers" in analysis_data:
            enabled_analyzers = {
                AnalysisType(analyzer)
                for analyzer in analysis_data["enabled_analyzers"]
                if analyzer in [a.value for a in AnalysisType]
            }
            analysis_data["enabled_analyzers"] = enabled_analyzers

        analysis = AnalysisConfig(**analysis_data)

        # Create storage configuration
        storage_data = config_data.get("storage", {})
        if "storage_type" in storage_data:
            storage_data["storage_type"] = StorageType(storage_data["storage_type"])
        if "storage_path" in storage_data and storage_data["storage_path"]:
            storage_data["storage_path"] = Path(storage_data["storage_path"])

        storage = StorageConfig(**storage_data)

        # Create visualization configuration
        visualization_data = config_data.get("visualization", {})
        visualization = VisualizationConfig(**visualization_data)

        # Create working directory path
        working_directory = None
        if "working_directory" in config_data and config_data["working_directory"]:
            working_directory = Path(config_data["working_directory"])

        # Create main configuration
        return ProfileConfig(
            target_package=config_data.get("target_package"),
            working_directory=working_directory,
            collectors=collectors,
            analysis=analysis,
            storage=storage,
            visualization=visualization,
            debug_mode=config_data.get("debug_mode", False),
            verbose=config_data.get("verbose", False),
            parallel_collection=config_data.get("parallel_collection", True),
            max_threads=config_data.get("max_threads", 4),
            timeout_seconds=config_data.get("timeout_seconds"),
        )

    @staticmethod
    def create_default() -> ProfileConfig:
        """Create default configuration."""
        return ProfileConfig()

    @staticmethod
    def save_to_file(config: ProfileConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = config.to_dict()

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == ".json":
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )


# Predefined configuration profiles for common use cases


def create_minimal_config() -> ProfileConfig:
    """Create minimal configuration with only essential collectors."""
    config = ProfileConfig()

    # Enable only line and memory profiling
    for collector_type in CollectorType:
        config.disable_collector(collector_type)

    config.enable_collector(CollectorType.LINE)
    config.enable_collector(CollectorType.MEMORY)

    # Disable visualization for minimal overhead
    config.visualization.enabled = False

    return config


def create_comprehensive_config() -> ProfileConfig:
    """Create comprehensive configuration with all features enabled."""
    config = ProfileConfig()

    # Enable all collectors
    for collector_type in CollectorType:
        config.enable_collector(collector_type)

    # Enable all analysis types
    config.analysis.enabled_analyzers = set(AnalysisType)

    # Enable all visualization features
    config.visualization.enabled = True
    config.visualization.interactive = True

    return config


def create_development_config() -> ProfileConfig:
    """Create configuration optimized for development workflow."""
    config = ProfileConfig()

    # Enable debug mode
    config.debug_mode = True
    config.verbose = True

    # Enable key collectors
    config.enable_collector(CollectorType.LINE)
    config.enable_collector(CollectorType.MEMORY)
    config.enable_collector(CollectorType.CALL)

    # Auto-open dashboard
    config.visualization.auto_open_dashboard = True

    return config
