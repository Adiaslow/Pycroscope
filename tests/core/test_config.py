"""
Unit tests for configuration management.

Tests ProfileConfig, CollectorConfig, AnalysisConfig, StorageConfig,
and related configuration classes.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from pycroscope.core.config import (
    AnalysisConfig,
    AnalysisType,
    CollectorConfig,
    CollectorType,
    ConfigLoader,
    ProfileConfig,
    StorageConfig,
    StorageType,
    VisualizationConfig,
)


class TestCollectorConfig:
    """Test CollectorConfig data class."""

    def test_default_creation(self):
        """Test CollectorConfig with default values."""
        config = CollectorConfig()

        assert config.enabled is True
        assert config.sampling_rate == 1.0
        assert config.buffer_size == 10000
        assert config.include_patterns == []
        assert config.exclude_patterns == []

    def test_custom_creation(self):
        """Test CollectorConfig with custom values."""
        config = CollectorConfig(
            enabled=False,
            sampling_rate=0.5,
            buffer_size=5000,
            include_patterns=["*.py"],
            exclude_patterns=["test_*"],
        )

        assert config.enabled is False
        assert config.sampling_rate == 0.5
        assert config.buffer_size == 5000
        assert config.include_patterns == ["*.py"]
        assert config.exclude_patterns == ["test_*"]

    def test_validation(self):
        """Test CollectorConfig validation."""
        # Invalid sampling rate
        with pytest.raises(ValueError):
            CollectorConfig(sampling_rate=-0.1)

        with pytest.raises(ValueError):
            CollectorConfig(sampling_rate=1.1)

        # Invalid buffer size
        with pytest.raises(ValueError):
            CollectorConfig(buffer_size=0)

        with pytest.raises(ValueError):
            CollectorConfig(buffer_size=-100)


class TestAnalysisConfig:
    """Test AnalysisConfig data class."""

    def test_default_creation(self):
        """Test AnalysisConfig with default values."""
        config = AnalysisConfig()

        assert config.enabled_analyzers == {
            AnalysisType.STATIC,
            AnalysisType.DYNAMIC,
            AnalysisType.PATTERN,
        }
        assert config.complexity_detection is True
        assert config.pattern_detection is True
        assert config.optimization_suggestions is True
        assert config.confidence_threshold == 0.7
        assert config.impact_threshold == 0.1  # Correct default value
        assert config.custom_patterns == []

    def test_custom_creation(self):
        """Test AnalysisConfig with custom values."""
        enabled = {AnalysisType.STATIC, AnalysisType.DYNAMIC}

        config = AnalysisConfig(
            enabled_analyzers=enabled, confidence_threshold=0.8, impact_threshold=0.3
        )

        assert config.enabled_analyzers == enabled
        assert config.confidence_threshold == 0.8
        assert config.impact_threshold == 0.3

    def test_validation(self):
        """Test AnalysisConfig validation."""
        # Invalid confidence threshold
        with pytest.raises(ValueError):
            AnalysisConfig(confidence_threshold=-0.1)

        with pytest.raises(ValueError):
            AnalysisConfig(confidence_threshold=1.1)

        # Invalid impact threshold
        with pytest.raises(ValueError):
            AnalysisConfig(impact_threshold=-0.1)

        with pytest.raises(ValueError):
            AnalysisConfig(impact_threshold=1.1)


class TestStorageConfig:
    """Test StorageConfig data class."""

    def test_default_creation(self):
        """Test StorageConfig with default values."""
        config = StorageConfig()

        assert config.storage_type == StorageType.FILE
        # Default path is set in __post_init__ to Path.home() / ".pycroscope" / "data"
        assert config.storage_path == Path.home() / ".pycroscope" / "data"
        assert config.max_sessions == 100
        assert config.compression is True
        assert config.retention_days == 30
        assert config.auto_cleanup is True
        assert config.connection_options == {}

    def test_custom_creation(self):
        """Test StorageConfig with custom values."""
        storage_path = Path("/custom/storage")

        config = StorageConfig(
            storage_type=StorageType.MEMORY,
            storage_path=storage_path,
            compression=False,
            max_sessions=50,
            retention_days=7,
            auto_cleanup=False,
        )

        assert config.storage_type == StorageType.MEMORY
        assert config.storage_path == storage_path
        assert config.compression is False
        assert config.max_sessions == 50
        assert config.retention_days == 7
        assert config.auto_cleanup is False

    def test_validation(self):
        """Test StorageConfig validation."""
        # Invalid max_sessions
        with pytest.raises(ValueError):
            StorageConfig(max_sessions=0)

        with pytest.raises(ValueError):
            StorageConfig(max_sessions=-10)

        # Invalid retention_days
        with pytest.raises(ValueError):
            StorageConfig(retention_days=-1)


class TestVisualizationConfig:
    """Test VisualizationConfig data class."""

    def test_default_creation(self):
        """Test VisualizationConfig with default values."""
        config = VisualizationConfig()

        assert config.enabled is True
        assert config.auto_open_dashboard is False
        assert config.dashboard_port == 8080
        assert config.interactive is True

    def test_custom_creation(self):
        """Test VisualizationConfig with custom values."""
        config = VisualizationConfig(
            enabled=False,
            auto_open_dashboard=True,
            dashboard_port=9000,
            interactive=False,
        )

        assert config.enabled is False
        assert config.auto_open_dashboard is True
        assert config.dashboard_port == 9000
        assert config.interactive is False

    def test_validation(self):
        """Test VisualizationConfig validation."""
        # Invalid port (below valid range)
        with pytest.raises(ValueError):
            VisualizationConfig(dashboard_port=500)

        # Invalid port (above valid range)
        with pytest.raises(ValueError):
            VisualizationConfig(dashboard_port=70000)

        # Invalid negative port
        with pytest.raises(ValueError):
            VisualizationConfig(dashboard_port=-1)


class TestProfileConfig:
    """Test ProfileConfig master configuration class."""

    def test_default_creation(self):
        """Test ProfileConfig with default values."""
        config = ProfileConfig()

        assert config.target_package is None
        assert config.working_directory == Path.cwd()
        assert config.debug_mode is False
        assert config.verbose is False
        assert config.parallel_collection is True
        assert config.max_threads == 4
        assert config.timeout_seconds is None

        # Should have all collector types configured
        for collector_type in CollectorType:
            assert collector_type in config.collectors
            assert isinstance(config.collectors[collector_type], CollectorConfig)

        # Should have analysis config
        assert isinstance(config.analysis, AnalysisConfig)

        # Should have storage config
        assert isinstance(config.storage, StorageConfig)

        # Should have visualization config
        assert isinstance(config.visualization, VisualizationConfig)

    def test_custom_creation(self):
        """Test ProfileConfig with custom values."""
        custom_collectors = {
            CollectorType.LINE: CollectorConfig(enabled=True, sampling_rate=0.8),
            CollectorType.MEMORY: CollectorConfig(enabled=False),
        }

        config = ProfileConfig(
            target_package="mypackage",
            working_directory=Path("/custom/dir"),
            debug_mode=True,
            verbose=True,
            parallel_collection=False,
            max_threads=2,
            timeout_seconds=60,
            collectors=custom_collectors,
        )

        assert config.target_package == "mypackage"
        assert config.working_directory == Path("/custom/dir")
        assert config.debug_mode is True
        assert config.verbose is True
        assert config.parallel_collection is False
        assert config.max_threads == 2
        assert config.timeout_seconds == 60
        assert config.collectors == custom_collectors

    def test_validation(self):
        """Test ProfileConfig validation."""
        # Invalid max_threads
        with pytest.raises(ValueError):
            ProfileConfig(max_threads=0)

        with pytest.raises(ValueError):
            ProfileConfig(max_threads=-2)

        # Invalid timeout_seconds
        with pytest.raises(ValueError):
            ProfileConfig(timeout_seconds=0)

        with pytest.raises(ValueError):
            ProfileConfig(timeout_seconds=-10)

    def test_post_init(self):
        """Test ProfileConfig __post_init__ behavior."""
        # Working directory should default to cwd if None
        config = ProfileConfig(working_directory=None)
        assert config.working_directory == Path.cwd()

    def test_enable_collector(self):
        """Test enable_collector method."""
        config = ProfileConfig()

        # Disable a collector first
        config.collectors[CollectorType.LINE].enabled = False
        assert config.collectors[CollectorType.LINE].enabled is False

        # Enable it using the method
        config.enable_collector(CollectorType.LINE)
        assert config.collectors[CollectorType.LINE].enabled is True

    def test_enable_collector_new(self):
        """Test enable_collector with new collector type."""
        config = ProfileConfig()

        # Remove a collector
        del config.collectors[CollectorType.CPU]
        assert CollectorType.CPU not in config.collectors

        # Enable it - should create new config
        config.enable_collector(CollectorType.CPU)
        assert CollectorType.CPU in config.collectors
        assert config.collectors[CollectorType.CPU].enabled is True

    def test_disable_collector(self):
        """Test disable_collector method."""
        config = ProfileConfig()

        # Ensure collector is enabled first
        config.collectors[CollectorType.MEMORY].enabled = True
        assert config.collectors[CollectorType.MEMORY].enabled is True

        # Disable it
        config.disable_collector(CollectorType.MEMORY)
        assert config.collectors[CollectorType.MEMORY].enabled is False

    def test_disable_collector_missing(self):
        """Test disable_collector with missing collector."""
        config = ProfileConfig()

        # Remove a collector
        del config.collectors[CollectorType.GC]

        # Disabling shouldn't fail, just be a no-op
        config.disable_collector(CollectorType.GC)
        assert CollectorType.GC not in config.collectors


class TestConfigLoader:
    """Test ConfigLoader utility class."""

    @pytest.fixture
    def sample_yaml_config(self, tmp_path):
        """Create a sample YAML config file."""
        config_content = """
target_package: "test_package"
debug_mode: true
verbose: false
max_threads: 2
timeout_seconds: 60

collectors:
  line:
    enabled: true
    sampling_rate: 0.8
  memory:
    enabled: false
    buffer_size: 5000

analysis:
  confidence_threshold: 0.8
  impact_threshold: 0.4

storage:
  storage_type: "file"
  compression: false
  max_sessions: 50
  retention_days: 14

visualization:
  auto_open_dashboard: false
  dashboard_port: 9000
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return config_file

    @pytest.fixture
    def sample_json_config(self, tmp_path):
        """Create a sample JSON config file."""
        config_content = """{
    "target_package": "json_package",
    "debug_mode": false,
    "max_threads": 8,
    "collectors": {
        "call": {
            "enabled": true,
            "sampling_rate": 1.0
        }
    },
    "storage": {
        "storage_type": "memory",
        "max_sessions": 25
    }
}"""
        config_file = tmp_path / "test_config.json"
        config_file.write_text(config_content)
        return config_file

    def test_load_yaml_config(self, sample_yaml_config):
        """Test loading YAML configuration."""
        config = ConfigLoader.from_file(sample_yaml_config)

        assert config.target_package == "test_package"
        assert config.debug_mode is True
        assert config.verbose is False
        assert config.max_threads == 2
        assert config.timeout_seconds == 60

        # Check collector config
        assert config.collectors[CollectorType.LINE].enabled is True
        assert config.collectors[CollectorType.LINE].sampling_rate == 0.8
        assert config.collectors[CollectorType.MEMORY].enabled is False
        assert config.collectors[CollectorType.MEMORY].buffer_size == 5000

        # Check analysis config
        assert config.analysis.confidence_threshold == 0.8
        assert config.analysis.impact_threshold == 0.4

        # Check storage config
        assert config.storage.storage_type == StorageType.FILE
        assert config.storage.compression is False
        assert config.storage.max_sessions == 50
        assert config.storage.retention_days == 14

        # Check visualization config
        assert config.visualization.auto_open_dashboard is False
        assert config.visualization.dashboard_port == 9000

    def test_load_json_config(self, sample_json_config):
        """Test loading JSON configuration."""
        config = ConfigLoader.from_file(sample_json_config)

        assert config.target_package == "json_package"
        assert config.debug_mode is False
        assert config.max_threads == 8

        # Check collector config
        assert config.collectors[CollectorType.CALL].enabled is True
        assert config.collectors[CollectorType.CALL].sampling_rate == 1.0

        # Check storage config
        assert config.storage.storage_type == StorageType.MEMORY
        assert config.storage.max_sessions == 25

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.from_file(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML file."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises((ValueError, yaml.YAMLError)):
            ConfigLoader.from_file(invalid_file)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text('{"invalid": json content}')

        with pytest.raises((ValueError, json.JSONDecodeError)):
            ConfigLoader.from_file(invalid_file)

    def test_save_config_yaml(self, tmp_path):
        """Test saving configuration to YAML."""
        config = ProfileConfig(
            target_package="save_test", debug_mode=True, max_threads=6
        )

        output_file = tmp_path / "saved_config.yaml"
        ConfigLoader.save_to_file(config, output_file)

        assert output_file.exists()

        # Load it back and verify
        loaded_config = ConfigLoader.from_file(output_file)
        assert loaded_config.target_package == "save_test"
        assert loaded_config.debug_mode is True
        assert loaded_config.max_threads == 6

    def test_save_config_json(self, tmp_path):
        """Test saving configuration to JSON."""
        config = ProfileConfig(target_package="json_save_test", verbose=True)

        output_file = tmp_path / "saved_config.json"
        ConfigLoader.save_to_file(config, output_file)

        assert output_file.exists()

        # Load it back and verify
        loaded_config = ConfigLoader.from_file(output_file)
        assert loaded_config.target_package == "json_save_test"
        assert loaded_config.verbose is True

    def test_create_from_dict(self):
        """Test creating config from dictionary."""
        config_data = {
            "target_package": "test_package",
            "max_threads": 6,
            "collectors": {"line": {"enabled": True, "sampling_rate": 0.5}},
        }

        config = ConfigLoader.from_dict(config_data)
        assert config.target_package == "test_package"
        assert config.max_threads == 6
        assert config.collectors[CollectorType.LINE].sampling_rate == 0.5
