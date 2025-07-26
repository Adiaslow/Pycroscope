"""
Unit tests for ProfileConfig core component.

Tests configuration validation, defaults, and transformations
following pytest best practices and our principles.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pycroscope.core.config import ProfileConfig
from pycroscope.core.exceptions import ConfigurationError
from tests.builders.test_data_builder import config


class TestProfileConfigCreation:
    """Test ProfileConfig instance creation and validation."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_minimal_config_creation(self, temp_dir):
        """Test creating minimal valid configuration."""
        # Arrange & Act
        profile_config = ProfileConfig(output_dir=temp_dir)

        # Assert
        assert profile_config.output_dir == temp_dir
        assert profile_config.line_profiling is True  # Default
        assert profile_config.memory_profiling is True  # Default
        assert profile_config.call_profiling is True  # Default
        assert profile_config.sampling_profiling is False  # Default

    @pytest.mark.unit
    @pytest.mark.core
    def test_explicit_config_creation(self, temp_dir):
        """Test creating configuration with explicit values."""
        # Arrange & Act
        profile_config = ProfileConfig(
            line_profiling=False,
            memory_profiling=True,
            call_profiling=False,
            sampling_profiling=True,
            output_dir=temp_dir,
            session_name="test_session",
        )

        # Assert
        assert profile_config.line_profiling is False
        assert profile_config.memory_profiling is True
        assert profile_config.call_profiling is False
        assert profile_config.sampling_profiling is True
        assert profile_config.session_name == "test_session"

    @pytest.mark.unit
    @pytest.mark.core
    def test_config_with_advanced_settings(self, temp_dir):
        """Test configuration with advanced profiler settings."""
        # Arrange & Act
        profile_config = ProfileConfig(
            output_dir=temp_dir,
            memory_precision=6,
            sampling_interval=0.001,
            max_call_depth=200,
            use_thread_isolation=False,
        )

        # Assert
        assert profile_config.memory_precision == 6
        assert profile_config.sampling_interval == 0.001
        assert profile_config.max_call_depth == 200
        assert profile_config.use_thread_isolation is False


class TestProfileConfigValidation:
    """Test ProfileConfig validation and error handling."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_invalid_memory_precision(self, temp_dir):
        """Test validation of memory precision bounds."""
        # Arrange, Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ProfileConfig(output_dir=temp_dir, memory_precision=10)

        assert "memory_precision" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.core
    def test_invalid_sampling_interval(self, temp_dir):
        """Test validation of sampling interval bounds."""
        # Arrange, Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ProfileConfig(output_dir=temp_dir, sampling_interval=2.0)

        assert "sampling_interval" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.core
    def test_invalid_call_depth(self, temp_dir):
        """Test validation of call depth bounds."""
        # Arrange, Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ProfileConfig(output_dir=temp_dir, max_call_depth=0)

        assert "max_call_depth" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.core
    def test_session_name_too_long(self, temp_dir):
        """Test validation of session name length."""
        # Arrange
        long_name = "x" * 101  # Exceeds limit

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ProfileConfig(output_dir=temp_dir, session_name=long_name)

        assert "session_name" in str(exc_info.value)

    @pytest.mark.unit
    @pytest.mark.core
    def test_profiler_prefix_too_long(self, temp_dir):
        """Test validation of profiler prefix length."""
        # Arrange
        long_prefix = "x" * 21  # Exceeds limit

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            ProfileConfig(output_dir=temp_dir, profiler_prefix=long_prefix)

        assert "profiler_prefix" in str(exc_info.value)


class TestProfileConfigMethods:
    """Test ProfileConfig instance methods and properties."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_enabled_profilers_property(self, temp_dir):
        """Test enabled_profilers property returns correct list."""
        # Arrange
        profile_config = ProfileConfig(
            line_profiling=True,
            memory_profiling=False,
            call_profiling=True,
            sampling_profiling=False,
            output_dir=temp_dir,
        )

        # Act
        enabled = profile_config.enabled_profilers

        # Assert
        assert "line" in enabled
        assert "call" in enabled
        assert "memory" not in enabled
        assert "sampling" not in enabled
        assert len(enabled) == 2

    @pytest.mark.unit
    @pytest.mark.core
    def test_with_minimal_overhead(self, temp_dir):
        """Test with_minimal_overhead method creates optimized config."""
        # Arrange
        original_config = ProfileConfig(
            line_profiling=True,
            memory_profiling=True,
            call_profiling=True,
            sampling_profiling=True,
            output_dir=temp_dir,
            generate_reports=True,
            create_visualizations=True,
        )

        # Act
        minimal_config = original_config.with_minimal_overhead()

        # Assert
        assert minimal_config.line_profiling is False
        assert minimal_config.memory_profiling is False
        assert minimal_config.call_profiling is True  # Only essential profiling
        assert minimal_config.sampling_profiling is False
        assert minimal_config.generate_reports is False
        assert minimal_config.create_visualizations is False
        assert minimal_config.output_dir == temp_dir  # Preserved

    @pytest.mark.unit
    @pytest.mark.core
    def test_with_thread_isolation(self, temp_dir):
        """Test with_thread_isolation method creates isolated config."""
        # Arrange
        original_config = ProfileConfig(output_dir=temp_dir, profiler_prefix="test")

        # Act
        isolated_config = original_config.with_thread_isolation("isolated")

        # Assert
        assert isolated_config.use_thread_isolation is True
        assert isolated_config.profiler_prefix == "isolated"
        assert isolated_config.output_dir == temp_dir  # Preserved

    @pytest.mark.unit
    @pytest.mark.core
    def test_with_thread_isolation_requires_prefix(self, temp_dir):
        """Test with_thread_isolation fails without prefix."""
        # Arrange
        original_config = ProfileConfig(output_dir=temp_dir)

        # Act & Assert
        with pytest.raises(ConfigurationError) as exc_info:
            original_config.with_thread_isolation(None)

        assert "prefix" in str(exc_info.value)


class TestProfileConfigBuilderPattern:
    """Test ProfileConfig using builder pattern from test utilities."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_builder_minimal_config(self, temp_dir):
        """Test creating minimal config using builder."""
        # Arrange & Act
        profile_config = config().minimal().with_output_dir(temp_dir).build()

        # Assert
        assert profile_config.call_profiling is True
        assert profile_config.line_profiling is False
        assert profile_config.memory_profiling is False
        assert profile_config.output_dir == temp_dir

    @pytest.mark.unit
    @pytest.mark.core
    def test_builder_comprehensive_config(self, temp_dir):
        """Test creating comprehensive config using builder."""
        # Arrange & Act
        profile_config = (
            config()
            .comprehensive()
            .with_output_dir(temp_dir)
            .with_session_name("comprehensive_test")
            .build()
        )

        # Assert
        assert profile_config.line_profiling is True
        assert profile_config.memory_profiling is True
        assert profile_config.call_profiling is True
        assert profile_config.generate_reports is True
        assert profile_config.session_name == "comprehensive_test"

    @pytest.mark.unit
    @pytest.mark.core
    def test_builder_custom_config(self, temp_dir):
        """Test creating custom config using builder."""
        # Arrange & Act
        profile_config = (
            config()
            .with_memory_profiling(True)
            .with_call_profiling(False)
            .with_memory_precision(4)
            .with_sampling_interval(0.005)
            .with_output_dir(temp_dir)
            .build()
        )

        # Assert
        assert profile_config.memory_profiling is True
        assert profile_config.call_profiling is False
        assert profile_config.memory_precision == 4
        assert profile_config.sampling_interval == 0.005


class TestProfileConfigSerialization:
    """Test ProfileConfig serialization and model operations."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_model_dump(self, temp_dir):
        """Test ProfileConfig model_dump method."""
        # Arrange
        profile_config = ProfileConfig(
            output_dir=temp_dir, session_name="test_session", memory_precision=3
        )

        # Act
        data = profile_config.model_dump()

        # Assert
        assert isinstance(data, dict)
        assert data["session_name"] == "test_session"
        assert data["memory_precision"] == 3
        assert "output_dir" in data

    @pytest.mark.unit
    @pytest.mark.core
    def test_model_dump_excludes_none(self, temp_dir):
        """Test model_dump excludes None values when requested."""
        # Arrange
        profile_config = ProfileConfig(output_dir=temp_dir)

        # Act
        data = profile_config.model_dump(exclude_none=True)

        # Assert
        none_values = [k for k, v in data.items() if v is None]
        assert len(none_values) == 0


class TestProfileConfigEdgeCases:
    """Test ProfileConfig edge cases and boundary conditions."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_all_profilers_disabled(self, temp_dir):
        """Test configuration with all profilers disabled."""
        # Arrange & Act
        profile_config = ProfileConfig(
            line_profiling=False,
            memory_profiling=False,
            call_profiling=False,
            sampling_profiling=False,
            output_dir=temp_dir,
        )

        # Assert
        assert len(profile_config.enabled_profilers) == 0

    @pytest.mark.unit
    @pytest.mark.core
    def test_minimum_valid_values(self, temp_dir):
        """Test configuration with minimum valid values."""
        # Arrange & Act
        profile_config = ProfileConfig(
            output_dir=temp_dir,
            memory_precision=1,
            sampling_interval=0.001,
            max_call_depth=1,
        )

        # Assert
        assert profile_config.memory_precision == 1
        assert profile_config.sampling_interval == 0.001
        assert profile_config.max_call_depth == 1

    @pytest.mark.unit
    @pytest.mark.core
    def test_maximum_valid_values(self, temp_dir):
        """Test configuration with maximum valid values."""
        # Arrange & Act
        profile_config = ProfileConfig(
            output_dir=temp_dir,
            memory_precision=6,
            sampling_interval=1.0,
            max_call_depth=1000,
        )

        # Assert
        assert profile_config.memory_precision == 6
        assert profile_config.sampling_interval == 1.0
        assert profile_config.max_call_depth == 1000
