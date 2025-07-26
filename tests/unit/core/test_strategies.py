"""
Tests for profiling strategies.

Tests the Strategy pattern implementation for profiling approaches
including strategy selection, configuration preparation, and validation.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from pycroscope.core.strategies import (
    ProfilingStrategy,
    FullProfilingStrategy,
    MinimalOverheadStrategy,
    MemoryFocusedStrategy,
    PerformanceFocusedStrategy,
    PerformanceMemoryStrategy,
    CallOnlyStrategy,
    ProfilingStrategySelector,
)
from pycroscope.core.constants import ProfilerType
from pycroscope.core.exceptions import ConfigurationError


class TestFullProfilingStrategy:
    """Test comprehensive profiling strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = FullProfilingStrategy()

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_enable_with_all_profilers(self):
        """Test strategy enables when all profilers are requested."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": True,
        }

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is True

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_not_enable_with_partial_profilers(self):
        """Test strategy doesn't enable with partial profilers."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": False,
            "call_profiling": True,
        }

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is False

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_profiler_types_returns_all(self):
        """Test strategy returns all profiler types."""
        # Act
        types = self.strategy.get_profiler_types()

        # Assert
        assert ProfilerType.CALL.value in types
        assert ProfilerType.LINE.value in types
        assert ProfilerType.MEMORY.value in types
        assert len(types) == 3

    @pytest.mark.unit
    @pytest.mark.core
    def test_prepare_config_enables_all_profilers(self):
        """Test config preparation enables all profilers."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": False,
            "output_dir": "/tmp",
        }

        # Act
        result = self.strategy.prepare_config(config)

        # Assert
        assert result["line_profiling"] is True
        assert result["memory_profiling"] is True
        assert result["call_profiling"] is True
        assert result["output_dir"] == "/tmp"  # Preserves other config

    @pytest.mark.unit
    @pytest.mark.core
    def test_validate_environment_always_true(self):
        """Test environment validation always succeeds."""
        # Act
        result = self.strategy.validate_environment()

        # Assert
        assert result is True


class TestMinimalOverheadStrategy:
    """Test minimal overhead profiling strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = MinimalOverheadStrategy()

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_enable_with_minimal_overhead_flag(self):
        """Test strategy enables when minimal overhead is requested."""
        # Arrange
        config = {"minimal_overhead": True}

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is True

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_not_enable_without_flag(self):
        """Test strategy doesn't enable without minimal overhead flag."""
        # Arrange
        config = {"minimal_overhead": False}

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is False

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_profiler_types_returns_only_call(self):
        """Test strategy returns only call profiler."""
        # Act
        types = self.strategy.get_profiler_types()

        # Assert
        assert types == [ProfilerType.CALL.value]

    @pytest.mark.unit
    @pytest.mark.core
    def test_prepare_config_optimizes_for_performance(self):
        """Test config preparation optimizes for minimal overhead."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": False,
            "memory_precision": 6,
            "max_call_depth": 100,
            "create_visualizations": True,
            "analyze_patterns": True,
        }

        # Act
        result = self.strategy.prepare_config(config)

        # Assert
        assert result["line_profiling"] is False
        assert result["memory_profiling"] is False
        assert result["call_profiling"] is True
        assert result["memory_precision"] == 1
        assert result["max_call_depth"] == 20
        assert result["create_visualizations"] is False
        assert result["analyze_patterns"] is False

    @pytest.mark.unit
    @pytest.mark.core
    def test_validate_environment_always_true(self):
        """Test environment validation always succeeds."""
        # Act
        result = self.strategy.validate_environment()

        # Assert
        assert result is True


class TestMemoryFocusedStrategy:
    """Test memory-focused profiling strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = MemoryFocusedStrategy()

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_enable_with_memory_only(self):
        """Test strategy enables when only memory profiling is requested."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": True,
            "call_profiling": False,
        }

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is True

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_not_enable_with_other_profilers(self):
        """Test strategy doesn't enable with other profilers."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": False,
        }

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is False

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_profiler_types_returns_only_memory(self):
        """Test strategy returns only memory profiler."""
        # Act
        types = self.strategy.get_profiler_types()

        # Assert
        assert types == [ProfilerType.MEMORY.value]

    @pytest.mark.unit
    @pytest.mark.core
    def test_prepare_config_enables_only_memory(self):
        """Test config preparation enables only memory profiling."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": False,
            "call_profiling": True,
        }

        # Act
        result = self.strategy.prepare_config(config)

        # Assert
        assert result["line_profiling"] is False
        assert result["memory_profiling"] is True
        assert result["call_profiling"] is False


class TestCallOnlyStrategy:
    """Test call-only profiling strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = CallOnlyStrategy()

    @pytest.mark.unit
    @pytest.mark.core
    def test_should_enable_with_call_only(self):
        """Test strategy enables when only call profiling is requested."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": True,
        }

        # Act
        result = self.strategy.should_enable(config)

        # Assert
        assert result is True

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_profiler_types_returns_only_call(self):
        """Test strategy returns only call profiler."""
        # Act
        types = self.strategy.get_profiler_types()

        # Assert
        assert types == [ProfilerType.CALL.value]

    @pytest.mark.unit
    @pytest.mark.core
    def test_prepare_config_enables_only_call(self):
        """Test config preparation enables only call profiling."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": False,
        }

        # Act
        result = self.strategy.prepare_config(config)

        # Assert
        assert result["line_profiling"] is False
        assert result["memory_profiling"] is False
        assert result["call_profiling"] is True


class TestProfilingStrategySelector:
    """Test profiling strategy selection logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selector = ProfilingStrategySelector()

    @pytest.mark.unit
    @pytest.mark.core
    def test_select_strategy_with_full_profiling(self):
        """Test strategy selection with full profiling configuration."""
        # Arrange
        config = {
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": True,
        }

        # Act
        result = self.selector.select_strategy(config)

        # Assert
        assert isinstance(result, FullProfilingStrategy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_select_strategy_minimal_overhead(self):
        """Test selection of minimal overhead strategy."""
        # Arrange
        config = {"minimal_overhead": True}

        # Act
        result = self.selector.select_strategy(config)

        # Assert
        assert isinstance(result, MinimalOverheadStrategy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_select_strategy_memory_focused(self):
        """Test selection of memory-focused strategy."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": True,
            "call_profiling": False,
        }

        # Act
        result = self.selector.select_strategy(config)

        # Assert
        assert isinstance(result, MemoryFocusedStrategy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_select_strategy_call_only(self):
        """Test selection of call-only strategy."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": True,
        }

        # Act
        result = self.selector.select_strategy(config)

        # Assert
        assert isinstance(result, CallOnlyStrategy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_select_strategy_performance_memory(self):
        """Test selection of performance+memory strategy."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": True,
            "call_profiling": True,
        }

        # Act
        result = self.selector.select_strategy(config)

        # Assert
        assert isinstance(result, PerformanceMemoryStrategy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_select_strategy_no_match_raises_error(self):
        """Test strategy selection raises error when no strategies match."""
        # Arrange
        config = {
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": False,
        }

        # Act & Assert
        with pytest.raises(ConfigurationError, match="No suitable profiling strategy"):
            self.selector.select_strategy(config)

    @pytest.mark.unit
    @pytest.mark.core
    def test_get_available_strategies(self):
        """Test getting list of available strategies."""
        # Act
        strategies = self.selector.get_available_strategies()

        # Assert
        assert (
            len(strategies) == 6
        )  # CallOnly, PerformanceMemory, MinimalOverhead, MemoryFocused, PerformanceFocused, Full
        assert "CallOnlyStrategy" in strategies
        assert "FullProfilingStrategy" in strategies
        assert "MinimalOverheadStrategy" in strategies


class TestStrategyIntegration:
    """Test strategy integration scenarios."""

    @pytest.mark.unit
    @pytest.mark.core
    def test_strategy_priority_order(self):
        """Test that strategies are evaluated in priority order."""
        # Arrange
        selector = ProfilingStrategySelector()

        # Config that matches both minimal overhead and full profiling
        config = {
            "minimal_overhead": True,
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": True,
        }

        # Act
        strategy = selector.select_strategy(config)

        # Assert - Should select MinimalOverheadStrategy due to higher priority
        assert isinstance(strategy, MinimalOverheadStrategy)

    @pytest.mark.unit
    @pytest.mark.core
    def test_strategy_config_transformation(self):
        """Test complete strategy configuration transformation."""
        # Arrange
        selector = ProfilingStrategySelector()
        config = {
            "minimal_overhead": True,
            "line_profiling": True,
            "memory_profiling": True,
            "max_call_depth": 100,
        }

        # Act
        strategy = selector.select_strategy(config)
        transformed_config = strategy.prepare_config(config)

        # Assert - MinimalOverheadStrategy should be selected and transform config
        assert isinstance(strategy, MinimalOverheadStrategy)
        assert transformed_config["line_profiling"] is False
        assert transformed_config["memory_profiling"] is False
        assert transformed_config["call_profiling"] is True
        assert transformed_config["max_call_depth"] == 20
