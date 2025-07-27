"""
Strategy patterns for profiling approaches.

Implements the Strategy pattern to provide different profiling strategies
while maintaining a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .interfaces import Profiler, ProfilerOrchestrator
from .exceptions import ProfilerError, ConfigurationError
from .constants import ProfilerType


class ProfilingStrategy(ABC):
    """Base strategy for profiling approaches."""

    @abstractmethod
    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Check if this strategy should be enabled for given configuration."""
        pass

    @abstractmethod
    def get_profiler_types(self) -> List[str]:
        """Get list of profiler types this strategy uses."""
        pass

    @abstractmethod
    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for this strategy."""
        pass

    @abstractmethod
    def validate_environment(self) -> bool:
        """Validate that environment supports this strategy."""
        pass


class FullProfilingStrategy(ProfilingStrategy):
    """Strategy for comprehensive profiling with all profilers."""

    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Enable if comprehensive profiling is requested."""
        return (
            config.get("line_profiling", False)
            and config.get("memory_profiling", False)
            and config.get("call_profiling", False)
        )

    def get_profiler_types(self) -> List[str]:
        """Get all profiler types."""
        return [
            ProfilerType.CALL.value,
            ProfilerType.LINE.value,
            ProfilerType.MEMORY.value,
        ]

    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for comprehensive profiling."""
        return {
            **config,
            "line_profiling": True,
            "memory_profiling": True,
            "call_profiling": True,
        }

    def validate_environment(self) -> bool:
        """Validate environment for comprehensive profiling."""
        return True


class MinimalOverheadStrategy(ProfilingStrategy):
    """Strategy for minimal overhead profiling."""

    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Enable for minimal overhead configurations."""
        return config.get("minimal_overhead", False)

    def get_profiler_types(self) -> List[str]:
        """Get minimal set of profiler types."""
        return [ProfilerType.CALL.value]

    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for minimal overhead."""
        return {
            **config,
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": True,
            "memory_precision": 1,
            "max_call_depth": 20,
            "create_visualizations": False,
            "analyze_patterns": False,
        }

    def validate_environment(self) -> bool:
        """Validate environment for minimal overhead profiling."""
        return True


class MemoryFocusedStrategy(ProfilingStrategy):
    """Strategy focused on memory profiling."""

    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Enable for memory-focused profiling."""
        return (
            config.get("memory_profiling", False)
            and not config.get("line_profiling", False)
            and not config.get("call_profiling", False)
        )

    def get_profiler_types(self) -> List[str]:
        """Get memory-related profiler types."""
        return [ProfilerType.MEMORY.value]

    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for memory profiling."""
        return {
            **config,
            "line_profiling": False,
            "memory_profiling": True,
            "call_profiling": False,
            "memory_precision": config.get("memory_precision", 3),
        }

    def validate_environment(self) -> bool:
        """Validate environment for memory profiling."""
        return True


class PerformanceFocusedStrategy(ProfilingStrategy):
    """Strategy focused on performance profiling."""

    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Enable for performance-focused profiling."""
        return (
            config.get("line_profiling", False)
            and config.get("call_profiling", False)
            and not config.get("memory_profiling", False)
        )

    def get_profiler_types(self) -> List[str]:
        """Get performance-related profiler types."""
        return [
            ProfilerType.CALL.value,
            ProfilerType.LINE.value,
        ]

    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for performance profiling."""
        return {
            **config,
            "line_profiling": True,
            "memory_profiling": False,
            "call_profiling": True,
        }

    def validate_environment(self) -> bool:
        """Validate environment for performance profiling."""
        return True


class PerformanceMemoryStrategy(ProfilingStrategy):
    """Strategy for combined call and memory profiling."""

    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Enable for call+memory profiling combinations."""
        return (
            config.get("call_profiling", False)
            and config.get("memory_profiling", False)
            and not config.get("line_profiling", False)
        )

    def get_profiler_types(self) -> List[str]:
        """Get call and memory profiler types."""
        return [ProfilerType.CALL.value, ProfilerType.MEMORY.value]

    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for call+memory profiling."""
        return {
            **config,
            "line_profiling": False,
            "max_call_depth": 30,
            "memory_precision": 3,
        }

    def validate_environment(self) -> bool:
        """Validate environment for call+memory profiling."""
        return True


class CallOnlyStrategy(ProfilingStrategy):
    """Strategy for call profiling only."""

    def should_enable(self, config: Dict[str, Any]) -> bool:
        """Enable for call-only profiling configurations."""
        return (
            config.get("call_profiling", False)
            and not config.get("line_profiling", False)
            and not config.get("memory_profiling", False)
        )

    def get_profiler_types(self) -> List[str]:
        """Get call profiler type only."""
        return [ProfilerType.CALL.value]

    def prepare_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for call profiling only."""
        return {
            **config,
            "line_profiling": False,
            "memory_profiling": False,
            "call_profiling": True,
        }

    def validate_environment(self) -> bool:
        """Validate environment for call profiling."""
        return True


class ProfilingStrategySelector:
    """Selects appropriate profiling strategy based on configuration."""

    def __init__(self):
        self._strategies: List[ProfilingStrategy] = [
            CallOnlyStrategy(),
            PerformanceMemoryStrategy(),
            MinimalOverheadStrategy(),
            MemoryFocusedStrategy(),
            PerformanceFocusedStrategy(),
            FullProfilingStrategy(),
        ]

    def select_strategy(self, config: Dict[str, Any]) -> ProfilingStrategy:
        """Select the most appropriate strategy for given configuration."""
        for strategy in self._strategies:
            if strategy.should_enable(config) and strategy.validate_environment():
                return strategy

        # Fail fast - no suitable strategy found
        raise ConfigurationError(
            "No suitable profiling strategy found for configuration",
            config_key="profiling_strategy",
        )

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return [strategy.__class__.__name__ for strategy in self._strategies]
