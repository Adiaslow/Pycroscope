"""
Bootstrap module for Pycroscope.

Sets up dependency injection container and registers components
following the Dependency Inversion Principle.
"""

from .container import get_container
from .factories import get_profiler_factory
from .constants import ProfilerType

# Import all profiler classes at module level to follow design principles
from ..infrastructure.profilers.call_profiler import CallProfiler
from ..infrastructure.profilers.line_profiler import LineProfiler
from ..infrastructure.profilers.memory_profiler import MemoryProfiler

# Import factory classes at module level
from .factories import (
    CallProfilerFactory,
    LineProfilerFactory,
    MemoryProfilerFactory,
)


def initialize() -> None:
    """Initialize dependency injection container and register components."""
    container = get_container()
    factory_registry = get_profiler_factory()

    # Register direct profiler factories with their respective profiler classes
    factory_registry.register_factory(
        ProfilerType.CALL.value,
        CallProfilerFactory(CallProfiler),
    )

    factory_registry.register_factory(
        ProfilerType.LINE.value,
        LineProfilerFactory(LineProfiler),
    )

    factory_registry.register_factory(
        ProfilerType.MEMORY.value,
        MemoryProfilerFactory(MemoryProfiler),
    )


def cleanup() -> None:
    """Clean up dependency injection container."""
    from .container import reset_container

    reset_container()


# Initialize on module import
initialize()
