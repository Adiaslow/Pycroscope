"""
Main profiling orchestrator for the Pycroscope system.

The ProfilerSuite serves as the central coordinator for all profiling activities,
managing collector lifecycle, session coordination, and providing the primary API.
"""

import atexit
import signal
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set, Type

from .config import CollectorType, ProfileConfig
from .interfaces import Analyzer, Collector, DataStore, Lifecycle, Visualizer
from .models import (
    AnalysisResult,
    CallNode,
    CallTree,
    Dashboard,
    EnvironmentInfo,
    EventType,
    ExecutionContext,
    ExecutionEvent,
    FrameInfo,
    MemorySnapshot,
    ProfileSession,
    SourceLocation,
    create_session_id,
)
from .registry import ComponentRegistry


class ProfilerSuite(Lifecycle):
    """
    Central orchestrator for the Pycroscope profiling system.

    Manages the complete profiling lifecycle from data collection through
    analysis and visualization, following the "One Way, Many Options" principle.
    """

    def __init__(self, config: Optional[ProfileConfig] = None):
        """
        Initialize the profiling suite.

        Args:
            config: Optional configuration. Uses default if not provided.
        """
        self._config = config or ProfileConfig()
        self._registry = ComponentRegistry()
        self._collectors: Dict[CollectorType, Collector] = {}
        self._analyzers: List[Analyzer] = []
        self._data_store: Optional[DataStore] = None
        self._visualizer: Optional[Visualizer] = None

        # Session management
        self._current_session: Optional[ProfileSession] = None
        self._session_lock = threading.RLock()
        self._is_running = False
        self._collection_thread: Optional[threading.Thread] = None

        # Event collection
        self._events_buffer: List[ExecutionEvent] = []
        self._buffer_lock = threading.Lock()

        # Lifecycle management
        self._cleanup_registered = False
        self._weak_ref = weakref.ref(self)

        # Register cleanup handlers for robust shutdown
        self._register_cleanup_handlers()

        # Initialize components
        self._initialize_components()

    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for robust shutdown."""
        if not self._cleanup_registered:
            # Register atexit handler
            atexit.register(self._cleanup_handler)

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            self._cleanup_registered = True

    def _cleanup_handler(self) -> None:
        """Cleanup handler for atexit."""
        try:
            if self._is_running:
                self.disable()
        except Exception:
            # Don't let cleanup errors propagate
            pass

    def _signal_handler(self, signum, frame) -> None:
        """Signal handler for graceful shutdown."""
        try:
            if self._is_running:
                self.disable()
        except Exception:
            pass
        finally:
            # Re-raise the signal to allow normal termination
            if signum == signal.SIGINT:
                sys.exit(130)  # Standard exit code for SIGINT
            elif signum == signal.SIGTERM:
                sys.exit(143)  # Standard exit code for SIGTERM

    def _initialize_components(self) -> None:
        """Initialize all system components based on configuration."""
        # Register cleanup handler
        if not self._cleanup_registered:
            atexit.register(self._cleanup_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            self._cleanup_registered = True

        # Initialize component registry with default implementations
        self._register_default_components()

        # Create collectors based on configuration
        self._create_collectors()

        # Create analyzers
        self._create_analyzers()

        # Create data store
        self._create_data_store()

        # Create visualizer
        self._create_visualizer()

    def _register_default_components(self) -> None:
        """Register default component implementations."""
        from ..analysis.complexity_detector import AlgorithmComplexityDetector
        from ..analysis.correlation_analyzer import CrossCorrelationAnalyzer
        from ..analysis.dynamic_analyzer import DynamicAnalyzer
        from ..analysis.optimization_engine import OptimizationRecommendationEngine
        from ..analysis.pattern_detector import AdvancedPatternDetector
        from ..analysis.static_analyzer import StaticAnalyzer
        from ..collectors import (
            CallCollector,
            CPUCollector,
            ExceptionCollector,
            GCCollector,
            ImportCollector,
            IOCollector,
            LineCollector,
            MemoryCollector,
        )
        from ..storage.file_store import FileDataStore
        from ..storage.memory_store import MemoryDataStore
        from .config import AnalysisType, CollectorType, StorageType

        # Register collectors
        self._registry.register_collector(CollectorType.LINE, LineCollector)
        self._registry.register_collector(CollectorType.MEMORY, MemoryCollector)
        self._registry.register_collector(CollectorType.CALL, CallCollector)
        self._registry.register_collector(CollectorType.EXCEPTION, ExceptionCollector)
        self._registry.register_collector(CollectorType.IMPORT, ImportCollector)
        self._registry.register_collector(CollectorType.GC, GCCollector)
        self._registry.register_collector(CollectorType.IO, IOCollector)
        self._registry.register_collector(CollectorType.CPU, CPUCollector)

        # Register analyzers
        self._registry.register_analyzer(AnalysisType.STATIC, StaticAnalyzer)
        self._registry.register_analyzer(AnalysisType.DYNAMIC, DynamicAnalyzer)
        self._registry.register_analyzer(AnalysisType.PATTERN, AdvancedPatternDetector)
        self._registry.register_analyzer(
            AnalysisType.CORRELATION, CrossCorrelationAnalyzer
        )
        self._registry.register_analyzer(
            AnalysisType.COMPLEXITY, AlgorithmComplexityDetector
        )
        self._registry.register_analyzer(
            AnalysisType.OPTIMIZATION, OptimizationRecommendationEngine
        )

        # Register storage backends
        self._registry.register_data_store(StorageType.FILE, FileDataStore)
        self._registry.register_data_store(StorageType.MEMORY, MemoryDataStore)

    def _create_collectors(self) -> None:
        """Create collector instances based on configuration."""
        for collector_type in self._config.get_enabled_collectors():
            collector_class = self._registry.get_collector_class(collector_type)
            if collector_class:
                collector_config = self._config.collectors[collector_type]
                collector = collector_class()
                self._collectors[collector_type] = collector

    def _create_analyzers(self) -> None:
        """Create analyzer instances based on configuration."""
        for analyzer_type in self._config.analysis.enabled_analyzers:
            analyzer_class = self._registry.get_analyzer_class(analyzer_type)
            if analyzer_class:
                analyzer = analyzer_class()  # Simplified for now
                self._analyzers.append(analyzer)

    def _create_data_store(self) -> None:
        """Create data store instance based on configuration."""
        store_class = self._registry.get_data_store_class(
            self._config.storage.storage_type
        )
        if store_class:
            self._data_store = store_class()  # Simplified for now

    def _create_visualizer(self) -> None:
        """Create visualizer instance based on configuration."""
        if self._config.visualization.enabled:
            visualizer_class = self._registry.get_visualizer_class()
            if visualizer_class:
                self._visualizer = visualizer_class()  # Simplified for now

    @property
    def is_running(self) -> bool:
        """Whether profiling is currently active."""
        return self._is_running

    @property
    def current_session(self) -> Optional[ProfileSession]:
        """Current profiling session, if any."""
        return self._current_session

    @property
    def configuration(self) -> ProfileConfig:
        """Current configuration."""
        return self._config

    def start(self) -> None:
        """Start the profiling suite without beginning a session."""
        if self._is_running:
            return

        self._is_running = True

        if self._config.verbose:
            print("Pycroscope: Profiling suite started")

    def stop(self) -> None:
        """Stop the profiling suite and any active session."""
        if not self._is_running:
            return

        # Stop any active session
        if self._current_session:
            self.end_session()

        # Stop collection thread
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)

        self._is_running = False

        if self._config.verbose:
            print("Pycroscope: Profiling suite stopped")

    def enable(self) -> "ProfilerSuite":
        """
        Enable profiling with automatic session management.

        This is the main convenience method for simple usage.
        Creates a session and starts collection automatically.

        Returns:
            Self for method chaining
        """
        self.start()
        self.begin_session()
        return self

    def disable(self) -> None:
        """Disable profiling and clean up resources."""
        self.stop()

    def begin_session(self, target_package: Optional[str] = None) -> str:
        """
        Begin a new profiling session.

        Args:
            target_package: Optional package name to profile

        Returns:
            Session identifier

        Raises:
            RuntimeError: If a session is already active
        """
        with self._session_lock:
            if self._current_session is not None:
                raise RuntimeError("A profiling session is already active")

            if not self._is_running:
                self.start()

            # Create session
            session_id = create_session_id()
            target = target_package or self._config.target_package or "unknown"

            self._current_session = ProfileSession(
                session_id=session_id,
                timestamp=datetime.now(),
                target_package=target,
                configuration=self._config,
                environment_info=self._collect_environment_info(),
                execution_context=ExecutionContext(
                    command_line=sys.argv.copy(), start_time=datetime.now()
                ),
            )

            # Install collectors
            self._install_collectors()

            # Start collection thread
            self._start_collection_thread()

            if self._config.verbose:
                print(f"Pycroscope: Session {session_id} started")

            return session_id

    def end_session(self) -> Optional[ProfileSession]:
        """
        End the current profiling session.

        Returns:
            Completed session data, or None if no session was active
        """
        with self._session_lock:
            if self._current_session is None:
                return None

            # Uninstall collectors
            self._uninstall_collectors()

            # Stop collection thread
            if self._collection_thread and self._collection_thread.is_alive():
                self._collection_thread.join(timeout=5.0)

            # Finalize session
            session = self._finalize_session()

            if self._config.verbose and session:
                print(f"Pycroscope: Session {session.session_id} ended")

            # Store session if data store is available
            if self._data_store and session:
                try:
                    self._data_store.store_session(session)
                except Exception as e:
                    if self._config.debug_mode:
                        print(f"Warning: Failed to store session: {e}")

            self._current_session = None
            return session

    @contextmanager
    def session(self, target_package: Optional[str] = None):
        """
        Context manager for automatic session lifecycle.

        Args:
            target_package: Optional package name to profile

        Yields:
            Session identifier
        """
        session_id = self.begin_session(target_package)
        try:
            yield session_id
        finally:
            self.end_session()

    def analyze_session(self, session: ProfileSession) -> Optional[AnalysisResult]:
        """
        Analyze a completed profiling session.

        Args:
            session: Completed profiling session

        Returns:
            Analysis results, or None if no analyzers available
        """
        if not self._analyzers:
            return None

        # Run analysis passes
        analysis_results = []
        for analyzer in self._analyzers:
            try:
                result = analyzer.analyze(session)
                analysis_results.append(result)
            except Exception as e:
                if self._config.debug_mode:
                    print(f"Analysis error in {analyzer.name}: {e}")

        if not analysis_results:
            return None

        # Combine analysis results (simplified for now)
        # In a full implementation, this would be more sophisticated
        combined_result = analysis_results[0]  # Placeholder

        if self._config.verbose:
            print(f"Pycroscope: Analysis completed for session {session.session_id}")

        return combined_result

    def create_dashboard(
        self, analysis_result: AnalysisResult
    ) -> Optional["Dashboard"]:
        """
        Create visualization dashboard from analysis results.

        Args:
            analysis_result: Complete analysis data

        Returns:
            Dashboard instance, or None if visualization disabled
        """
        if not self._visualizer:
            return None

        try:
            dashboard = self._visualizer.create_dashboard(analysis_result)

            if self._config.verbose:
                print("Pycroscope: Dashboard created")

            return dashboard
        except Exception as e:
            if self._config.debug_mode:
                print(f"Visualization error: {e}")
            return None

    def _install_collectors(self) -> None:
        """Install all configured collectors."""
        for collector in self._collectors.values():
            try:
                collector.install()
                if self._config.debug_mode:
                    print(f"Installed collector: {collector.name}")
            except Exception as e:
                if self._config.debug_mode:
                    print(f"Failed to install collector {collector.name}: {e}")

    def _uninstall_collectors(self) -> None:
        """Uninstall all collectors."""
        for collector in self._collectors.values():
            try:
                collector.uninstall()
                if self._config.debug_mode:
                    print(f"Uninstalled collector: {collector.name}")
            except Exception as e:
                if self._config.debug_mode:
                    print(f"Failed to uninstall collector {collector.name}: {e}")

    def _start_collection_thread(self) -> None:
        """Start background thread for event collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            return

        self._collection_thread = threading.Thread(
            target=self._collection_loop, name="pycroscope-collector", daemon=True
        )
        self._collection_thread.start()

    def _collection_loop(self) -> None:
        """Background collection loop."""
        while self._is_running and self._current_session:
            try:
                # Collect events from all collectors
                for collector in self._collectors.values():
                    if collector.is_installed:
                        for event_data in collector.collect():
                            # Convert raw collector data to ExecutionEvent
                            execution_event = self._process_collector_event(
                                event_data, collector
                            )
                            if execution_event:
                                # Buffer the processed event
                                with self._buffer_lock:
                                    self._events_buffer.append(execution_event)

                # Small sleep to prevent overwhelming the system
                time.sleep(0.001)  # 1ms

            except Exception as e:
                if self._config.debug_mode:
                    print(f"Collection error: {e}")

    def _process_collector_event(
        self, event_data: Dict[str, Any], collector: Collector
    ) -> Optional[ExecutionEvent]:
        """
        Convert raw collector event data to ExecutionEvent.

        Args:
            event_data: Raw event data from collector
            collector: Source collector instance

        Returns:
            Processed ExecutionEvent or None if invalid
        """
        try:
            # Extract common fields
            timestamp = event_data.get("timestamp", time.perf_counter_ns())
            event_type_str = event_data.get("event_type", "unknown")
            thread_id = event_data.get("thread_id", threading.get_ident())

            # Map event type string to EventType enum
            event_type = self._map_event_type(event_type_str)

            # Create frame info if available
            frame_info = self._extract_frame_info(event_data)

            # Skip events without valid frame info for now
            if not frame_info:
                # Create a default frame info for events without location data
                frame_info = FrameInfo(
                    source_location=SourceLocation(
                        filename="<unknown>", line_number=0, function_name="<unknown>"
                    ),
                    local_variables={},
                )

            # Extract execution time
            execution_time = event_data.get("execution_time")

            # Create ExecutionEvent
            execution_event = ExecutionEvent(
                timestamp=timestamp,
                event_type=event_type,
                thread_id=thread_id,
                frame_info=frame_info,
                execution_time=execution_time,
                event_data={
                    "collector": collector.name,
                    **{
                        k: v
                        for k, v in event_data.items()
                        if k
                        not in [
                            "timestamp",
                            "event_type",
                            "thread_id",
                            "execution_time",
                        ]
                    },
                },
            )

            return execution_event

        except Exception as e:
            if self._config.debug_mode:
                print(f"Error processing event from {collector.name}: {e}")
            return None

    def _map_event_type(self, event_type_str: str) -> EventType:
        """Map string event type to EventType enum."""
        mapping = {
            "call": EventType.CALL,
            "return": EventType.RETURN,
            "line": EventType.LINE,
            "exception": EventType.EXCEPTION,
            "memory_alloc": EventType.MEMORY_ALLOC,
            "memory_dealloc": EventType.MEMORY_DEALLOC,
            "memory_snapshot": EventType.MEMORY_ALLOC,  # Map to closest
            "gc_start": EventType.GC_START,
            "gc_stop": EventType.GC_END,
            "io_read": EventType.IO_READ,
            "io_write": EventType.IO_WRITE,
            "function_cpu": EventType.CALL,  # Map CPU events to call events
            "high_cpu_usage": EventType.CALL,
            "exception_cpu_impact": EventType.EXCEPTION,
            "import": EventType.CALL,  # Map import events to call events
            "open": EventType.IO_READ,  # Map file operations to IO events
            "write": EventType.IO_WRITE,
            "read": EventType.IO_READ,
            "close": EventType.IO_READ,
            "connect": EventType.IO_READ,  # Map network operations to IO events
            "send": EventType.IO_WRITE,
            "recv": EventType.IO_READ,
        }

        return mapping.get(event_type_str, EventType.LINE)  # Default to LINE

    def _extract_frame_info(self, event_data: Dict[str, Any]) -> Optional[FrameInfo]:
        """Extract frame information from event data."""
        try:
            # Look for frame information in various formats
            filename = (
                event_data.get("filename")
                or event_data.get("source_location", "").split(":")[0]
            )
            line_number = event_data.get("line_number", 0)
            function_name = event_data.get("function_name", "unknown")

            if not filename:
                return None

            # Create source location
            source_location = SourceLocation(
                filename=filename, line_number=line_number, function_name=function_name
            )

            # Create frame info (don't capture locals for performance)
            return FrameInfo(source_location=source_location, local_variables={})

        except Exception:
            return None

    def _finalize_session(self) -> Optional[ProfileSession]:
        """Finalize the current session with collected data."""
        if not self._current_session:
            return None

        # Flush all collectors and process any remaining events
        all_events = []
        for collector in self._collectors.values():
            try:
                events = collector.flush()
                # Process each flushed event
                for event_data in events:
                    execution_event = self._process_collector_event(
                        event_data, collector
                    )
                    if execution_event:
                        all_events.append(execution_event)
            except Exception as e:
                if self._config.debug_mode:
                    print(f"Error flushing collector {collector.name}: {e}")

        # Add any events from the buffer
        with self._buffer_lock:
            all_events.extend(self._events_buffer)
            self._events_buffer.clear()

        # Update execution context
        execution_context = self._current_session.execution_context
        if execution_context:
            execution_context = ExecutionContext(
                command_line=execution_context.command_line,
                start_time=execution_context.start_time,
                end_time=datetime.now(),
                exit_code=0,
            )

        # Create memory snapshots from memory events
        memory_snapshots = self._extract_memory_snapshots(all_events)

        # Build call tree from call events
        call_tree = self._build_call_tree(all_events)

        # Create source mapping from events
        source_mapping = self._build_source_mapping(all_events)

        # Create finalized session with processed events
        return ProfileSession(
            session_id=self._current_session.session_id,
            timestamp=self._current_session.timestamp,
            target_package=self._current_session.target_package,
            configuration=self._current_session.configuration,
            execution_events=all_events,  # Now contains all processed events
            memory_snapshots=memory_snapshots,
            call_tree=call_tree,
            source_mapping=source_mapping,
            environment_info=self._current_session.environment_info,
            execution_context=execution_context,
        )

    def _extract_memory_snapshots(
        self, events: List[ExecutionEvent]
    ) -> List[MemorySnapshot]:
        """Extract memory snapshots from execution events."""
        from ..core.models import MemorySnapshot

        snapshots = []
        for event in events:
            if (
                event.event_data.get("collector") == "memory"
                and "current_memory" in event.event_data
            ):
                snapshot = MemorySnapshot(
                    timestamp=event.timestamp,
                    total_memory=event.event_data.get("current_memory", 0),
                    peak_memory=event.event_data.get("peak_memory", 0),
                    gc_collections=event.event_data.get("gc_collections", 0),
                    object_counts=event.event_data.get("object_counts", {}),
                )
                snapshots.append(snapshot)

        return snapshots

    def _build_call_tree(self, events: List[ExecutionEvent]) -> Optional[CallTree]:
        """Build call tree from execution events."""
        from ..core.models import CallNode, CallTree

        # Simple call tree building - group by function calls
        call_events = [
            e for e in events if e.event_type in [EventType.CALL, EventType.RETURN]
        ]

        if not call_events:
            return None

        # Create root source location
        root_location = SourceLocation(
            filename="<root>", line_number=0, function_name="<root>"
        )

        # Calculate total time from events
        if call_events:
            start_time = min(e.timestamp for e in call_events)
            end_time = max(e.timestamp for e in call_events)
            total_time = end_time - start_time
        else:
            total_time = 0

        # Create root node
        root = CallNode(
            source_location=root_location,
            total_time=total_time,
            self_time=0,
            call_count=1,
            memory_allocated=0,
            children=[],
        )

        return CallTree(
            root=root,
            total_calls=len(call_events),
            total_time=total_time,
            max_depth=1,  # Simple implementation
        )

    def _build_source_mapping(
        self, events: List[ExecutionEvent]
    ) -> Dict[str, SourceLocation]:
        """Build source location mapping from events."""
        mapping = {}

        for event in events:
            if event.frame_info and event.frame_info.source_location:
                loc = event.frame_info.source_location
                key = f"{loc.filename}:{loc.line_number}"
                mapping[key] = loc

        return mapping

    def _collect_environment_info(self) -> EnvironmentInfo:
        """Collect current environment information."""
        import os
        import platform

        import psutil

        return EnvironmentInfo(
            python_version=platform.python_version(),
            platform=platform.platform(),
            cpu_count=os.cpu_count() or 1,
            memory_total=psutil.virtual_memory().total,
            working_directory=str(self._config.working_directory),
            environment_variables={
                k: v for k, v in os.environ.items() if k.startswith(("PYTHON", "PY"))
            },
        )

    def __enter__(self) -> "ProfilerSuite":
        """Context manager entry."""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disable()

    def __del__(self) -> None:
        """Destructor for cleanup."""
        try:
            if self._is_running:
                self.stop()
        except:
            pass  # Ignore errors during cleanup
