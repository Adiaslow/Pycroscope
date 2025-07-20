"""
Session serialization and deserialization for Pycroscope.

Handles conversion of ProfileSession objects to and from persistent formats
with support for compression, versioning, and data integrity.
"""

import json
import pickle
import gzip
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import asdict

from ..core.models import ProfileSession, ExecutionEvent, MemorySnapshot, CallTree
from ..core.config import ProfileConfig


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""

    pass


class SessionSerializer:
    """
    Handles serialization and deserialization of profiling sessions.

    Supports multiple formats and compression options while maintaining
    data integrity and version compatibility.
    """

    CURRENT_VERSION = "1.0"
    SUPPORTED_FORMATS = ["json", "pickle", "compressed_json", "compressed_pickle"]

    def __init__(self, format: str = "compressed_json", verify_integrity: bool = True):
        """
        Initialize the session serializer.

        Args:
            format: Serialization format to use
            verify_integrity: Whether to verify data integrity with checksums
        """
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}")

        self._format = format
        self._verify_integrity = verify_integrity

    def serialize_session(self, session: ProfileSession) -> bytes:
        """
        Serialize a ProfileSession to bytes.

        Args:
            session: ProfileSession to serialize

        Returns:
            Serialized session data as bytes

        Raises:
            SerializationError: If serialization fails
        """
        try:
            # Convert session to serializable format
            session_data = self._session_to_dict(session)

            # Add metadata
            serialized_data = {
                "version": self.CURRENT_VERSION,
                "format": self._format,
                "timestamp": datetime.now().isoformat(),
                "session": session_data,
            }

            # Serialize based on format
            if self._format in ["json", "compressed_json"]:
                data = json.dumps(
                    serialized_data, default=self._json_serializer
                ).encode("utf-8")
            elif self._format in ["pickle", "compressed_pickle"]:
                data = pickle.dumps(serialized_data)
            else:
                raise SerializationError(f"Unknown format: {self._format}")

            # Apply compression if requested
            if "compressed" in self._format:
                data = gzip.compress(data)

            # Add integrity check
            if self._verify_integrity:
                checksum = hashlib.sha256(data).hexdigest()
                header = f"PYCROSCOPE:{checksum}:".encode("utf-8")
                data = header + data

            return data

        except Exception as e:
            raise SerializationError(f"Failed to serialize session: {e}")

    def deserialize_session(self, data: bytes) -> ProfileSession:
        """
        Deserialize bytes to a ProfileSession.

        Args:
            data: Serialized session data

        Returns:
            Reconstructed ProfileSession

        Raises:
            SerializationError: If deserialization fails
        """
        try:
            # Verify integrity if enabled
            if self._verify_integrity:
                if not data.startswith(b"PYCROSCOPE:"):
                    raise SerializationError("Invalid data format - missing header")

                # Extract checksum and data
                header_end = data.find(b":", 11)  # Find second colon
                if header_end == -1:
                    raise SerializationError("Invalid header format")

                expected_checksum = data[11:header_end].decode("utf-8")
                actual_data = data[header_end + 1 :]

                # Verify checksum
                actual_checksum = hashlib.sha256(actual_data).hexdigest()
                if expected_checksum != actual_checksum:
                    raise SerializationError("Data integrity check failed")

                data = actual_data

            # Decompress if needed
            if "compressed" in self._format:
                data = gzip.decompress(data)

            # Deserialize based on format
            if self._format in ["json", "compressed_json"]:
                serialized_data = json.loads(data.decode("utf-8"))
            elif self._format in ["pickle", "compressed_pickle"]:
                serialized_data = pickle.loads(data)
            else:
                raise SerializationError(f"Unknown format: {self._format}")

            # Check version compatibility
            version = serialized_data.get("version", "unknown")
            if not self._is_version_compatible(version):
                raise SerializationError(f"Incompatible version: {version}")

            # Reconstruct session
            session_data = serialized_data["session"]
            return self._dict_to_session(session_data)

        except Exception as e:
            raise SerializationError(f"Failed to deserialize session: {e}")

    def serialize_to_file(
        self, session: ProfileSession, file_path: Union[str, Path]
    ) -> None:
        """
        Serialize session directly to a file.

        Args:
            session: ProfileSession to serialize
            file_path: Path to write serialized data
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.serialize_session(session)

        with open(file_path, "wb") as f:
            f.write(data)

    def deserialize_from_file(self, file_path: Union[str, Path]) -> ProfileSession:
        """
        Deserialize session from a file.

        Args:
            file_path: Path to read serialized data from

        Returns:
            Reconstructed ProfileSession
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise SerializationError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            data = f.read()

        return self.deserialize_session(data)

    def _session_to_dict(self, session: ProfileSession) -> Dict[str, Any]:
        """Convert ProfileSession to dictionary."""
        return {
            "session_id": session.session_id,
            "timestamp": session.timestamp.isoformat(),
            "target_package": session.target_package,
            "configuration": self._config_to_dict(session.configuration),
            "environment_info": self._environment_info_to_dict(
                session.environment_info
            ),
            "execution_context": self._execution_context_to_dict(
                session.execution_context
            ),
            "execution_events": [
                self._event_to_dict(event) for event in session.execution_events
            ],
            "memory_snapshots": [
                self._memory_snapshot_to_dict(snapshot)
                for snapshot in session.memory_snapshots
            ],
            "call_tree": (
                self._call_tree_to_dict(session.call_tree)
                if session.call_tree
                else None
            ),
            "source_mapping": {
                k: self._source_location_to_dict(v)
                for k, v in session.source_mapping.items()
            },
            "analysis_result": (
                self._analysis_result_to_dict(session.analysis_result)
                if session.analysis_result
                else None
            ),
            "total_events": session.total_events,
            "peak_memory": session.peak_memory,
        }

    def _dict_to_session(self, data: Dict[str, Any]) -> ProfileSession:
        """Convert dictionary back to ProfileSession."""
        from ..core.models import (
            ProfileSession,
            ExecutionEvent,
            MemorySnapshot,
            EnvironmentInfo,
            ExecutionContext,
            AnalysisResult,
            CallTree,
        )

        # Reconstruct timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])

        # Reconstruct all components
        config = self._dict_to_config(data["configuration"])
        environment_info = self._dict_to_environment_info(data["environment_info"])
        execution_context = self._dict_to_execution_context(data["execution_context"])

        # Reconstruct events
        execution_events = [
            self._dict_to_event(event_data)
            for event_data in data.get("execution_events", [])
        ]

        # Reconstruct memory snapshots
        memory_snapshots = [
            self._dict_to_memory_snapshot(snapshot_data)
            for snapshot_data in data.get("memory_snapshots", [])
        ]

        # Reconstruct call tree
        call_tree = None
        if data.get("call_tree"):
            call_tree = self._dict_to_call_tree(data["call_tree"])

        # Reconstruct source mapping
        source_mapping = {}
        for key, location_data in data.get("source_mapping", {}).items():
            source_mapping[key] = self._dict_to_source_location(location_data)

        # Reconstruct analysis result
        analysis_result = None
        if data.get("analysis_result"):
            analysis_result = self._dict_to_analysis_result(data["analysis_result"])

        return ProfileSession(
            session_id=data["session_id"],
            timestamp=timestamp,
            target_package=data["target_package"],
            configuration=config,
            environment_info=environment_info,
            execution_context=execution_context,
            execution_events=execution_events,
            memory_snapshots=memory_snapshots,
            call_tree=call_tree,
            source_mapping=source_mapping,
            analysis_result=analysis_result,
        )

    def _config_to_dict(self, config: ProfileConfig) -> Dict[str, Any]:
        """Convert ProfileConfig to dictionary."""
        return {
            "target_package": config.target_package,
            "working_directory": str(config.working_directory),
            "debug_mode": config.debug_mode,
            "verbose": config.verbose,
            "parallel_collection": config.parallel_collection,
            "max_threads": config.max_threads,
            "timeout_seconds": config.timeout_seconds,
            "collectors": {
                str(ct): {
                    "enabled": collector_config.enabled,
                    "sampling_rate": collector_config.sampling_rate,
                    "buffer_size": collector_config.buffer_size,
                    "include_patterns": collector_config.include_patterns,
                    "exclude_patterns": collector_config.exclude_patterns,
                }
                for ct, collector_config in config.collectors.items()
            },
            "analysis": {
                "enabled_analyzers": [
                    str(a) for a in config.analysis.enabled_analyzers
                ],
                "confidence_threshold": config.analysis.confidence_threshold,
                "impact_threshold": config.analysis.impact_threshold,
            },
            "storage": {
                "storage_type": str(config.storage.storage_type),
                "storage_path": (
                    str(config.storage.storage_path)
                    if config.storage.storage_path
                    else None
                ),
                "compression": config.storage.compression,
                "max_sessions": config.storage.max_sessions,
                "retention_days": config.storage.retention_days,
                "auto_cleanup": config.storage.auto_cleanup,
            },
        }

    def _dict_to_config(self, data: Dict[str, Any]) -> ProfileConfig:
        """Convert dictionary to ProfileConfig."""
        from ..core.config import (
            ProfileConfig,
            CollectorConfig,
            AnalysisConfig,
            StorageConfig,
        )
        from ..core.config import CollectorType, AnalysisType, StorageType
        from pathlib import Path

        config = ProfileConfig()
        config.target_package = data.get("target_package")
        config.working_directory = Path(data.get("working_directory", "."))
        config.debug_mode = data.get("debug_mode", False)
        config.verbose = data.get("verbose", False)
        config.parallel_collection = data.get("parallel_collection", True)
        config.max_threads = data.get("max_threads", 4)
        config.timeout_seconds = data.get("timeout_seconds")

        # Reconstruct collectors config
        collectors_data = data.get("collectors", {})
        config.collectors = {}
        for ct_str, collector_data in collectors_data.items():
            try:
                collector_type = CollectorType(ct_str)
                collector_config = CollectorConfig(
                    enabled=collector_data.get("enabled", True),
                    sampling_rate=collector_data.get("sampling_rate", 1.0),
                    buffer_size=collector_data.get("buffer_size", 10000),
                    include_patterns=collector_data.get("include_patterns", []),
                    exclude_patterns=collector_data.get("exclude_patterns", []),
                )
                config.collectors[collector_type] = collector_config
            except ValueError:
                continue  # Skip unknown collector types

        # Reconstruct analysis config
        analysis_data = data.get("analysis", {})
        config.analysis = AnalysisConfig()
        config.analysis.enabled_analyzers = set()
        for analyzer_str in analysis_data.get("enabled_analyzers", []):
            try:
                analyzer_type = AnalysisType(analyzer_str)
                config.analysis.enabled_analyzers.add(analyzer_type)
            except ValueError:
                continue  # Skip unknown analyzer types
        config.analysis.confidence_threshold = analysis_data.get(
            "confidence_threshold", 0.7
        )
        config.analysis.impact_threshold = analysis_data.get("impact_threshold", 0.5)

        # Reconstruct storage config
        storage_data = data.get("storage", {})
        config.storage = StorageConfig()
        storage_type_str = storage_data.get("storage_type", "file")
        try:
            config.storage.storage_type = StorageType(storage_type_str)
        except ValueError:
            config.storage.storage_type = StorageType.FILE

        if storage_data.get("storage_path"):
            config.storage.storage_path = Path(storage_data["storage_path"])
        config.storage.compression = storage_data.get("compression", True)
        config.storage.max_sessions = storage_data.get("max_sessions", 100)
        config.storage.retention_days = storage_data.get("retention_days", 30)
        config.storage.auto_cleanup = storage_data.get("auto_cleanup", True)

        return config

    def _environment_info_to_dict(self, env_info) -> Dict[str, Any]:
        """Convert EnvironmentInfo to dictionary."""
        return {
            "python_version": env_info.python_version,
            "platform": env_info.platform,
            "cpu_count": env_info.cpu_count,
            "memory_total": env_info.memory_total,
            "working_directory": env_info.working_directory,
        }

    def _dict_to_environment_info(self, data: Dict[str, Any]):
        """Convert dictionary to EnvironmentInfo."""
        from ..core.models import EnvironmentInfo

        return EnvironmentInfo(
            python_version=data["python_version"],
            platform=data["platform"],
            cpu_count=data["cpu_count"],
            memory_total=data["memory_total"],
            working_directory=data["working_directory"],
        )

    def _execution_context_to_dict(self, exec_context) -> Dict[str, Any]:
        """Convert ExecutionContext to dictionary."""
        return {
            "command_line": exec_context.command_line,
            "start_time": exec_context.start_time.isoformat(),
            "end_time": (
                exec_context.end_time.isoformat() if exec_context.end_time else None
            ),
            "exit_code": exec_context.exit_code,
        }

    def _dict_to_execution_context(self, data: Dict[str, Any]):
        """Convert dictionary to ExecutionContext."""
        from ..core.models import ExecutionContext

        start_time = datetime.fromisoformat(data["start_time"])
        end_time = (
            datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None
        )

        return ExecutionContext(
            command_line=data["command_line"],
            start_time=start_time,
            end_time=end_time,
            exit_code=data.get("exit_code"),
        )

    def _event_to_dict(self, event: ExecutionEvent) -> Dict[str, Any]:
        """Convert ExecutionEvent to dictionary."""
        return {
            "timestamp": event.timestamp,
            "event_type": str(event.event_type),
            "thread_id": event.thread_id,
            "frame_info": {
                "source_location": {
                    "filename": event.frame_info.source_location.filename,
                    "line_number": event.frame_info.source_location.line_number,
                    "function_name": event.frame_info.source_location.function_name,
                },
                "local_variables": event.frame_info.local_variables,
                "frame_id": event.frame_info.frame_id,
            },
            "execution_time": event.execution_time,
            "memory_delta": event.memory_delta,
            "cpu_usage": event.cpu_usage,
            "call_stack": event.call_stack,
            "event_data": event.event_data,
        }

    def _dict_to_event(self, data: Dict[str, Any]) -> ExecutionEvent:
        """Convert dictionary to ExecutionEvent."""
        from ..core.models import ExecutionEvent, FrameInfo, SourceLocation, EventType

        # Reconstruct event type
        event_type = EventType(data["event_type"])

        # Reconstruct source location
        source_location_data = data["frame_info"]["source_location"]
        source_location = SourceLocation(
            filename=source_location_data["filename"],
            line_number=source_location_data["line_number"],
            function_name=source_location_data["function_name"],
        )

        # Reconstruct frame info
        frame_info = FrameInfo(
            source_location=source_location,
            local_variables=data["frame_info"].get("local_variables", {}),
            frame_id=data["frame_info"].get("frame_id", ""),
        )

        return ExecutionEvent(
            timestamp=data["timestamp"],
            event_type=event_type,
            thread_id=data["thread_id"],
            frame_info=frame_info,
            execution_time=data.get("execution_time"),
            memory_delta=data.get("memory_delta"),
            cpu_usage=data.get("cpu_usage"),
            call_stack=data.get("call_stack", []),
            event_data=data.get("event_data", {}),
        )

    def _memory_snapshot_to_dict(self, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """Convert MemorySnapshot to dictionary."""
        return {
            "timestamp": snapshot.timestamp,
            "total_memory": snapshot.total_memory,
            "peak_memory": snapshot.peak_memory,
            "gc_collections": snapshot.gc_collections,
            "object_counts": snapshot.object_counts,
        }

    def _dict_to_memory_snapshot(self, data: Dict[str, Any]) -> MemorySnapshot:
        """Convert dictionary to MemorySnapshot."""
        from ..core.models import MemorySnapshot

        return MemorySnapshot(
            timestamp=data["timestamp"],
            total_memory=data["total_memory"],
            peak_memory=data["peak_memory"],
            gc_collections=data["gc_collections"],
            object_counts=data.get("object_counts", {}),
        )

    def _call_tree_to_dict(self, call_tree: CallTree) -> Dict[str, Any]:
        """Convert CallTree to dictionary."""
        return {
            "root": self._call_node_to_dict(
                getattr(
                    call_tree,
                    "root",
                    call_tree.root if hasattr(call_tree, "root") else None,
                )
            ),
            "total_calls": getattr(call_tree, "total_calls", 0),
            "total_time": getattr(call_tree, "total_time", 0),
            "max_depth": getattr(call_tree, "max_depth", 0),
        }

    def _dict_to_call_tree(self, data: Dict[str, Any]) -> CallTree:
        """Convert dictionary to CallTree."""
        from ..core.models import CallTree

        root_node = self._dict_to_call_node(data["root"])

        # Ensure we have a valid root node
        if root_node is None:
            from ..core.models import CallNode, SourceLocation

            root_node = CallNode(
                source_location=SourceLocation("", 0, ""),
                total_time=0,
                self_time=0,
                call_count=0,
                memory_allocated=0,
            )

        return CallTree(
            root=root_node,
            total_calls=data.get("total_calls", 0),
            total_time=data.get("total_time", 0),
            max_depth=data.get("max_depth", 0),
        )

    def _call_node_to_dict(self, node) -> Dict[str, Any]:
        """Convert CallNode to dictionary."""
        if node is None:
            return {}

        return {
            "source_location": self._source_location_to_dict(node.source_location),
            "total_time": getattr(node, "total_time", 0),
            "self_time": getattr(node, "self_time", 0),
            "call_count": getattr(node, "call_count", 0),
            "memory_allocated": getattr(node, "memory_allocated", 0),
            "children": [
                self._call_node_to_dict(child)
                for child in getattr(node, "children", [])
            ],
        }

    def _dict_to_call_node(self, data: Dict[str, Any]):
        """Convert dictionary to CallNode."""
        if not data:
            return None

        from ..core.models import CallNode

        source_location = self._dict_to_source_location(data["source_location"])

        node = CallNode(
            source_location=source_location,
            total_time=data.get("total_time", 0),
            self_time=data.get("self_time", 0),
            call_count=data.get("call_count", 0),
            memory_allocated=data.get("memory_allocated", 0),
        )

        # Add children if they exist
        children_data = data.get("children", [])
        if hasattr(node, "children") and isinstance(children_data, list):
            for child_data in children_data:
                child_node = self._dict_to_call_node(child_data)
                if child_node:
                    node.children.append(child_node)

        return node

    def _source_location_to_dict(self, location) -> Dict[str, Any]:
        """Convert SourceLocation to dictionary."""
        return {
            "filename": location.filename,
            "line_number": location.line_number,
            "function_name": location.function_name,
        }

    def _dict_to_source_location(self, data: Dict[str, Any]):
        """Convert dictionary to SourceLocation."""
        from ..core.models import SourceLocation

        return SourceLocation(
            filename=data["filename"],
            line_number=data["line_number"],
            function_name=data["function_name"],
        )

    def _analysis_result_to_dict(self, analysis_result) -> Dict[str, Any]:
        """Convert AnalysisResult to dictionary."""
        return {
            "session_id": analysis_result.session_id,
            "analysis_timestamp": analysis_result.analysis_timestamp.isoformat(),
            "static_analysis": self._static_analysis_to_dict(
                analysis_result.static_analysis
            ),
            "dynamic_analysis": self._dynamic_analysis_to_dict(
                analysis_result.dynamic_analysis
            ),
            "detected_patterns": [
                self._pattern_to_dict(p) for p in analysis_result.detected_patterns
            ],
            "recommendations": [
                self._recommendation_to_dict(r) for r in analysis_result.recommendations
            ],
            "overall_score": analysis_result.overall_score,
            "performance_grade": analysis_result.performance_grade,
        }

    def _dict_to_analysis_result(self, data: Dict[str, Any]):
        """Convert dictionary to AnalysisResult."""
        from ..core.models import AnalysisResult

        analysis_timestamp = datetime.fromisoformat(data["analysis_timestamp"])
        static_analysis = self._dict_to_static_analysis(data["static_analysis"])
        dynamic_analysis = self._dict_to_dynamic_analysis(data["dynamic_analysis"])

        detected_patterns = [
            self._dict_to_pattern(pattern_data)
            for pattern_data in data.get("detected_patterns", [])
        ]

        # Filter out None recommendations
        recommendations = []
        for rec_data in data.get("recommendations", []):
            rec = self._dict_to_recommendation(rec_data)
            if rec is not None:
                recommendations.append(rec)

        return AnalysisResult(
            session_id=data["session_id"],
            analysis_timestamp=analysis_timestamp,
            static_analysis=static_analysis,
            dynamic_analysis=dynamic_analysis,
            detected_patterns=detected_patterns,
            recommendations=recommendations,
            overall_score=data["overall_score"],
            performance_grade=data["performance_grade"],
        )

    def _static_analysis_to_dict(self, static_analysis) -> Dict[str, Any]:
        """Convert StaticAnalysisResult to dictionary."""
        return {
            "complexity_metrics": static_analysis.complexity_metrics,
            "detected_patterns": [
                self._pattern_to_dict(p) for p in static_analysis.detected_patterns
            ],
            "code_quality_score": static_analysis.code_quality_score,
        }

    def _dict_to_static_analysis(self, data: Dict[str, Any]):
        """Convert dictionary to StaticAnalysisResult."""
        from ..core.models import StaticAnalysisResult

        detected_patterns = [
            self._dict_to_pattern(pattern_data)
            for pattern_data in data.get("detected_patterns", [])
        ]

        return StaticAnalysisResult(
            complexity_metrics=data.get("complexity_metrics", {}),
            detected_patterns=detected_patterns,
            code_quality_score=data.get("code_quality_score", 0.0),
        )

    def _dynamic_analysis_to_dict(self, dynamic_analysis) -> Dict[str, Any]:
        """Convert DynamicAnalysisResult to dictionary."""
        return {
            "hotspots": [
                self._call_node_to_dict(node) for node in dynamic_analysis.hotspots
            ],
            "memory_leaks": [
                self._pattern_to_dict(p) for p in dynamic_analysis.memory_leaks
            ],
            "performance_metrics": dynamic_analysis.performance_metrics,
        }

    def _dict_to_dynamic_analysis(self, data: Dict[str, Any]):
        """Convert dictionary to DynamicAnalysisResult."""
        from ..core.models import DynamicAnalysisResult

        # Filter out None hotspots
        hotspots = []
        for node_data in data.get("hotspots", []):
            node = self._dict_to_call_node(node_data)
            if node is not None:
                hotspots.append(node)

        memory_leaks = [
            self._dict_to_pattern(pattern_data)
            for pattern_data in data.get("memory_leaks", [])
        ]

        return DynamicAnalysisResult(
            hotspots=hotspots,
            memory_leaks=memory_leaks,
            performance_metrics=data.get("performance_metrics", {}),
        )

    def _pattern_to_dict(self, pattern) -> Dict[str, Any]:
        """Convert DetectedPattern to dictionary."""
        return {
            "pattern_type": pattern.pattern_type,
            "severity": pattern.severity,
            "source_location": self._source_location_to_dict(pattern.source_location),
            "description": pattern.description,
            "impact_estimate": pattern.impact_estimate,
            "evidence": pattern.evidence,
        }

    def _dict_to_pattern(self, data: Dict[str, Any]):
        """Convert dictionary to DetectedPattern."""
        from ..core.models import DetectedPattern

        source_location = self._dict_to_source_location(data["source_location"])

        return DetectedPattern(
            pattern_type=data["pattern_type"],
            severity=data["severity"],
            source_location=source_location,
            description=data["description"],
            impact_estimate=data["impact_estimate"],
            evidence=data.get("evidence", {}),
        )

    def _recommendation_to_dict(self, recommendation) -> Dict[str, Any]:
        """Convert OptimizationRecommendation to dictionary."""
        return {
            "title": getattr(recommendation, "title", ""),
            "description": getattr(recommendation, "description", ""),
            "category": getattr(recommendation, "category", ""),
            "estimated_improvement": getattr(
                recommendation, "estimated_improvement", 0.0
            ),
            "confidence": getattr(recommendation, "confidence", 0.0),
            "implementation_effort": getattr(
                recommendation, "implementation_effort", "medium"
            ),
            "suggested_actions": getattr(recommendation, "suggested_actions", []),
            "code_examples": getattr(recommendation, "code_examples", []),
            # Handle both new and old model formats
            "recommendation_id": getattr(recommendation, "recommendation_id", ""),
            "target_location": (
                self._source_location_to_dict(
                    getattr(recommendation, "target_location", None)
                )
                if hasattr(recommendation, "target_location")
                and getattr(recommendation, "target_location")
                else None
            ),
            "complexity": getattr(recommendation, "complexity", "medium"),
            "suggested_approach": getattr(recommendation, "suggested_approach", ""),
            "code_example": getattr(recommendation, "code_example", None),
            "resources": getattr(recommendation, "resources", []),
        }

    def _dict_to_recommendation(self, data: Dict[str, Any]):
        """Convert dictionary to OptimizationRecommendation."""
        # Try to use the newer model format first, fall back to basic structure
        try:
            from ..core.models import OptimizationRecommendation, SourceLocation

            # Create a minimal source location if target_location exists
            target_location = None
            if data.get("target_location"):
                target_location = self._dict_to_source_location(data["target_location"])
            else:
                # Create a dummy location
                target_location = SourceLocation("", 0, "")

            return OptimizationRecommendation(
                recommendation_id=data.get("recommendation_id", ""),
                title=data.get("title", ""),
                description=data.get("description", ""),
                target_location=target_location,
                estimated_improvement=data.get("estimated_improvement", 0.0),
                confidence=data.get("confidence", 0.0),
                complexity=data.get("complexity", "medium"),
                suggested_approach=data.get("suggested_approach", ""),
                code_example=data.get("code_example"),
                resources=data.get("resources", []),
            )
        except Exception:
            # Fall back to a simple object if the model doesn't match
            class SimpleRecommendation:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            return SimpleRecommendation(**data)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)

    def _is_version_compatible(self, version: str) -> bool:
        """Check if a version is compatible with current serializer."""
        # Simple version compatibility check
        if version == "unknown":
            return False

        # For now, only support exact version match
        return version == self.CURRENT_VERSION

    def get_metadata(self, data: bytes) -> Dict[str, Any]:
        """
        Extract metadata from serialized data without full deserialization.

        Args:
            data: Serialized session data

        Returns:
            Dictionary with metadata information
        """
        try:
            # Handle integrity check
            if self._verify_integrity and data.startswith(b"PYCROSCOPE:"):
                header_end = data.find(b":", 11)
                data = data[header_end + 1 :]

            # Decompress if needed
            if "compressed" in self._format:
                data = gzip.decompress(data)

            # Parse just the top level to get metadata
            if self._format in ["json", "compressed_json"]:
                metadata = json.loads(data.decode("utf-8"))
            elif self._format in ["pickle", "compressed_pickle"]:
                metadata = pickle.loads(data)
            else:
                return {}

            # Return just metadata fields
            return {
                "version": metadata.get("version"),
                "format": metadata.get("format"),
                "timestamp": metadata.get("timestamp"),
                "session_id": metadata.get("session", {}).get("session_id"),
                "target_package": metadata.get("session", {}).get("target_package"),
            }

        except Exception:
            return {"error": "Failed to extract metadata"}


# Utility functions for common serialization tasks


def quick_save_session(session: ProfileSession, file_path: Union[str, Path]) -> None:
    """
    Quickly save a session with default settings.

    Args:
        session: ProfileSession to save
        file_path: Path to save to
    """
    serializer = SessionSerializer()
    serializer.serialize_to_file(session, file_path)


def quick_load_session(file_path: Union[str, Path]) -> ProfileSession:
    """
    Quickly load a session with default settings.

    Args:
        file_path: Path to load from

    Returns:
        Loaded ProfileSession
    """
    serializer = SessionSerializer()
    return serializer.deserialize_from_file(file_path)
