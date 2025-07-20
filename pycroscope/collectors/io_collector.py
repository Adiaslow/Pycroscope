"""
I/O operations profiler for Pycroscope.

Monitors file and network I/O operations to identify performance
bottlenecks in data access and transfer operations.
"""

import functools
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..core.models import EventType, ExecutionEvent, FrameInfo, SourceLocation
from .base import BaseCollector


class IOCollector(BaseCollector):
    """
    Collector for I/O operation profiling.

    Tracks file and network I/O operations to identify performance
    bottlenecks and optimization opportunities in data access patterns.
    """

    def __init__(self, config=None):
        """
        Initialize the I/O collector.

        Args:
            config: Optional collector configuration
        """
        super().__init__(config)

        # Original functions (to restore)
        self._original_open = None
        self._original_socket_functions = {}
        self._original_os_functions = {}

        # I/O tracking
        self._io_stats = {
            "file_operations": {
                "total_reads": 0,
                "total_writes": 0,
                "total_bytes_read": 0,
                "total_bytes_written": 0,
                "total_file_time": 0,
                "expensive_operations": [],
                "file_access_patterns": defaultdict(list),
            },
            "network_operations": {
                "total_connections": 0,
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "total_network_time": 0,
                "connection_patterns": defaultdict(list),
                "expensive_operations": [],
            },
        }

        # Active file/socket tracking
        self._active_files = {}
        self._active_sockets = {}
        self._file_descriptors = {}

        # Performance thresholds
        self._slow_io_threshold_ms = 10  # 10ms
        self._very_slow_io_threshold_ms = 100  # 100ms

        # Thread safety
        self._io_lock = threading.RLock()

        # Track file types and patterns
        self._file_types = defaultdict(int)
        self._access_patterns = defaultdict(list)

    @property
    def name(self) -> str:
        """Unique identifier for this collector type."""
        return "io"

    def _install_hooks(self) -> None:
        """Install I/O monitoring hooks."""
        # Save original functions
        self._original_open = open

        # Hook file operations
        __builtins__["open"] = self._open_hook

        # Hook socket operations
        self._hook_socket_operations()

        # Hook os operations
        self._hook_os_operations()

    def _uninstall_hooks(self) -> None:
        """Remove I/O monitoring hooks."""
        # Restore original functions
        if self._original_open:
            __builtins__["open"] = self._original_open

        # Restore socket functions
        self._unhook_socket_operations()

        # Restore os functions
        self._unhook_os_operations()

    def _collect_events(self) -> Iterator[Dict[str, Any]]:
        """Collect I/O events from buffer."""
        # Return buffered events - BaseCollector handles the actual buffering
        yield from []

    def _open_hook(
        self,
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        """Hook into file open operations."""
        start_time = time.perf_counter_ns()

        # Get calling frame info
        frame = sys._getframe(1)
        caller_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        try:
            # Perform the actual open
            if self._original_open and callable(self._original_open):
                file_obj = self._original_open(
                    file, mode, buffering, encoding, errors, newline, closefd, opener
                )
            else:
                # Fallback to built-in open if original is not available
                import builtins

                file_obj = builtins.open(
                    file, mode, buffering, encoding, errors, newline, closefd, opener
                )

            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            # Wrap the file object to track I/O operations
            wrapped_file = FileWrapper(file_obj, self, caller_location, str(file), mode)

            # Record the open operation
            self._record_file_operation(
                "open",
                str(file),
                mode,
                duration,
                caller_location,
                bytes_transferred=0,
                success=True,
            )

            # Track the file
            with self._io_lock:
                self._active_files[id(wrapped_file)] = {
                    "file_path": str(file),
                    "mode": mode,
                    "opened_at": start_time,
                    "opener_location": caller_location,
                    "operations": [],
                }

            return wrapped_file

        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            # Record failed open operation
            self._record_file_operation(
                "open",
                str(file),
                mode,
                duration,
                caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )

            # Re-raise the exception
            raise

    def _hook_socket_operations(self):
        """Hook socket operations for network I/O monitoring."""
        # Save original socket functions
        self._original_socket_functions = {
            "socket": socket.socket,
            "connect": socket.socket.connect,
            "send": socket.socket.send,
            "recv": socket.socket.recv,
            "sendall": socket.socket.sendall,
            "close": socket.socket.close,
        }

        # Replace with our hooks
        socket.socket = self._socket_hook

    def _unhook_socket_operations(self):
        """Restore original socket operations."""
        for name, original_func in self._original_socket_functions.items():
            if name == "socket":
                socket.socket = original_func

    def _hook_os_operations(self):
        """Hook OS-level file operations."""
        self._original_os_functions = {
            "read": os.read if hasattr(os, "read") else None,
            "write": os.write if hasattr(os, "write") else None,
            "open": os.open if hasattr(os, "open") else None,
            "close": os.close if hasattr(os, "close") else None,
        }

        # Hook the functions that exist
        if hasattr(os, "read"):
            os.read = self._os_read_hook
        if hasattr(os, "write"):
            os.write = self._os_write_hook

    def _unhook_os_operations(self):
        """Restore original OS operations."""
        for name, original_func in self._original_os_functions.items():
            if original_func and hasattr(os, name):
                setattr(os, name, original_func)

    def _socket_hook(
        self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None
    ):
        """Hook socket creation."""
        start_time = time.perf_counter_ns()

        # Get calling frame info
        frame = sys._getframe(1)
        caller_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        # Create the socket
        sock = self._original_socket_functions["socket"](family, type, proto, fileno)

        # Wrap the socket to track operations
        wrapped_socket = SocketWrapper(sock, self, caller_location)

        end_time = time.perf_counter_ns()
        duration = end_time - start_time

        # Record socket creation
        self._record_network_operation(
            "socket_create",
            "",
            duration,
            caller_location,
            bytes_transferred=0,
            success=True,
        )

        return wrapped_socket

    def _os_read_hook(self, fd, length):
        """Hook os.read operations."""
        start_time = time.perf_counter_ns()

        # Get calling frame info
        frame = sys._getframe(1)
        caller_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        try:
            # Perform the read
            data = self._original_os_functions["read"](fd, length)

            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            # Record the operation
            self._record_file_operation(
                "os_read",
                f"fd:{fd}",
                "r",
                duration,
                caller_location,
                bytes_transferred=len(data),
                success=True,
            )

            return data

        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._record_file_operation(
                "os_read",
                f"fd:{fd}",
                "r",
                duration,
                caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )

            raise

    def _os_write_hook(self, fd, data):
        """Hook os.write operations."""
        start_time = time.perf_counter_ns()

        # Get calling frame info
        frame = sys._getframe(1)
        caller_location = SourceLocation(
            filename=frame.f_code.co_filename,
            line_number=frame.f_lineno,
            function_name=frame.f_code.co_name,
        )

        try:
            # Perform the write
            bytes_written = self._original_os_functions["write"](fd, data)

            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            # Record the operation
            self._record_file_operation(
                "os_write",
                f"fd:{fd}",
                "w",
                duration,
                caller_location,
                bytes_transferred=bytes_written,
                success=True,
            )

            return bytes_written

        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._record_file_operation(
                "os_write",
                f"fd:{fd}",
                "w",
                duration,
                caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )

            raise

    def _record_file_operation(
        self,
        operation,
        file_path,
        mode,
        duration,
        caller_location,
        bytes_transferred=0,
        success=True,
        error=None,
    ):
        """Record a file I/O operation."""
        current_time = time.perf_counter_ns()

        # Create frame info
        frame_info = FrameInfo(source_location=caller_location, local_variables={})

        # Determine event type
        event_type = (
            EventType.IO_READ
            if "r" in mode or "read" in operation
            else EventType.IO_WRITE
        )

        # Create execution event
        event = ExecutionEvent(
            timestamp=current_time,
            event_type=event_type,
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            execution_time=duration,
            event_data={
                "operation": operation,
                "file_path": file_path,
                "mode": mode,
                "bytes_transferred": bytes_transferred,
                "duration_ms": duration / 1_000_000,
                "success": success,
                "error": error,
                "is_slow": duration > self._slow_io_threshold_ms * 1_000_000,
                "is_very_slow": duration > self._very_slow_io_threshold_ms * 1_000_000,
                "file_type": self._get_file_type(file_path),
            },
        )

        # Buffer the event
        self._add_to_buffer(event.__dict__)

        # Update statistics
        self._update_file_stats(
            operation, file_path, mode, duration, bytes_transferred, success
        )

    def _record_network_operation(
        self,
        operation,
        endpoint,
        duration,
        caller_location,
        bytes_transferred=0,
        success=True,
        error=None,
    ):
        """Record a network I/O operation."""
        current_time = time.perf_counter_ns()

        # Create frame info
        frame_info = FrameInfo(source_location=caller_location, local_variables={})

        # Determine event type based on operation
        if "send" in operation or "write" in operation:
            event_type = EventType.IO_WRITE
        else:
            event_type = EventType.IO_READ

        # Create execution event
        event = ExecutionEvent(
            timestamp=current_time,
            event_type=event_type,
            thread_id=threading.get_ident(),
            frame_info=frame_info,
            execution_time=duration,
            event_data={
                "operation": operation,
                "endpoint": endpoint,
                "bytes_transferred": bytes_transferred,
                "duration_ms": duration / 1_000_000,
                "success": success,
                "error": error,
                "is_slow": duration > self._slow_io_threshold_ms * 1_000_000,
                "is_very_slow": duration > self._very_slow_io_threshold_ms * 1_000_000,
                "protocol": (
                    "tcp"
                    if "tcp" in operation.lower()
                    else "udp" if "udp" in operation.lower() else "unknown"
                ),
            },
        )

        # Buffer the event
        self._add_to_buffer(event.__dict__)

        # Update statistics
        self._update_network_stats(
            operation, endpoint, duration, bytes_transferred, success
        )

    def _update_file_stats(
        self, operation, file_path, mode, duration, bytes_transferred, success
    ):
        """Update file I/O statistics."""
        with self._io_lock:
            if success:
                if "r" in mode or "read" in operation:
                    self._io_stats["file_operations"]["total_reads"] += 1
                    self._io_stats["file_operations"][
                        "total_bytes_read"
                    ] += bytes_transferred
                else:
                    self._io_stats["file_operations"]["total_writes"] += 1
                    self._io_stats["file_operations"][
                        "total_bytes_written"
                    ] += bytes_transferred

                self._io_stats["file_operations"]["total_file_time"] += duration

                # Track expensive operations
                duration_ms = duration / 1_000_000
                if duration_ms > self._slow_io_threshold_ms:
                    expensive_info = {
                        "operation": operation,
                        "file_path": file_path,
                        "duration_ms": duration_ms,
                        "bytes_transferred": bytes_transferred,
                        "timestamp": time.perf_counter_ns(),
                        "severity": (
                            "very_slow"
                            if duration_ms > self._very_slow_io_threshold_ms
                            else "slow"
                        ),
                    }
                    self._io_stats["file_operations"]["expensive_operations"].append(
                        expensive_info
                    )

                # Track file access patterns
                self._io_stats["file_operations"]["file_access_patterns"][
                    file_path
                ].append(
                    {
                        "operation": operation,
                        "timestamp": time.perf_counter_ns(),
                        "bytes": bytes_transferred,
                        "duration_ms": duration_ms,
                    }
                )

    def _update_network_stats(
        self, operation, endpoint, duration, bytes_transferred, success
    ):
        """Update network I/O statistics."""
        with self._io_lock:
            if success:
                if "connect" in operation:
                    self._io_stats["network_operations"]["total_connections"] += 1
                elif "send" in operation:
                    self._io_stats["network_operations"][
                        "total_bytes_sent"
                    ] += bytes_transferred
                elif "recv" in operation:
                    self._io_stats["network_operations"][
                        "total_bytes_received"
                    ] += bytes_transferred

                self._io_stats["network_operations"]["total_network_time"] += duration

                # Track expensive operations
                duration_ms = duration / 1_000_000
                if duration_ms > self._slow_io_threshold_ms:
                    expensive_info = {
                        "operation": operation,
                        "endpoint": endpoint,
                        "duration_ms": duration_ms,
                        "bytes_transferred": bytes_transferred,
                        "timestamp": time.perf_counter_ns(),
                        "severity": (
                            "very_slow"
                            if duration_ms > self._very_slow_io_threshold_ms
                            else "slow"
                        ),
                    }
                    self._io_stats["network_operations"]["expensive_operations"].append(
                        expensive_info
                    )

                # Track connection patterns
                if endpoint:
                    self._io_stats["network_operations"]["connection_patterns"][
                        endpoint
                    ].append(
                        {
                            "operation": operation,
                            "timestamp": time.perf_counter_ns(),
                            "bytes": bytes_transferred,
                            "duration_ms": duration_ms,
                        }
                    )

    def _get_file_type(self, file_path):
        """Determine file type from path."""
        try:
            path = Path(file_path)
            return path.suffix.lower() if path.suffix else "no_extension"
        except Exception:
            return "unknown"

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics including I/O analysis."""
        base_stats = super().get_stats()

        with self._io_lock:
            # Calculate I/O efficiency metrics
            file_efficiency = self._calculate_file_efficiency()
            network_efficiency = self._calculate_network_efficiency()

            # Get top file patterns
            top_files = self._get_top_accessed_files()

            # Get network patterns
            network_patterns = self._get_network_patterns()

            io_analysis = {
                "file_operations": {
                    **self._io_stats["file_operations"],
                    "average_read_size": self._get_average_read_size(),
                    "average_write_size": self._get_average_write_size(),
                    "efficiency_score": file_efficiency,
                    "top_accessed_files": top_files,
                    "file_type_distribution": dict(self._file_types),
                },
                "network_operations": {
                    **self._io_stats["network_operations"],
                    "average_connection_time": self._get_average_connection_time(),
                    "efficiency_score": network_efficiency,
                    "connection_patterns": network_patterns,
                },
                "active_files": len(self._active_files),
                "active_sockets": len(self._active_sockets),
                "io_recommendations": self._generate_io_recommendations(),
            }

            base_stats["io_analysis"] = io_analysis

        return base_stats

    def _calculate_file_efficiency(self) -> float:
        """Calculate file I/O efficiency score."""
        total_ops = (
            self._io_stats["file_operations"]["total_reads"]
            + self._io_stats["file_operations"]["total_writes"]
        )
        total_time = self._io_stats["file_operations"]["total_file_time"]

        if total_ops == 0 or total_time == 0:
            return 0.0

        # Operations per millisecond
        return total_ops / (total_time / 1_000_000)

    def _calculate_network_efficiency(self) -> float:
        """Calculate network I/O efficiency score."""
        total_bytes = (
            self._io_stats["network_operations"]["total_bytes_sent"]
            + self._io_stats["network_operations"]["total_bytes_received"]
        )
        total_time = self._io_stats["network_operations"]["total_network_time"]

        if total_bytes == 0 or total_time == 0:
            return 0.0

        # Bytes per millisecond
        return total_bytes / (total_time / 1_000_000)

    def _get_average_read_size(self) -> float:
        """Get average file read size."""
        total_reads = self._io_stats["file_operations"]["total_reads"]
        if total_reads == 0:
            return 0.0
        return self._io_stats["file_operations"]["total_bytes_read"] / total_reads

    def _get_average_write_size(self) -> float:
        """Get average file write size."""
        total_writes = self._io_stats["file_operations"]["total_writes"]
        if total_writes == 0:
            return 0.0
        return self._io_stats["file_operations"]["total_bytes_written"] / total_writes

    def _get_average_connection_time(self) -> float:
        """Get average network connection time."""
        connections = self._io_stats["network_operations"]["total_connections"]
        if connections == 0:
            return 0.0
        return (
            self._io_stats["network_operations"]["total_network_time"]
            / connections
            / 1_000_000
        )

    def _get_top_accessed_files(self) -> List[Dict[str, Any]]:
        """Get most frequently accessed files."""
        file_patterns = self._io_stats["file_operations"]["file_access_patterns"]

        file_stats = []
        for file_path, operations in file_patterns.items():
            total_ops = len(operations)
            total_bytes = sum(op["bytes"] for op in operations)
            total_time = sum(op["duration_ms"] for op in operations)

            file_stats.append(
                {
                    "file_path": file_path,
                    "total_operations": total_ops,
                    "total_bytes": total_bytes,
                    "total_time_ms": total_time,
                    "average_op_time_ms": (
                        total_time / total_ops if total_ops > 0 else 0
                    ),
                }
            )

        return sorted(file_stats, key=lambda x: x["total_operations"], reverse=True)[
            :10
        ]

    def _get_network_patterns(self) -> Dict[str, Any]:
        """Analyze network connection patterns."""
        patterns = self._io_stats["network_operations"]["connection_patterns"]

        if not patterns:
            return {}

        # Find most active endpoints
        endpoint_stats = []
        for endpoint, operations in patterns.items():
            total_ops = len(operations)
            total_bytes = sum(op["bytes"] for op in operations)

            endpoint_stats.append(
                {
                    "endpoint": endpoint,
                    "total_operations": total_ops,
                    "total_bytes": total_bytes,
                }
            )

        return {
            "most_active_endpoints": sorted(
                endpoint_stats, key=lambda x: x["total_operations"], reverse=True
            )[:5],
            "total_unique_endpoints": len(patterns),
        }

    def _generate_io_recommendations(self) -> List[str]:
        """Generate I/O optimization recommendations."""
        recommendations = []

        # Check for small file operations
        avg_read_size = self._get_average_read_size()
        if avg_read_size > 0 and avg_read_size < 1024:  # Less than 1KB
            recommendations.append(
                "Consider buffering small file reads to improve efficiency"
            )

        # Check for frequent file access
        file_patterns = self._io_stats["file_operations"]["file_access_patterns"]
        for file_path, operations in file_patterns.items():
            if len(operations) > 100:  # Frequently accessed
                recommendations.append(
                    f"Consider caching frequently accessed file: {file_path}"
                )

        # Check for slow I/O operations
        expensive_file_ops = len(
            self._io_stats["file_operations"]["expensive_operations"]
        )
        expensive_network_ops = len(
            self._io_stats["network_operations"]["expensive_operations"]
        )

        if expensive_file_ops > 10:
            recommendations.append(
                "Multiple slow file operations detected - consider async I/O"
            )

        if expensive_network_ops > 5:
            recommendations.append(
                "Slow network operations detected - check connection pooling"
            )

        return recommendations


class FileWrapper:
    """Wrapper for file objects to track I/O operations."""

    def __init__(self, file_obj, collector, caller_location, file_path, mode):
        self._file = file_obj
        self._collector = collector
        self._caller_location = caller_location
        self._file_path = file_path
        self._mode = mode

    def read(self, size=-1):
        """Wrapped read operation."""
        start_time = time.perf_counter_ns()
        try:
            data = self._file.read(size)
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_file_operation(
                "read",
                self._file_path,
                self._mode,
                duration,
                self._caller_location,
                bytes_transferred=len(data) if data else 0,
                success=True,
            )

            return data
        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_file_operation(
                "read",
                self._file_path,
                self._mode,
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )
            raise

    def write(self, data):
        """Wrapped write operation."""
        start_time = time.perf_counter_ns()
        try:
            result = self._file.write(data)
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_file_operation(
                "write",
                self._file_path,
                self._mode,
                duration,
                self._caller_location,
                bytes_transferred=len(data) if data else 0,
                success=True,
            )

            return result
        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_file_operation(
                "write",
                self._file_path,
                self._mode,
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )
            raise

    def close(self):
        """Wrapped close operation."""
        start_time = time.perf_counter_ns()
        try:
            result = self._file.close()
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_file_operation(
                "close",
                self._file_path,
                self._mode,
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=True,
            )

            return result
        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_file_operation(
                "close",
                self._file_path,
                self._mode,
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )
            raise

    def __getattr__(self, name):
        """Delegate other attributes to the original file object."""
        return getattr(self._file, name)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class SocketWrapper:
    """Wrapper for socket objects to track network operations."""

    def __init__(self, socket_obj, collector, caller_location):
        self._socket = socket_obj
        self._collector = collector
        self._caller_location = caller_location
        self._endpoint = None

    def connect(self, address):
        """Wrapped connect operation."""
        start_time = time.perf_counter_ns()
        self._endpoint = str(address)

        try:
            result = self._socket.connect(address)
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_network_operation(
                "connect",
                self._endpoint,
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=True,
            )

            return result
        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_network_operation(
                "connect",
                self._endpoint,
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )
            raise

    def send(self, data, flags=0):
        """Wrapped send operation."""
        start_time = time.perf_counter_ns()
        try:
            bytes_sent = self._socket.send(data, flags)
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_network_operation(
                "send",
                self._endpoint or "unknown",
                duration,
                self._caller_location,
                bytes_transferred=bytes_sent,
                success=True,
            )

            return bytes_sent
        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_network_operation(
                "send",
                self._endpoint or "unknown",
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )
            raise

    def recv(self, bufsize, flags=0):
        """Wrapped recv operation."""
        start_time = time.perf_counter_ns()
        try:
            data = self._socket.recv(bufsize, flags)
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_network_operation(
                "recv",
                self._endpoint or "unknown",
                duration,
                self._caller_location,
                bytes_transferred=len(data) if data else 0,
                success=True,
            )

            return data
        except Exception as e:
            end_time = time.perf_counter_ns()
            duration = end_time - start_time

            self._collector._record_network_operation(
                "recv",
                self._endpoint or "unknown",
                duration,
                self._caller_location,
                bytes_transferred=0,
                success=False,
                error=str(e),
            )
            raise

    def __getattr__(self, name):
        """Delegate other attributes to the original socket object."""
        return getattr(self._socket, name)
