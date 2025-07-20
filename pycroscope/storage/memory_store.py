"""
In-memory data store implementation for Pycroscope.

Provides temporary storage of profiling sessions in memory,
useful for testing and scenarios where persistence is not required.
"""

import threading
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..core.interfaces import DataStore, Configurable
from ..core.models import ProfileSession
from ..core.config import StorageConfig


class MemoryDataStore(DataStore, Configurable):
    """
    In-memory implementation of the DataStore interface.

    Stores profiling sessions in memory with automatic cleanup
    based on session limits. Data is lost when the process ends.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the memory data store.

        Args:
            config: Storage configuration
        """
        self._config = config or StorageConfig()
        self._sessions: Dict[str, ProfileSession] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        # Track access patterns for LRU cleanup
        self._access_times: Dict[str, datetime] = {}
        self._creation_times: Dict[str, datetime] = {}

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings."""
        # Update configuration from dictionary
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def store_session(self, session: ProfileSession) -> str:
        """
        Store a profiling session in memory.

        Args:
            session: ProfileSession to store

        Returns:
            Session identifier for retrieval
        """
        with self._lock:
            session_id = session.session_id

            # Store session
            self._sessions[session_id] = session

            # Update metadata
            now = datetime.now()
            self._creation_times[session_id] = now
            self._access_times[session_id] = now

            self._metadata[session_id] = {
                "session_id": session_id,
                "timestamp": session.timestamp.isoformat(),
                "target_package": session.target_package,
                "total_events": session.total_events,
                "peak_memory": session.peak_memory,
                "created_at": now.isoformat(),
                "last_accessed": now.isoformat(),
            }

            # Enforce session limit
            self._enforce_session_limit()

            return session_id

    def load_session(self, session_id: str) -> Optional[ProfileSession]:
        """
        Load a stored profiling session from memory.

        Args:
            session_id: Session identifier

        Returns:
            ProfileSession if found, None otherwise
        """
        with self._lock:
            if session_id not in self._sessions:
                return None

            # Update access time
            self._access_times[session_id] = datetime.now()
            self._metadata[session_id]["last_accessed"] = datetime.now().isoformat()

            return self._sessions[session_id]

    def list_sessions(
        self, package_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """
        List available profiling sessions in memory.

        Args:
            package_name: Filter by target package name
            limit: Maximum number of sessions to return

        Returns:
            List of session identifiers
        """
        with self._lock:
            sessions = []

            for session_id, metadata in self._metadata.items():
                # Filter by package name if specified
                if package_name and metadata.get("target_package") != package_name:
                    continue

                sessions.append(session_id)

            # Sort by creation time (newest first)
            sessions.sort(
                key=lambda sid: self._creation_times.get(sid, datetime.min),
                reverse=True,
            )

            # Apply limit
            if limit:
                sessions = sessions[:limit]

            return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a stored profiling session from memory.

        Args:
            session_id: Session identifier to delete

        Returns:
            True if deletion successful, False if session not found
        """
        with self._lock:
            if session_id not in self._sessions:
                return False

            # Remove all traces of the session
            del self._sessions[session_id]
            del self._metadata[session_id]
            del self._creation_times[session_id]
            del self._access_times[session_id]

            return True

    def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a session without loading the full session.

        Args:
            session_id: Session identifier

        Returns:
            Metadata dictionary or None if not found
        """
        with self._lock:
            return self._metadata.get(session_id)

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage information
        """
        with self._lock:
            total_sessions = len(self._sessions)

            # Calculate approximate memory usage
            # This is a rough estimate based on object sizes
            total_memory = 0
            for session in self._sessions.values():
                # Rough estimate: events + memory snapshots + overhead
                total_memory += (
                    len(session.execution_events) * 100
                )  # ~100 bytes per event
                total_memory += (
                    len(session.memory_snapshots) * 200
                )  # ~200 bytes per snapshot
                total_memory += 1000  # overhead per session

            # Group by package
            packages = {}
            for metadata in self._metadata.values():
                package = metadata.get("target_package", "unknown")
                packages[package] = packages.get(package, 0) + 1

            return {
                "total_sessions": total_sessions,
                "estimated_memory_bytes": total_memory,
                "estimated_memory_mb": total_memory / (1024 * 1024),
                "storage_type": "memory",
                "sessions_by_package": packages,
                "max_sessions_limit": self._config.max_sessions,
                "oldest_session": (
                    min(self._creation_times.values()) if self._creation_times else None
                ),
                "newest_session": (
                    max(self._creation_times.values()) if self._creation_times else None
                ),
            }

    def clear_all(self) -> int:
        """
        Clear all stored sessions.

        Returns:
            Number of sessions removed
        """
        with self._lock:
            count = len(self._sessions)

            self._sessions.clear()
            self._metadata.clear()
            self._creation_times.clear()
            self._access_times.clear()

            return count

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get detailed memory usage information.

        Returns:
            Dictionary with memory usage breakdown
        """
        with self._lock:
            usage = {"total_sessions": len(self._sessions), "session_details": {}}

            for session_id, session in self._sessions.items():
                usage["session_details"][session_id] = {
                    "execution_events": len(session.execution_events),
                    "memory_snapshots": len(session.memory_snapshots),
                    "has_call_tree": session.call_tree is not None,
                    "has_analysis": session.analysis_result is not None,
                    "created_at": self._creation_times.get(session_id),
                    "last_accessed": self._access_times.get(session_id),
                }

            return usage

    def _enforce_session_limit(self) -> None:
        """Remove oldest sessions if we exceed the configured limit."""
        if len(self._sessions) <= self._config.max_sessions:
            return

        # Find sessions to remove (oldest by creation time)
        sessions_by_age = sorted(self._creation_times.items(), key=lambda item: item[1])

        # Remove excess sessions
        excess_count = len(self._sessions) - self._config.max_sessions
        for i in range(excess_count):
            session_id = sessions_by_age[i][0]
            self.delete_session(session_id)

    def get_lru_sessions(self, count: int = 5) -> List[str]:
        """
        Get least recently used sessions.

        Args:
            count: Number of LRU sessions to return

        Returns:
            List of session IDs ordered by access time (oldest first)
        """
        with self._lock:
            sessions_by_access = sorted(
                self._access_times.items(), key=lambda item: item[1]
            )

            return [session_id for session_id, _ in sessions_by_access[:count]]

    def get_oldest_sessions(self, count: int = 5) -> List[str]:
        """
        Get oldest sessions by creation time.

        Args:
            count: Number of oldest sessions to return

        Returns:
            List of session IDs ordered by creation time (oldest first)
        """
        with self._lock:
            sessions_by_creation = sorted(
                self._creation_times.items(), key=lambda item: item[1]
            )

            return [session_id for session_id, _ in sessions_by_creation[:count]]

    def __len__(self) -> int:
        """Return number of stored sessions."""
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        """Check if a session is stored."""
        return session_id in self._sessions

    def __iter__(self):
        """Iterate over session IDs."""
        return iter(self._sessions.keys())

    def keys(self):
        """Return session IDs."""
        return self._sessions.keys()

    def values(self):
        """Return stored sessions."""
        return self._sessions.values()

    def items(self):
        """Return session ID and session pairs."""
        return self._sessions.items()
