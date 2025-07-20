"""
File-based data store implementation for Pycroscope.

Provides persistent storage of profiling sessions using the file system
with automatic cleanup, indexing, and efficient retrieval.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from ..core.interfaces import DataStore, Configurable
from ..core.models import ProfileSession
from ..core.config import StorageConfig
from .session_serializer import SessionSerializer, SerializationError


class FileDataStore(DataStore, Configurable):
    """
    File-based implementation of the DataStore interface.

    Stores profiling sessions as individual files with automatic indexing,
    cleanup, and compression support.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the file data store.

        Args:
            config: Storage configuration
        """
        self._config = config or StorageConfig()
        self._storage_path = Path(
            self._config.storage_path or Path.home() / ".pycroscope" / "data"
        )
        self._serializer = SessionSerializer(
            format="compressed_json" if self._config.compression else "json"
        )

        # Session index for fast lookups
        self._index_file = self._storage_path / "session_index.json"
        self._index: Dict[str, Dict] = {}
        self._index_lock = threading.RLock()

        # Initialize storage
        self._initialize_storage()
        self._load_index()

        # Auto-cleanup if enabled
        if self._config.auto_cleanup:
            self._cleanup_old_sessions()

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration settings."""
        # Update configuration from dictionary
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def store_session(self, session: ProfileSession) -> str:
        """
        Store a profiling session to disk.

        Args:
            session: ProfileSession to store

        Returns:
            Session identifier for retrieval
        """
        try:
            # Create session file path
            session_file = self._get_session_file_path(session.session_id)

            # Serialize and save session
            self._serializer.serialize_to_file(session, session_file)

            # Update index
            self._update_index(session)

            # Cleanup old sessions if we exceed the limit
            self._enforce_session_limit()

            return session.session_id

        except Exception as e:
            raise RuntimeError(f"Failed to store session {session.session_id}: {e}")

    def load_session(self, session_id: str) -> Optional[ProfileSession]:
        """
        Load a stored profiling session.

        Args:
            session_id: Session identifier

        Returns:
            ProfileSession if found, None otherwise
        """
        try:
            # Check if session exists in index
            with self._index_lock:
                if session_id not in self._index:
                    return None

            # Load session from file
            session_file = self._get_session_file_path(session_id)

            if not session_file.exists():
                # Remove from index if file is missing
                self._remove_from_index(session_id)
                return None

            session = self._serializer.deserialize_from_file(session_file)

            # Update access time in index
            self._update_access_time(session_id)

            return session

        except SerializationError:
            # Handle corrupted session files
            self._handle_corrupted_session(session_id)
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to load session {session_id}: {e}")

    def list_sessions(
        self, package_name: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """
        List available profiling sessions.

        Args:
            package_name: Filter by target package name
            limit: Maximum number of sessions to return

        Returns:
            List of session identifiers
        """
        with self._index_lock:
            sessions = []

            for session_id, metadata in self._index.items():
                # Filter by package name if specified
                if package_name and metadata.get("target_package") != package_name:
                    continue

                sessions.append(session_id)

            # Sort by creation time (newest first)
            sessions.sort(
                key=lambda sid: self._index[sid].get("timestamp", ""), reverse=True
            )

            # Apply limit
            if limit:
                sessions = sessions[:limit]

            return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a stored profiling session.

        Args:
            session_id: Session identifier to delete

        Returns:
            True if deletion successful, False if session not found
        """
        try:
            # Check if session exists
            with self._index_lock:
                if session_id not in self._index:
                    return False

            # Delete session file
            session_file = self._get_session_file_path(session_id)

            if session_file.exists():
                session_file.unlink()

            # Remove from index
            self._remove_from_index(session_id)

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to delete session {session_id}: {e}")

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """
        Get metadata for a session without loading the full session.

        Args:
            session_id: Session identifier

        Returns:
            Metadata dictionary or None if not found
        """
        with self._index_lock:
            return self._index.get(session_id)

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage information
        """
        with self._index_lock:
            total_sessions = len(self._index)

            # Calculate storage size
            total_size = 0
            for session_id in self._index:
                session_file = self._get_session_file_path(session_id)
                if session_file.exists():
                    total_size += session_file.stat().st_size

            # Group by package
            packages = {}
            for metadata in self._index.values():
                package = metadata.get("target_package", "unknown")
                packages[package] = packages.get(package, 0) + 1

            return {
                "total_sessions": total_sessions,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "storage_path": str(self._storage_path),
                "sessions_by_package": packages,
                "index_file_exists": self._index_file.exists(),
                "compression_enabled": self._config.compression,
            }

    def cleanup_old_sessions(self, max_age_days: Optional[int] = None) -> int:
        """
        Remove sessions older than specified age.

        Args:
            max_age_days: Maximum age in days (uses config default if None)

        Returns:
            Number of sessions removed
        """
        max_age = max_age_days or self._config.retention_days
        cutoff_date = datetime.now() - timedelta(days=max_age)

        removed_count = 0

        with self._index_lock:
            sessions_to_remove = []

            for session_id, metadata in self._index.items():
                timestamp_str = metadata.get("timestamp", "")
                try:
                    session_date = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if session_date < cutoff_date:
                        sessions_to_remove.append(session_id)
                except ValueError:
                    # Remove sessions with invalid timestamps
                    sessions_to_remove.append(session_id)

        # Remove old sessions
        for session_id in sessions_to_remove:
            if self.delete_session(session_id):
                removed_count += 1

        return removed_count

    def _initialize_storage(self) -> None:
        """Initialize storage directory structure."""
        # Create storage directory
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Create sessions subdirectory
        sessions_dir = self._storage_path / "sessions"
        sessions_dir.mkdir(exist_ok=True)

    def _get_session_file_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        # Use subdirectories to avoid too many files in one directory
        prefix = session_id[:2] if len(session_id) >= 2 else "00"
        subdir = self._storage_path / "sessions" / prefix
        subdir.mkdir(exist_ok=True)

        return subdir / f"{session_id}.pycroscope"

    def _load_index(self) -> None:
        """Load session index from disk."""
        if not self._index_file.exists():
            self._index = {}
            return

        try:
            with open(self._index_file, "r") as f:
                self._index = json.load(f)
        except Exception:
            # Rebuild index if corrupted
            self._rebuild_index()

    def _save_index(self) -> None:
        """Save session index to disk."""
        try:
            with open(self._index_file, "w") as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            # Index save failure is not fatal, but should be logged
            pass

    def _update_index(self, session: ProfileSession) -> None:
        """Update index with session metadata."""
        with self._index_lock:
            metadata = {
                "session_id": session.session_id,
                "timestamp": session.timestamp.isoformat(),
                "target_package": session.target_package,
                "total_events": session.total_events,
                "peak_memory": session.peak_memory,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "file_size": 0,  # Will be updated after file is written
            }

            self._index[session.session_id] = metadata

            # Update file size
            session_file = self._get_session_file_path(session.session_id)
            if session_file.exists():
                metadata["file_size"] = session_file.stat().st_size

            self._save_index()

    def _update_access_time(self, session_id: str) -> None:
        """Update last access time for a session."""
        with self._index_lock:
            if session_id in self._index:
                self._index[session_id]["last_accessed"] = datetime.now().isoformat()
                self._save_index()

    def _remove_from_index(self, session_id: str) -> None:
        """Remove session from index."""
        with self._index_lock:
            if session_id in self._index:
                del self._index[session_id]
                self._save_index()

    def _rebuild_index(self) -> None:
        """Rebuild index by scanning session files."""
        self._index = {}

        sessions_dir = self._storage_path / "sessions"
        if not sessions_dir.exists():
            return

        # Scan all session files
        for session_file in sessions_dir.rglob("*.pycroscope"):
            try:
                # Extract metadata from file
                with open(session_file, "rb") as f:
                    data = f.read()

                metadata = self._serializer.get_metadata(data)
                if metadata.get("session_id"):
                    session_id = metadata["session_id"]

                    self._index[session_id] = {
                        "session_id": session_id,
                        "timestamp": metadata.get("timestamp", ""),
                        "target_package": metadata.get("target_package", ""),
                        "created_at": datetime.now().isoformat(),
                        "last_accessed": datetime.now().isoformat(),
                        "file_size": session_file.stat().st_size,
                    }

            except Exception:
                # Skip corrupted files
                continue

        self._save_index()

    def _enforce_session_limit(self) -> None:
        """Remove oldest sessions if we exceed the configured limit."""
        with self._index_lock:
            if len(self._index) <= self._config.max_sessions:
                return

            # Sort sessions by creation time (oldest first)
            sessions_by_age = sorted(
                self._index.items(), key=lambda item: item[1].get("created_at", "")
            )

            # Remove oldest sessions
            excess_count = len(self._index) - self._config.max_sessions
            for i in range(excess_count):
                session_id = sessions_by_age[i][0]
                self.delete_session(session_id)

    def _cleanup_old_sessions(self) -> None:
        """Automatic cleanup of old sessions."""
        if self._config.retention_days > 0:
            removed = self.cleanup_old_sessions()
            if removed > 0:
                # Could log this information
                pass

    def _handle_corrupted_session(self, session_id: str) -> None:
        """Handle corrupted session files."""
        # Remove from index
        self._remove_from_index(session_id)

        # Optionally move corrupted file to quarantine
        session_file = self._get_session_file_path(session_id)
        if session_file.exists():
            quarantine_dir = self._storage_path / "quarantine"
            quarantine_dir.mkdir(exist_ok=True)

            quarantine_file = quarantine_dir / f"{session_id}_corrupted.pycroscope"
            try:
                session_file.rename(quarantine_file)
            except Exception:
                # If move fails, just delete
                session_file.unlink(missing_ok=True)
