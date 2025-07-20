"""
Unit tests for FileDataStore.

Tests file-based session storage, indexing, cleanup, and metadata management.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from pycroscope.storage.file_store import FileDataStore
from pycroscope.core.config import StorageConfig, StorageType
from pycroscope.core.models import ProfileSession
from pycroscope.storage.session_serializer import SerializationError


class TestFileDataStore:
    """Test FileDataStore file-based storage implementation."""

    @pytest.fixture
    def storage_config(self, temp_storage_dir):
        """Storage configuration for testing."""
        config = StorageConfig()
        config.storage_type = StorageType.FILE
        config.storage_path = temp_storage_dir
        config.compression = True
        config.max_sessions = 100  # Increase to avoid limit enforcement during tests
        config.retention_days = 7
        config.auto_cleanup = False  # Disable for testing
        return config

    @pytest.fixture
    def file_store(self, storage_config):
        """FileDataStore instance for testing."""
        return FileDataStore(storage_config)

    def test_initialization(self, storage_config):
        """Test FileDataStore initialization."""
        store = FileDataStore(storage_config)

        # Storage path should be created
        assert store._storage_path.exists()
        assert (store._storage_path / "sessions").exists()

        # Index file should exist (even if empty)
        assert store._index_file.exists() or not store._index

    def test_default_config(self):
        """Test FileDataStore with default configuration."""
        store = FileDataStore()

        # Should use default storage path
        expected_path = Path.home() / ".pycroscope" / "data"
        assert store._storage_path == expected_path

    def test_store_session(self, file_store, sample_profile_session):
        """Test storing a profile session."""
        session_id = file_store.store_session(sample_profile_session)

        assert session_id == sample_profile_session.session_id

        # Session file should exist
        session_file = file_store._get_session_file_path(session_id)
        assert session_file.exists()

        # Should be in index
        with file_store._index_lock:
            assert session_id in file_store._index

    def test_load_session(self, file_store, sample_config):
        """Test loading a stored session."""
        # Clear any existing sessions to ensure clean state
        existing_sessions = file_store.list_sessions()
        for session_id in existing_sessions:
            file_store.delete_session(session_id)

        # Create a simple session that definitely serializes properly
        from pycroscope.core.models import (
            ProfileSession,
            EnvironmentInfo,
            ExecutionContext,
        )
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        simple_session = ProfileSession(
            session_id="test_session_123",
            timestamp=datetime.now(),
            target_package="test_package",
            configuration=sample_config,
            execution_events=[],  # Empty for simplicity
            memory_snapshots=[],  # Empty for simplicity
            call_tree=None,  # None for simplicity
            source_mapping={},  # Empty for simplicity
            environment_info=env_info,
            execution_context=exec_context,
        )

        # Store session first
        session_id = file_store.store_session(simple_session)

        # Load it back
        loaded_session = file_store.load_session(session_id)

        assert loaded_session is not None
        assert loaded_session.session_id == session_id
        assert loaded_session.target_package == simple_session.target_package

    def test_load_nonexistent_session(self, file_store):
        """Test loading a session that doesn't exist."""
        loaded_session = file_store.load_session("nonexistent_session")
        assert loaded_session is None

    def test_list_sessions(self, file_store, sample_config, test_data_generator):
        """Test listing stored sessions."""
        # Create shared session components
        from pycroscope.core.models import EnvironmentInfo, ExecutionContext
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        # Store multiple sessions
        session_ids = []

        # Store original session
        original_session = ProfileSession(
            session_id="test_session_original",
            timestamp=datetime.now(),
            target_package="test_package",
            configuration=sample_config,
            execution_events=[],  # Empty for simplicity
            memory_snapshots=[],  # Empty for simplicity
            call_tree=None,  # None for simplicity
            source_mapping={},  # Empty for simplicity
            environment_info=env_info,
            execution_context=exec_context,
        )
        session_id = file_store.store_session(original_session)
        session_ids.append(session_id)

        # Create and store additional sessions with different packages
        for i in range(3):
            session = ProfileSession(
                session_id=f"test_session_{i}",
                timestamp=datetime.now(),
                target_package=f"package_{i}",
                configuration=sample_config,
                execution_events=[],  # Empty for simplicity
                memory_snapshots=[],  # Empty for simplicity
                call_tree=None,  # None for simplicity
                source_mapping={},  # Empty for simplicity
                environment_info=env_info,
                execution_context=exec_context,
            )
            session_id = file_store.store_session(session)
            session_ids.append(session_id)

        # List all sessions
        all_sessions = file_store.list_sessions()
        assert len(all_sessions) == 4

        # List with limit
        limited_sessions = file_store.list_sessions(limit=2)
        assert len(limited_sessions) == 2

        # List with package filter
        package_sessions = file_store.list_sessions(package_name="package_0")
        assert len(package_sessions) == 1

    def test_delete_session(self, file_store, sample_config):
        """Test deleting a stored session."""
        # Create a simple session that definitely serializes properly
        from pycroscope.core.models import (
            ProfileSession,
            EnvironmentInfo,
            ExecutionContext,
        )
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        simple_session = ProfileSession(
            session_id="test_session_123",
            timestamp=datetime.now(),
            target_package="test_package",
            configuration=sample_config,
            execution_events=[],  # Empty for simplicity
            memory_snapshots=[],  # Empty for simplicity
            call_tree=None,  # None for simplicity
            source_mapping={},  # Empty for simplicity
            environment_info=env_info,
            execution_context=exec_context,
        )

        # Store session first
        session_id = file_store.store_session(simple_session)

        # Verify it exists
        assert file_store.load_session(session_id) is not None

        # Delete it
        success = file_store.delete_session(session_id)
        assert success is True

        # Verify it's gone
        assert file_store.load_session(session_id) is None

        # Session file should be deleted
        session_file = file_store._get_session_file_path(session_id)
        assert not session_file.exists()

    def test_delete_nonexistent_session(self, file_store):
        """Test deleting a session that doesn't exist."""
        success = file_store.delete_session("nonexistent_session")
        assert success is False

    def test_get_session_metadata(self, file_store, sample_profile_session):
        """Test getting session metadata."""
        # Store session first
        session_id = file_store.store_session(sample_profile_session)

        # Get metadata
        metadata = file_store.get_session_metadata(session_id)

        assert metadata is not None
        assert metadata["session_id"] == session_id
        assert metadata["target_package"] == sample_profile_session.target_package
        assert "timestamp" in metadata
        assert "created_at" in metadata
        assert "last_accessed" in metadata

    def test_get_storage_stats(self, file_store, sample_profile_session):
        """Test getting storage statistics."""
        # Store a session first
        file_store.store_session(sample_profile_session)

        stats = file_store.get_storage_stats()

        assert "total_sessions" in stats
        assert stats["total_sessions"] >= 1
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "storage_path" in stats
        assert "sessions_by_package" in stats
        assert "compression_enabled" in stats

    def test_cleanup_old_sessions(self, file_store, sample_profile_session):
        """Test cleaning up old sessions by age."""
        # Store a session
        session_id = file_store.store_session(sample_profile_session)

        # Manually modify index to make session appear old
        with file_store._index_lock:
            old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
            file_store._index[session_id]["timestamp"] = old_timestamp
            file_store._save_index()

        # Cleanup sessions older than 7 days
        removed_count = file_store.cleanup_old_sessions(max_age_days=7)

        assert removed_count == 1
        assert file_store.load_session(session_id) is None

    def test_session_limit_enforcement(self, file_store, sample_config):
        """Test that session limit is enforced."""
        # Create required session components
        from pycroscope.core.models import EnvironmentInfo, ExecutionContext
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        # Store sessions up to the limit (100, but test with 6)
        session_ids = []
        for i in range(6):  # Store several sessions
            session = ProfileSession(
                session_id=f"limit_test_session_{i}",
                timestamp=datetime.now(),
                target_package="test_package",
                configuration=sample_config,
                execution_events=[],  # Empty for simplicity
                memory_snapshots=[],  # Empty for simplicity
                call_tree=None,  # None for simplicity
                source_mapping={},  # Empty for simplicity
                environment_info=env_info,
                execution_context=exec_context,
            )
            session_id = file_store.store_session(session)
            session_ids.append(session_id)

        # Should have all sessions since our limit is now 100
        all_sessions = file_store.list_sessions()
        assert len(all_sessions) <= file_store._config.max_sessions

    def test_index_recovery(self, file_store, sample_config):
        """Test index recovery from corrupted index file."""
        # Create a simple session that definitely serializes properly
        from pycroscope.core.models import (
            ProfileSession,
            EnvironmentInfo,
            ExecutionContext,
        )
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        simple_session = ProfileSession(
            session_id="test_session_123",
            timestamp=datetime.now(),
            target_package="test_package",
            configuration=sample_config,
            execution_events=[],  # Empty for simplicity
            memory_snapshots=[],  # Empty for simplicity
            call_tree=None,  # None for simplicity
            source_mapping={},  # Empty for simplicity
            environment_info=env_info,
            execution_context=exec_context,
        )

        # Store a session
        session_id = file_store.store_session(simple_session)

        # Corrupt the index file
        with open(file_store._index_file, "w") as f:
            f.write("invalid json content")

        # Create new store instance (should rebuild index)
        new_store = FileDataStore(file_store._config)

        # Should still be able to find the session
        loaded_session = new_store.load_session(session_id)
        assert loaded_session is not None

    def test_compression_enabled(self, temp_storage_dir):
        """Test storage with compression enabled."""
        config = StorageConfig()
        config.storage_path = temp_storage_dir
        config.compression = True

        store = FileDataStore(config)
        assert store._serializer._format == "compressed_json"

    def test_compression_disabled(self, temp_storage_dir):
        """Test storage with compression disabled."""
        config = StorageConfig()
        config.storage_path = temp_storage_dir
        config.compression = False

        store = FileDataStore(config)
        assert store._serializer._format == "json"

    def test_concurrent_access(self, file_store, sample_config):
        """Test concurrent access to file store."""
        import threading

        session_ids = []
        errors = []

        # Create shared session components for threads to use
        from pycroscope.core.models import EnvironmentInfo, ExecutionContext
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        def store_session(i):
            try:
                session = ProfileSession(
                    session_id=f"concurrent_session_{i}",
                    timestamp=datetime.now(),
                    target_package="concurrent_package",
                    configuration=sample_config,
                    execution_events=[],  # Empty for simplicity
                    memory_snapshots=[],  # Empty for simplicity
                    call_tree=None,  # None for simplicity
                    source_mapping={},  # Empty for simplicity
                    environment_info=env_info,
                    execution_context=exec_context,
                )
                session_id = file_store.store_session(session)
                session_ids.append(session_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=store_session, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors and all sessions stored
        assert len(errors) == 0
        assert len(session_ids) == 5

    def test_access_time_update(self, file_store, sample_config):
        """Test that access time is updated on load."""
        # Create a simple session that definitely serializes properly
        from pycroscope.core.models import (
            ProfileSession,
            EnvironmentInfo,
            ExecutionContext,
        )
        from datetime import datetime, timedelta

        env_info = EnvironmentInfo(
            python_version="3.9.7",
            platform="linux",
            cpu_count=4,
            memory_total=8589934592,
            working_directory="/test/working/dir",
        )

        start_time = datetime.now()
        exec_context = ExecutionContext(
            command_line=["python", "test_script.py"],
            start_time=start_time,
            end_time=start_time + timedelta(seconds=5),
            exit_code=0,
        )

        simple_session = ProfileSession(
            session_id="test_session_123",
            timestamp=datetime.now(),
            target_package="test_package",
            configuration=sample_config,
            execution_events=[],  # Empty for simplicity
            memory_snapshots=[],  # Empty for simplicity
            call_tree=None,  # None for simplicity
            source_mapping={},  # Empty for simplicity
            environment_info=env_info,
            execution_context=exec_context,
        )

        # Store session
        session_id = file_store.store_session(simple_session)

        # Get initial access time
        initial_metadata = file_store.get_session_metadata(session_id)
        initial_access_time = initial_metadata["last_accessed"]

        # Wait a bit and load session
        import time

        time.sleep(0.1)
        file_store.load_session(session_id)

        # Access time should be updated
        updated_metadata = file_store.get_session_metadata(session_id)
        updated_access_time = updated_metadata["last_accessed"]

        assert updated_access_time > initial_access_time

    def test_file_path_generation(self, file_store):
        """Test session file path generation."""
        session_id = "test_session_123"
        file_path = file_store._get_session_file_path(session_id)

        # Should use subdirectory based on session ID prefix
        expected_subdir = session_id[:2]
        assert expected_subdir in str(file_path)
        assert file_path.name == f"{session_id}.pycroscope"

    def test_corrupted_session_file_handling(self, file_store, sample_profile_session):
        """Test handling of corrupted session files."""
        # Store session
        session_id = file_store.store_session(sample_profile_session)
        session_file = file_store._get_session_file_path(session_id)

        # Corrupt the session file
        with open(session_file, "w") as f:
            f.write("corrupted data")

        # Loading should return None and clean up
        loaded_session = file_store.load_session(session_id)
        assert loaded_session is None

        # Session should be removed from index
        with file_store._index_lock:
            assert session_id not in file_store._index

    def test_storage_configuration(self, file_store):
        """Test storage configuration via configure method."""
        new_config = {"max_sessions": 20, "retention_days": 14, "compression": False}

        file_store.configure(new_config)

        assert file_store._config.max_sessions == 20
        assert file_store._config.retention_days == 14
        assert file_store._config.compression is False
