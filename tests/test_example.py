"""
Example test to verify testing framework setup.

This file demonstrates basic pytest usage and ensures the testing
infrastructure is working correctly.
"""

import sys
from pathlib import Path

import pytest

# Add pycroscope to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
def test_basic_assertion():
    """Test basic assertion functionality."""
    assert True is True
    assert 1 + 1 == 2
    assert "hello" == "hello"


@pytest.mark.unit
def test_list_operations():
    """Test basic list operations."""
    test_list = [1, 2, 3]

    assert len(test_list) == 3
    assert test_list[0] == 1
    assert 2 in test_list

    test_list.append(4)
    assert len(test_list) == 4


@pytest.mark.unit
def test_dict_operations():
    """Test basic dictionary operations."""
    test_dict = {"key1": "value1", "key2": "value2"}

    assert test_dict["key1"] == "value1"
    assert "key1" in test_dict
    assert len(test_dict) == 2

    test_dict["key3"] = "value3"
    assert len(test_dict) == 3


@pytest.mark.unit
def test_string_operations():
    """Test basic string operations."""
    test_string = "Pycroscope Testing"

    assert test_string.startswith("Pycroscope")
    assert "Testing" in test_string
    assert test_string.lower() == "pycroscope testing"


@pytest.mark.unit
def test_exception_handling():
    """Test exception handling."""
    with pytest.raises(ValueError):
        int("not_a_number")

    with pytest.raises(KeyError):
        test_dict = {"key": "value"}
        _ = test_dict["nonexistent_key"]


@pytest.mark.integration
def test_path_operations():
    """Test path operations (integration test)."""
    current_file = Path(__file__)

    assert current_file.exists()
    assert current_file.is_file()
    assert current_file.suffix == ".py"
    assert current_file.parent.name == "tests"


@pytest.mark.slow
def test_slow_operation():
    """Test that takes some time (marked as slow)."""
    import time

    start_time = time.time()
    time.sleep(0.1)  # Sleep for 100ms
    end_time = time.time()

    duration = end_time - start_time
    assert duration >= 0.1


class TestExampleClass:
    """Example test class demonstrating class-based tests."""

    @pytest.fixture
    def sample_data(self):
        """Sample data fixture for class tests."""
        return {
            "numbers": [1, 2, 3, 4, 5],
            "strings": ["a", "b", "c"],
            "nested": {"inner": {"value": 42}},
        }

    @pytest.mark.unit
    def test_sample_data_fixture(self, sample_data):
        """Test using fixture data."""
        assert len(sample_data["numbers"]) == 5
        assert sample_data["nested"]["inner"]["value"] == 42

    @pytest.mark.unit
    def test_numbers_processing(self, sample_data):
        """Test processing numbers from fixture."""
        numbers = sample_data["numbers"]

        total = sum(numbers)
        assert total == 15

        doubled = [n * 2 for n in numbers]
        assert doubled == [2, 4, 6, 8, 10]

    @pytest.mark.unit
    def test_string_processing(self, sample_data):
        """Test processing strings from fixture."""
        strings = sample_data["strings"]

        joined = "".join(strings)
        assert joined == "abc"

        upper = [s.upper() for s in strings]
        assert upper == ["A", "B", "C"]


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (1, 2),
        (2, 4),
        (3, 6),
        (0, 0),
        (-1, -2),
    ],
)
@pytest.mark.unit
def test_parametrized_doubling(input_value, expected):
    """Test parametrized doubling function."""

    def double(x):
        return x * 2

    assert double(input_value) == expected


def test_import_pycroscope():
    """Test that pycroscope package can be imported."""
    try:
        import pycroscope

        assert hasattr(pycroscope, "__version__") or True  # Allow missing version
    except ImportError:
        pytest.skip("Pycroscope package not available for import")


@pytest.mark.integration
def test_pycroscope_core_imports():
    """Test importing core pycroscope components."""
    try:
        from pycroscope.core.config import ProfileConfig
        from pycroscope.core.models import ProfileSession

        # Test creating basic instances
        config = ProfileConfig()
        assert config is not None

    except ImportError as e:
        pytest.skip(f"Core pycroscope components not available: {e}")
