import pytest
from src.preprocessing.dataset import get_dataset

def test_get_dataset_invalid_path():
    """Test that get_dataset gracefully handles non-existent file paths."""
    df = get_dataset("invalid/path/file.csv")
    assert df is None