"""Essential tests for data configuration constants."""
import pytest
from pathlib import Path
from src.utils.config_data import (
    PROJECT_ROOT,
    REFERENCE_RATIO,
    VALIDATION_RATIO,
    PRODUCTION_RATIO,
    NUM_BATCHES
)


class TestConfigData:
    """Test configuration constants."""
    
    def test_ratios_sum_to_one(self):
        """Test that data split ratios sum to 1.0."""
        total = REFERENCE_RATIO + VALIDATION_RATIO + PRODUCTION_RATIO
        assert total == pytest.approx(1.0, abs=0.01)
    
    def test_paths_are_valid(self):
        """Test that configured paths are Path objects."""
        assert isinstance(PROJECT_ROOT, Path)
    
    def test_num_batches_positive(self):
        """Test that number of batches is positive."""
        assert NUM_BATCHES > 0
        assert isinstance(NUM_BATCHES, int)