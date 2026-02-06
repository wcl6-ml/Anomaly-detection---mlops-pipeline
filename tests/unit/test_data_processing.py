"""Essential tests for data loading and splitting."""

import pytest
import pandas as pd
import json

from src.data.loader import load_raw_data
from src.data.splitter import create_time_splits, save_splits


class TestDataLoader:
    """Core data loading tests."""
    
    def test_load_raw_data_sorts_by_time(self, temp_data_dir):
        """Test that data is sorted by Time."""
        # Create unsorted data
        df = pd.DataFrame({
            'Time': [300, 100, 200],
            'V1': [1, 2, 3],
            'Class': [0, 1, 0]
        })
        
        temp_file = temp_data_dir / "test.csv"
        df.to_csv(temp_file, index=False)
        
        loaded_df = load_raw_data(temp_file)
        
        assert loaded_df['Time'].is_monotonic_increasing
        assert loaded_df['Time'].tolist() == [100, 200, 300]


class TestDataSplitter:
    """Core splitting tests."""
    
    def test_splits_maintain_chronological_order(self, sample_creditcard_data):
        """Test that reference -> validation -> batches are in time order."""
        reference, validation, batches = create_time_splits(
            sample_creditcard_data,
            reference_ratio=0.5,
            validation_ratio=0.2,
            num_batches=3
        )
        
        # Check time ordering
        assert reference['Time'].max() <= validation['Time'].min()
        assert validation['Time'].max() <= batches[0]['Time'].min()
        
        # Check no data loss
        total = len(reference) + len(validation) + sum(len(b) for b in batches)
        assert total == len(sample_creditcard_data)
    
    def test_save_splits_creates_all_files(self, sample_creditcard_data, temp_data_dir):
        """Test that all files are created with correct structure."""
        reference, validation, batches = create_time_splits(
            sample_creditcard_data,
            reference_ratio=0.5,
            validation_ratio=0.2,
            num_batches=3
        )
        
        save_splits(reference, validation, batches, temp_data_dir)
        
        # Check main files exist
        assert (temp_data_dir / "reference.csv").exists()
        assert (temp_data_dir / "validation.csv").exists()
        assert (temp_data_dir / "batches" / "batch_001.csv").exists()
        assert (temp_data_dir / "splits_metadata.json").exists()
        
        # Verify saved data is correct
        loaded_ref = pd.read_csv(temp_data_dir / "reference.csv")
        assert len(loaded_ref) == len(reference)