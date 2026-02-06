"""Pytest fixtures shared across all tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_creditcard_data():
    """
    Create synthetic credit card data for testing.
    Mimics the structure of the real creditcard.csv dataset.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create time column (sorted)
    time = np.sort(np.random.uniform(0, 172800, n_samples))  # ~48 hours
    
    # Create V1-V28 features (PCA components in real dataset)
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = np.random.randn(n_samples)
    
    # Create Amount column
    features['Amount'] = np.random.uniform(0, 1000, n_samples)
    
    # Create Class column (fraud label) - ~2% fraud rate
    features['Class'] = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'Time': time,
        **features
    })
    
    return df


@pytest.fixture
def sample_model_data():
    """
    Create sample data for model training/testing (X, y format).
    """
    np.random.seed(42)
    n_samples = 500
    n_features = 29  # V1-V28 + Amount
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    
    return X, y


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return {
        'model': {
            'type': 'isolation_forest',
            'contamination': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'anomaly_threshold': 95
        },
        'training': {
            'reference_data': 'data/processed/reference.csv',
            'validation_data': 'data/processed/validation.csv',
            'epochs': 50,
            'learning_rate': 0.001
        },
        'mlflow': {
            'experiment_name': 'test_experiment',
            'run_name': 'test_run'
        }
    }