"""Essential tests for drift detection."""

import pytest
import pandas as pd
import numpy as np

from src.drift.detector import DriftDetector


class TestDriftDetector:
    """Core drift detection tests."""
    
    def test_no_drift_when_identical_data(self):
        """Test that identical data shows no drift."""
        # Create reference data
        np.random.seed(42)
        reference = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000)
        })
        
        detector = DriftDetector(reference, threshold_ks=0.05, threshold_psi=0.1)
        
        # Test with same data
        results = detector.detect_drift(reference)
        
        assert results['drift_detected'] == False
        assert len(results['drifted_features']) == 0
        assert results['overall_psi'] < 0.1
    
    def test_drift_detected_with_shifted_data(self):
        """Test that drift is detected when data distribution shifts."""
        np.random.seed(42)
        
        # Reference data
        reference = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000)
        })
        
        # Shifted data (mean +3)
        shifted = pd.DataFrame({
            'feature1': np.random.randn(1000) + 5,  # Significant shift
            'feature2': np.random.randn(1000) + 5
        })
        
        detector = DriftDetector(reference, threshold_ks=0.05, threshold_psi=0.1)
        results = detector.detect_drift(shifted)
        
        assert results['drift_detected'] == True
        assert len(results['drifted_features']) > 0
    
    
    def test_psi_calculation(self):
        """Test PSI calculation returns valid values."""
        np.random.seed(42)
        reference = pd.DataFrame({'feature1': np.random.randn(1000)})
        
        detector = DriftDetector(reference)
        
        # PSI with identical distribution
        expected = np.random.randn(500)
        actual = np.random.randn(500)
        psi = detector.calculate_psi(expected, actual)
        
        assert isinstance(psi, float)
        assert psi >= 0  # PSI is always non-negative
        assert not np.isnan(psi)
        assert not np.isinf(psi)
    
    def test_feature_level_drift_results(self):
        """Test that feature-level drift metrics are returned."""
        np.random.seed(42)
        reference = pd.DataFrame({
            'feature1': np.random.randn(500),
            'feature2': np.random.randn(500)
        })
        
        detector = DriftDetector(reference)
        results = detector.detect_drift(reference)
        
        # Check structure
        assert 'feature_drifts' in results
        assert 'feature1' in results['feature_drifts']
        assert 'feature2' in results['feature_drifts']
        
        # Check each feature has required metrics
        for feature_name, metrics in results['feature_drifts'].items():
            assert 'ks_statistic' in metrics
            assert 'ks_pvalue' in metrics
            assert 'psi' in metrics
            assert 'drift' in metrics
            assert isinstance(metrics['drift'], bool)