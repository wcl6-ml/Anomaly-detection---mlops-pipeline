"""Data drift detection module."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift between reference and production data."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold_ks: float = 0.05, 
                 threshold_psi: float = 0.1):
        """
        Args:
            reference_data: Training/reference dataset
            threshold_ks: KS test p-value threshold (lower = more sensitive)
            threshold_psi: PSI threshold (higher = drift)
        """
        self.reference_data = reference_data
        self.threshold_ks = threshold_ks
        self.threshold_psi = threshold_psi
        self.feature_names = reference_data.columns.tolist()
        
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                     buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        def scale_range(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-10)
        
        expected_scaled = scale_range(expected)
        actual_scaled = scale_range(actual)
        
        breakpoints = np.linspace(0, 1, buckets + 1)
        
        expected_percents = np.histogram(expected_scaled, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual_scaled, breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return psi
    
    def detect_drift(self, batch_data: pd.DataFrame) -> Dict[str, any]:
        """
        Detect drift in batch data compared to reference.
        
        Returns:
            dict with drift metrics and status
        """
        results = {
            'drift_detected': False,
            'feature_drifts': {},
            'overall_psi': 0.0,
            'drifted_features': []
        }
        
        psi_values = []
        
        for feature in self.feature_names:
            ref_values = self.reference_data[feature].values
            batch_values = batch_data[feature].values
            
            # KS test for statistical difference
            ks_stat, p_value = stats.ks_2samp(ref_values, batch_values)
            
            # PSI calculation
            psi = self.calculate_psi(ref_values, batch_values)
            psi_values.append(psi)
            
            # Determine drift
            drift = p_value < self.threshold_ks or psi > self.threshold_psi
            
            results['feature_drifts'][feature] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(p_value),
                'psi': float(psi),
                'drift': drift
            }
            
            if drift:
                results['drifted_features'].append(feature)
        
        # Overall metrics
        results['overall_psi'] = float(np.mean(psi_values))
        results['drift_detected'] = len(results['drifted_features']) > 0
        
        logger.info(f"Drift check: {len(results['drifted_features'])}/{len(self.feature_names)} features drifted")
        
        return results