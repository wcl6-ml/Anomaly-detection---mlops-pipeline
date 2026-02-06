"""Batch processing simulation with drift detection."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import requests
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from src.drift.detector import DriftDetector
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for drift
registry = CollectorRegistry()
drift_detected_gauge = Gauge('batch_drift_detected', 'Whether drift was detected', 
                            ['batch_id'], registry=registry)
overall_psi_gauge = Gauge('batch_overall_psi', 'Overall PSI score', 
                         ['batch_id'], registry=registry)
drifted_features_gauge = Gauge('batch_drifted_features_count', 'Number of drifted features',
                              ['batch_id'], registry=registry)

# API endpoint
API_URL = "http://localhost:8000/predict"
PROMETHEUS_GATEWAY = "localhost:9090"  # If you use pushgateway

def process_batch(batch_path: Path, batch_id: str, drift_detector: DriftDetector):
    """Process a single batch with drift detection and prediction."""
    logger.info(f"Processing {batch_id}...")
    
    # Load batch
    batch_df = pd.read_csv(batch_path)
    
    # Remove Time, Class, Amount for drift detection (keep only V1-V28)
    drift_columns = [col for col in batch_df.columns 
                    if col not in ['Time', 'Class', 'Amount']]
    drift_df = batch_df[drift_columns]
    
    # Drift detection (on V1-V28 only)
    drift_results = drift_detector.detect_drift(drift_df)
    
    # Log drift metrics
    drift_detected_gauge.labels(batch_id=batch_id).set(int(drift_results['drift_detected']))
    overall_psi_gauge.labels(batch_id=batch_id).set(drift_results['overall_psi'])
    drifted_features_gauge.labels(batch_id=batch_id).set(len(drift_results['drifted_features']))
    
    # Make predictions via API (V1-V28 + Amount, no Time/Class)
    api_columns = [col for col in batch_df.columns if col not in ['Time', 'Class']]
    # Use .astype(float) and .tolist() ensures native Python floats
    features = batch_df[api_columns].astype(float).values.tolist()
    
    # send to API
    response = requests.post(
        API_URL,
        json={"features": features},
        timeout=30
    )
    
    if response.status_code == 200:
        predictions = response.json()
        anomaly_rate = sum(predictions['predictions']) / len(predictions['predictions'])
        
        logger.info(f"{batch_id} - Anomaly rate: {anomaly_rate:.2%}, "
                   f"Drift: {drift_results['drift_detected']}, "
                   f"PSI: {drift_results['overall_psi']:.4f}, "
                   f"Drifted features: {len(drift_results['drifted_features'])}")
        
        return {
            'batch_id': batch_id,
            'anomaly_rate': anomaly_rate,
            'drift_detected': drift_results['drift_detected'],
            'overall_psi': drift_results['overall_psi'],
            'drifted_features': drift_results['drifted_features'],
            'inference_time_ms': predictions['inference_time_ms']
        }
    else:
        logger.error(f"API error: {response.status_code}")
        return None


def main():
    """Run batch processing simulation."""
    # Load reference data
    reference_path = project_root / "data/processed/reference.csv"
    reference_df = pd.read_csv(reference_path)
    
    # Remove Time, Class, and Amount columns - keep only V1-V28
    feature_columns = [col for col in reference_df.columns 
                      if col not in ['Time', 'Class', 'Amount']]
    reference_df = reference_df[feature_columns]
    
    logger.info(f"Loaded reference data: {reference_df.shape}")
    logger.info(f"Features: {reference_df.columns.tolist()[:5]}... (total {len(feature_columns)})")
    
    # Initialize drift detector
    drift_detector = DriftDetector(reference_df, threshold_ks=0.05, threshold_psi=0.28)
    
    # Get all batches
    batch_dir = project_root / "data/processed/batches"
    batch_files = sorted(batch_dir.glob("batch_*.csv"))
    
    logger.info(f"Found {len(batch_files)} batches to process")
    
    results = []
    
    for batch_file in batch_files:
        batch_id = batch_file.stem
        
        result = process_batch(batch_file, batch_id, drift_detector)
        if result:
            results.append(result)
        
        time.sleep(1)  # Simulate time between batches
    
    # Save summary
    summary_path = project_root / "data/processed/batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Batch processing complete. Summary saved to {summary_path}")
    
    # Print summary statistics
    total_batches = len(results)
    drifted_batches = sum(1 for r in results if r['drift_detected'])
    avg_psi = sum(r['overall_psi'] for r in results) / total_batches
    
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total batches: {total_batches}")
    print(f"Batches with drift: {drifted_batches} ({drifted_batches/total_batches:.1%})")
    print(f"Average PSI: {avg_psi:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()