import numpy as np
import time
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_model(model, X, y, get_scores_fn):
    """
    Evaluate isolation forest on test data.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels (1 = fraud, 0 = normal)
        get_scores_fn: Function to extract anomaly scores from model
    
    Returns:
        dict of metrics
    """
    # Anomaly scores
    start_time = time.time()
    scores = get_scores_fn(model, X)
    inference_time = (time.time() - start_time) / len(X) * 1000  # ms per sample
    
    # Metrics
    roc_auc = roc_auc_score(y, scores)
    
    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall, precision)
    
    # Anomaly rate at 99th percentile threshold
    threshold = np.percentile(scores, 99)
    predictions = (scores > threshold).astype(int)
    anomaly_rate = predictions.mean()
    
    # Detected fraud rate
    detected_frauds = (predictions & y).sum()
    total_frauds = y.sum()
    fraud_detection_rate = detected_frauds / total_frauds if total_frauds > 0 else 0
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "anomaly_rate": anomaly_rate,
        "fraud_detection_rate": fraud_detection_rate,
        "inference_time_ms": inference_time
    }