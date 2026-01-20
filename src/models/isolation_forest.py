from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def create_isolation_forest(contamination=0.001, n_estimators=100, random_state=42):
    """Factory function for Isolation Forest pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        ))
    ])

def get_anomaly_scores(model, X):
    """Extract anomaly scores (higher = more anomalous)."""
    return -model.decision_function(X)