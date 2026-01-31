import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.mlflow_config import setup_mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

setup_mlflow()

# Load model
model = mlflow.pyfunc.load_model("models:/fraud-detector/Production")

print("Testing different feature counts...")

for n_features in [28, 29, 30, 31]:
    try:
        test_data = pd.DataFrame(np.random.randn(1, n_features))
        predictions = model.predict(test_data)
        print(f"{n_features} features: WORKS")
        print(f"   Prediction shape: {predictions.shape}")
        break
    except Exception as e:
        print(f"{n_features} features: FAILED - {e}")

print(f"\nModel expects {n_features} features")