import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import mlflow
from mlflow.tracking import MlflowClient

from config.mlflow_config import setup_mlflow


def main():
    setup_mlflow()
    client = MlflowClient()
    
    # Search all runs from the same mlflow/experiment_name in configs/.yaml
    experiment_name = "fraud_detection_baseline" 
    
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["metrics.pr_auc DESC"],
            max_results=10
        )
    except Exception as e:
        print(f"Error searching runs: {e}")
        print("Make sure you've run train.py first!")
        return
    
    if runs.empty:
        print(f"No runs found in experiment '{experiment_name}'")
        print("Run train.py first!")
        return
    
    print(f"\nFound {len(runs)} runs in '{experiment_name}'\n")
    
    # Display top runs
    display_cols = ["run_id", "metrics.pr_auc", "metrics.fraud_detection_rate", "metrics.training_time_seconds"]
    available_cols = [col for col in display_cols if col in runs.columns]
    
    if "tags.model_type" in runs.columns:
        available_cols.insert(1, "tags.model_type")
    
    print("Top 5 runs by PR-AUC:")
    print(runs[available_cols].head())
    
    # Get best run
    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    pr_auc = best_run.get("metrics.pr_auc", None)
    model_type = best_run.get("tags.model_type", "unknown")
    
    print(f"\nBest Model:")
    print(f"   Type: {model_type}")
    print(f"   Run ID: {run_id}")
    print(f"   PR-AUC: {pr_auc:.4f}" if pr_auc else "   PR-AUC: N/A")
    
    # Check if model is already registered (via registered_model_name in train.py)
    model_name = "fraud-detector"
    versions = client.search_model_versions(f"run_id='{run_id}'")
    
    if versions:
        print(f"\nModel already registered as '{model_name}' version {versions[0].version}")
        mv = versions[0]
    else:
        print(f"\nModel not auto-registered. Registering now...")
        model_uri = f"runs:/{run_id}/model"
        try:
            mv = mlflow.register_model(model_uri, model_name)
            print(f"Registered as '{model_name}' version {mv.version}")
        except Exception as e:
            print(f"Registration failed: {e}")
            return
    
    # Promote to Production
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True  # Archive old production versions
        )
        print(f"Version {mv.version} promoted to Production (old versions archived)")
    except Exception as e:
        print(f"Could not promote to Production: {e}")
    
    # Add description
    description = f"{model_type} model - PR-AUC: {pr_auc:.4f}" if pr_auc else f"{model_type} model"
    try:
        client.update_model_version(
            name=model_name,
            version=mv.version,
            description=description
        )
        print(f"Updated model description")
    except Exception as e:
        print(f"Could not update description: {e}")
    
    print(f"\n{'='*60}")
    print(f"Model '{model_name}' version {mv.version} is now in Production!")
    print(f"Load it with: mlflow.pyfunc.load_model('models:/fraud-detector/Production')")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()