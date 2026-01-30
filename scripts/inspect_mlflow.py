import mlflow
from config.mlflow_config import setup_mlflow

setup_mlflow()

# List all experiments
experiments = mlflow.search_experiments()
print("Experiments:")
for exp in experiments:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")

# List all runs
runs = mlflow.search_runs()

if runs.empty:
    print("\nNo runs found!")
    print("Run: python train.py --config configs/isolation_forest.yaml")
else:
    print(f"\nFound {len(runs)} runs\n")
    
    # Show all columns
    print("Available columns:")
    for col in sorted(runs.columns):
        print(f"  - {col}")
    
    # Show metric columns specifically
    metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
    print(f"\nMetrics columns: {metric_cols}")
    
    # Show sample data
    display_cols = ["run_id", "experiment_id", "status", "start_time"]
    display_cols.extend(metric_cols[:5])  # First 5 metrics
    
    print("\nsample runs:")
    print(runs[display_cols].head())