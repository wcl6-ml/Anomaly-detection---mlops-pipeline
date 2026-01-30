# src/train.py
import argparse
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import time

from utils.config_model_loader import load_config, validate_config
from models.isolation_forest import create_isolation_forest, get_anomaly_scores as get_if_scores
from models.autoencoder import train_autoencoder, get_anomaly_scores as get_ae_scores
from evaluation.evaluate import evaluate_model


def load_data(data_path: str):
    """Load data and separate features from labels."""
    df = pd.read_csv(data_path)
    
    # Drop Time and Class columns
    X = df.drop(columns=["Class", "Time"]).values
    y = df["Class"].values
    
    return X, y


def train_isolation_forest_from_config(config: dict):
    """Train Isolation Forest using config."""
    print("\n" + "="*60)
    print("Training Isolation Forest")
    print("="*60)
    
    validate_config(config, "isolation_forest")
    
    # Load data
    X_train, _ = load_data(config['training']['reference_data'])
    X_val, y_val = load_data(config['training']['validation_data'])
    
    # Start MLflow run
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=config['mlflow']['run_name']):
        # Log all model parameters
        mlflow.set_tag("run_name", config['mlflow']['run_name'])
        mlflow.log_params({
            "model_type": config['model']['type'],
            **config['model']  # Log all model config as params
        })
        
        # Create and train model
        print("\nCreating model...")
        model = create_isolation_forest(
            contamination=config['model']['contamination'],
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state']
        )
        
        print("Training...")
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.2f} seconds")
        mlflow.log_metric("training_time_seconds", train_time)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        metrics = evaluate_model(model, X_val, y_val, get_if_scores, config['model']['anomaly_threshold'])
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print("\nResults:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("-" * 40)
        
        return metrics


def train_autoencoder_from_config(config: dict):
    """Train Autoencoder using config."""
    print("\n" + "="*60)
    print("Training Autoencoder")
    print("="*60)
    
    validate_config(config, "autoencoder")
    
    # Load data
    X_train, _ = load_data(config['training']['reference_data'])
    X_val, y_val = load_data(config['training']['validation_data'])
    
    # Update input_dim from data
    config['model']['input_dim'] = X_train.shape[1]
    
    # Start MLflow run
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=config['mlflow']['run_name']):
        # Log parameters
        mlflow.set_tag("run_name", config['mlflow']['run_name'])
        mlflow.log_params({
            "model_type": config['model']['type'],
            **config['model'],
            **{k: v for k, v in config['training'].items() 
               if k not in ['reference_data', 'validation_data']}
        })
        
        # Train model
        print("\nTraining autoencoder...")
        start_time = time.time()
        model, scaler = train_autoencoder(
            X_train,
            encoding_dim=config['model']['encoding_dim'],
            epochs=config['training']['epochs'],
            lr=config['training']['learning_rate'],
            random_state=config['model']['random_state']
        )
        train_time = time.time() - start_time
        
        print(f"\nTraining completed in {train_time:.2f} seconds")
        mlflow.log_metric("training_time_seconds", train_time)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        scores = get_ae_scores(model, scaler, X_val)
        
        # Compute metrics manually for autoencoder
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as pr_auc
        
        metrics = {
            "roc_auc": roc_auc_score(y_val, scores),
            "anomaly_rate": (scores > np.percentile(scores, config['model']['anomaly_threshold'])).mean(),
        }
        
        precision, recall, _ = precision_recall_curve(y_val, scores)
        metrics["pr_auc"] = pr_auc(recall, precision)
        
        # Fraud detection rate
        threshold = np.percentile(scores, config['model']['anomaly_threshold'])
        predictions = (scores > threshold).astype(int)
        detected_frauds = (predictions & y_val).sum()
        total_frauds = y_val.sum()
        metrics["fraud_detection_rate"] = detected_frauds / total_frauds if total_frauds > 0 else 0
        
        # Inference time
        start = time.time()
        _ = get_ae_scores(model, scaler, X_val)
        metrics["inference_time_ms"] = (time.time() - start) / len(X_val) * 1000
        
        mlflow.log_metrics(metrics)
        
        # Log model - MLFLOW.PYTORCH vs MLFLOW.SKLEARN differs 
        mlflow.pytorch.log_model(model, "model")
        
        # Also save scaler
        import joblib
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        # Print results
        print("\nResults:")
        print("-" * 40)
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("-" * 40)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train fraud detection models using YAML configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/isolation_forest.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Train based on model type
    model_type = config['model']['type']
    
    if model_type == "isolation_forest":
        metrics = train_isolation_forest_from_config(config)
    elif model_type == "autoencoder":
        metrics = train_autoencoder_from_config(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"View results: mlflow ui --port 5000")
    print("="*60)


if __name__ == "__main__":
    main()