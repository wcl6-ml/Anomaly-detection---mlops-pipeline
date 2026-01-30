import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import json

def create_time_splits(
    df: pd.DataFrame,
    reference_ratio: float,
    validation_ratio: float,
    num_batches: int
):
    """
    Split data by time into reference, validation, and production batches.
    
    Args:
        df: Sorted DataFrame by time
        reference_ratio: Proportion for training
        validation_ratio: Proportion for validation
        num_batches: Number of production batches to create
        
    Returns:
        (reference_df, validation_df, list_of_batch_dfs)
    """
    n = len(df)
    
    # Calculate split indices
    ref_end = int(n * reference_ratio)
    val_end = int(n * (reference_ratio + validation_ratio))
    
    # Create splits
    reference = df.iloc[:ref_end].copy()
    validation = df.iloc[ref_end:val_end].copy()
    production = df.iloc[val_end:].copy()
    
    # Split production into batches
    batch_size = len(production) // num_batches
    batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        # Last batch gets remaining data
        end_idx = (i + 1) * batch_size if i < num_batches - 1 else len(production)
        batch = production.iloc[start_idx:end_idx].copy()
        batches.append(batch)
    
    return reference, validation, batches

def save_splits(
    reference: pd.DataFrame,
    validation: pd.DataFrame,
    batches: List[pd.DataFrame],
    output_dir: Path
):
    """
    Save all splits to disk and create metadata.
    
    Args:
        reference: Reference/training data
        validation: Validation data
        batches: List of production batch DataFrames
        output_dir: Directory to save processed data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(exist_ok=True)
    
    # Save main splits
    reference.to_csv(output_dir / "reference.csv", index=False)
    validation.to_csv(output_dir / "validation.csv", index=False)
    
    # Save batches
    for i, batch in enumerate(batches, 1):
        batch_filename = f"batch_{i:03}.csv"
        batch.to_csv(batch_dir / batch_filename, index=False)
    
    # Create metadata
    metadata = {
        "reference": {
            "size": len(reference),
            "time_range": [float(reference['Time'].min()), float(reference['Time'].max())],
            "fraud_count": int(reference['Class'].sum()),
            "fraud_ratio": float(reference['Class'].mean())
        },
        "validation": {
            "size": len(validation),
            "time_range": [float(validation['Time'].min()), float(validation['Time'].max())],
            "fraud_count": int(validation['Class'].sum()),
            "fraud_ratio": float(validation['Class'].mean())
        },
        "batches": [
            {
                "batch_id": i,
                "filename": f"batch_{i:03}.csv",
                "size": len(batch),
                "time_range": [float(batch['Time'].min()), float(batch['Time'].max())],
                "fraud_count": int(batch['Class'].sum()),
                "fraud_ratio": float(batch['Class'].mean())
            }
            for i, batch in enumerate(batches, 1)
        ]
    }
    
    with open(output_dir / "splits_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved reference: {len(reference)} samples")
    print(f"Saved validation: {len(validation)} samples")
    print(f"Saved {len(batches)} batches to {batch_dir}")
    print(f"Metadata saved to splits_metadata.json")

