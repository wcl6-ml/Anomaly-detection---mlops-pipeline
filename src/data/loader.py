import pandas as pd
from pathlib import Path

def load_raw_data(filepath: Path):
    """
    Load raw credit card fraud dataset.
    
    Args:
        filepath: Path to raw CSV file
        
    Returns:
        DataFrame with sorted data by Time
    """
    df = pd.read_csv(filepath)
    
    # Sort by time 
    df = df.sort_values('Time').reset_index(drop=True)
    
    return df

def load_processed_data(split: str = "reference"):
    """
    Load processed split.
    
    Args:
        split: One of 'reference', 'validation', or 'batch_XXX'
    """
    from src.utils.config import PROCESSED_DATA_DIR, BATCH_DIR
    
    if split.startswith("batch_"):
        filepath = BATCH_DIR / f"{split}.csv"
    else:
        filepath = PROCESSED_DATA_DIR / f"{split}.csv"
    
    return pd.read_csv(filepath)

# test code - wcl
