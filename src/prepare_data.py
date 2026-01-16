from data.loader import load_raw_data
from data.splitter import create_time_splits, save_splits
from utils.config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR,
    REFERENCE_RATIO,
    VALIDATION_RATIO,
    NUM_BATCHES
)

def main():
    print("=" * 60)
    print("Credit Card Fraud - Data Preparation Pipeline")
    print("=" * 60)
    
    # Load raw data
    print("\nLoading raw data...")
    raw_file = RAW_DATA_DIR / "creditcard.csv"
    df = load_raw_data(raw_file)
    print(f"Loaded {len(df)} samples")
    print(f"  - Fraud ratio: {df['Class'].mean():.4f}")
    print(f"  - Time range: {df['Time'].min():.0f} - {df['Time'].max():.0f}")
    
    # Create splits
    print(f"Creating time-based splits...")
    print(f"  - Reference: {REFERENCE_RATIO*100:.0f}%")
    print(f"  - Validation: {VALIDATION_RATIO*100:.0f}%")
    print(f"  - Production batches: {NUM_BATCHES}")
    
    reference, validation, batches = create_time_splits(
        df,
        REFERENCE_RATIO,
        VALIDATION_RATIO,
        NUM_BATCHES
    )
    
    # Save splits
    print(f"Saving processed data...")
    save_splits(reference, validation, batches, PROCESSED_DATA_DIR)
    

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()