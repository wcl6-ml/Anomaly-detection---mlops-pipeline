import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def validate_config(config: Dict[str, Any], model_type: str) -> None:
    """
    Validate that config has required fields.
    
    Args:
        config: Configuration dictionary
        model_type: Expected model type
    """
    # Check required top-level keys
    required_keys = ['model', 'training', 'mlflow']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Check model type matches
    if config['model']['type'] != model_type:
        raise ValueError(
            f"Config model type '{config['model']['type']}' "
            f"doesn't match expected '{model_type}'"
        )
    
    print(f"✓ Config validated for {model_type}")