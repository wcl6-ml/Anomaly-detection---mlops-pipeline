"""Essential tests for configuration loading."""

import pytest
import yaml
from pathlib import Path

from src.utils.config_model_loader import load_config, validate_config


class TestConfigLoader:
    """Core config loading tests."""
    
    def test_load_valid_config(self, temp_data_dir):
        """Test loading a valid YAML config."""
        config_data = {
            'model': {'type': 'isolation_forest', 'contamination': 0.1},
            'training': {'reference_data': 'data/reference.csv'},
            'mlflow': {'experiment_name': 'test'}
        }
        
        config_file = temp_data_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded = load_config(str(config_file))
        
        assert loaded['model']['type'] == 'isolation_forest'
        assert loaded['model']['contamination'] == 0.1
    
    def test_load_missing_file_raises_error(self):
        """Test that loading missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_validate_config_success(self):
        """Test config validation with correct structure."""
        config = {
            'model': {'type': 'isolation_forest'},
            'training': {'reference_data': 'data/ref.csv'},
            'mlflow': {'experiment_name': 'test'}
        }
        
        # Should not raise
        validate_config(config, 'isolation_forest')
    
    def test_validate_config_missing_section(self):
        """Test validation fails with missing required section."""
        config = {
            'model': {'type': 'isolation_forest'},
            # Missing 'training' and 'mlflow'
        }
        
        with pytest.raises(ValueError, match="Missing required config section"):
            validate_config(config, 'isolation_forest')
    
    def test_validate_config_wrong_model_type(self):
        """Test validation fails when model type doesn't match."""
        config = {
            'model': {'type': 'autoencoder'},  # Wrong type
            'training': {'reference_data': 'data/ref.csv'},
            'mlflow': {'experiment_name': 'test'}
        }
        
        with pytest.raises(ValueError, match="doesn't match expected"):
            validate_config(config, 'isolation_forest')