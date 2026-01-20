import torch
import torch.nn as nn
import numpy as np

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim, encoding_dim=14):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_anomaly_scores(model, X):
    """Extract reconstruction error scores."""
    return -model.decision_function(X)