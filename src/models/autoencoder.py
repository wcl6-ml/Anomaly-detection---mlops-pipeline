import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
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
def train_autoencoder(X, encoding_dim=14, epochs=50, lr=0.001, random_state=42):
    """Train autoencoder and return model + scaler."""
    torch.manual_seed(random_state)
    
    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Model setup
    model = Autoencoder(X.shape[1], encoding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    return model, scaler


def get_anomaly_scores(model, scaler, X):
    """Get reconstruction error scores."""
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
    
    return mse.numpy()  # Higher = more anomalous