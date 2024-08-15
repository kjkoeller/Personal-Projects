import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, data, look_back):
        self.data = data
        self.look_back = look_back

    def __len__(self):
        return len(self.data) - self.look_back

    def __getitem__(self, index):
        x = self.data[index:index + self.look_back, :]
        y = self.data[index + self.look_back, 0]  # Assuming traffic_volume is the target
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_heads, dropout):
        super(TransformerModel, self).__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(self.dropout(x[-1]))
        return x

def prepare_data(data, look_back, n_heads):
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Ensure input_dim is divisible by n_heads by adding dummy features if necessary
    input_dim = scaled_data.shape[1]
    if input_dim % n_heads != 0:
        padding_needed = n_heads - (input_dim % n_heads)
        padding = np.zeros((scaled_data.shape[0], padding_needed))
        scaled_data = np.hstack([scaled_data, padding])
        input_dim += padding_needed

    return scaled_data, input_dim

def create_dataloaders(scaled_data, look_back):
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    train_dataset = TrafficDataset(train_data, look_back)
    test_dataset = TrafficDataset(test_data, look_back)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x.permute(1, 0, 2))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x.permute(1, 0, 2))
            predictions.append(output.numpy())
            actuals.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(actuals)