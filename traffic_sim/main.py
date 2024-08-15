import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_model import TransformerModel, prepare_data, create_dataloaders, train_model, evaluate_model

# Load the synthetic traffic data
data_path = 'data/historical_traffic_data.csv'
look_back = 60
n_heads = 4

# Prepare the data
scaled_data, input_dim = prepare_data(data_path, look_back, n_heads)
train_loader, test_loader = create_dataloaders(scaled_data, look_back)

# Initialize and train the Transformer model
model = TransformerModel(input_dim=input_dim, hidden_dim=64, output_dim=1, n_layers=3, n_heads=n_heads, dropout=0.1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=20)

# Evaluate the model
predicted_traffic, actual_traffic = evaluate_model(model, test_loader)

# Example analysis: Compare predicted vs. actual traffic
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actual_traffic, label='Actual Traffic')
plt.plot(predicted_traffic, label='Predicted Traffic')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.title('Actual vs Predicted Traffic Volume')
plt.legend()
plt.show()

