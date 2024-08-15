import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from transformer_model import TransformerModel, prepare_data, create_dataloaders, train_model, evaluate_model

# Load the synthetic traffic data
data_path = 'data/historical_traffic_data.csv'
look_back = 60
n_heads = 4

# Load the dataset
data = pd.read_csv(data_path)

# Convert categorical columns (road_segment and vehicle_type) to numerical values
label_encoder_segment = LabelEncoder()
label_encoder_vehicle = LabelEncoder()

data['road_segment'] = label_encoder_segment.fit_transform(data['road_segment'])
data['vehicle_type'] = label_encoder_vehicle.fit_transform(data['vehicle_type'])

# Prepare the data for the Transformer model
features = ['traffic_volume', 'average_speed', 'occupancy', 'road_segment', 'vehicle_type']
scaled_data, input_dim = prepare_data(data[features], look_back, n_heads)

# Create data loaders
train_loader, test_loader = create_dataloaders(scaled_data, look_back)

# Initialize and train the Transformer model
model = TransformerModel(input_dim=input_dim, hidden_dim=64, output_dim=1, n_layers=3, n_heads=n_heads, dropout=0.1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=20)

# Evaluate the model
predicted_traffic, actual_traffic = evaluate_model(model, test_loader)

# Example analysis: Compare predicted vs. actual traffic volume
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actual_traffic, label='Actual Traffic Volume')
plt.plot(predicted_traffic, label='Predicted Traffic Volume')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.title('Actual vs Predicted Traffic Volume')
plt.legend()
plt.show()