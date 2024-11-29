import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# Sliding window function for a single dataframe
def sliding_window(df, window_size, step_size):
    data = df.values
    num_samples, num_timestamps = data.shape

    num_windows = (num_timestamps - window_size) // step_size
    windows = []

    for i in range(num_samples):
        for start_idx in range(0, num_timestamps - window_size, step_size):
            window = data[i, start_idx:start_idx + window_size]
            windows.append(window)

    windows = np.array(windows)
    return windows

# Parameters
window_size = 24
step_size = 1

# Applying the sliding window to each dataframe
all_windows = []

for df in normalized_dfs_2017_2021:
    windows = sliding_window(df, window_size, step_size)
    all_windows.append(windows)

# Stack along the feature dimension
all_windows = np.stack(all_windows, axis=1)

# The shape of the resulting array
print(all_windows.shape)  # (sample, features, window_size)

# Transpose the array to the correct shape
all_windows = np.transpose(all_windows, (0, 2, 1))
print(all_windows.shape)  # Should be (sample, window_size, features)

# Split the data ino 80% training and 20% validation
X = all_windows[:, :, :]  # Input sequences (all except the last timestamp)

# Flatten the DataFrame
flattened_array = normalized_deforest_2018_2021.values.flatten()

print(flattened_array.shape)

y = flattened_array

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader with reduced batch size
batch_size = 32  # Reduced batch size
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the PyTorch model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out

model = LSTMModel(input_size=X_train.shape[2], hidden_size=128, output_size=1)

# Define the custom FocalR loss function
class FocalR(nn.Module):
    def __init__(self, beta, gamma, reduction='mean'):
        super(FocalR, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = torch.pow(F.sigmoid(torch.abs(self.beta * F.l1_loss(inputs, targets, reduction=self.reduction))), self.gamma) * F.l1_loss(inputs, targets, reduction=self.reduction)
        return loss

# Instantiate the loss function and optimizer
criterion = FocalR(beta=1.0, gamma=2.0)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 300
early_stop_patience = 20
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the model
model.eval()
val_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        val_loss += loss.item()

val_loss /= len(val_loader)
print(f'Validation Loss: {val_loss}')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'D:/ITC/Thesis/Scripts/results/models/lstm_loss_pytorch.pth')
