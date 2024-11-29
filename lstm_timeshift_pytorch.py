import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import pickle

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

# +
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

# +
# Flatten the DataFrame
flattened_array = normalized_deforest_2018_2021.values.flatten()

print(flattened_array.shape)

y = flattened_array

# +
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)

# +
# Define the bins explicitly for y values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 5 bins
bin_indices = np.digitize(y_train, bins[:-1]) - 1  # Flatten the array and exclude the last edge to ensure 5 bins

# Print bin counts before sampling
bin_counts = np.bincount(bin_indices)
print("Bin counts before sampling:", bin_counts)
# -

# Find the maximum bin size
max_bin_size = max(bin_counts)
print(max_bin_size)


# Function to augment data by shifting windows
def augment_data(X_train, y_train, bin_indices, max_bin_size):
    augmented_X = []
    augmented_y = []
    
    for bin_idx in range(len(np.bincount(bin_indices))):
        bin_mask = bin_indices == bin_idx
        bin_X = X_train[bin_mask]
        bin_y = y_train[bin_mask]
        
        num_samples_needed = max_bin_size - len(bin_X)
        
        if num_samples_needed > 0:
            sampled_indices = np.random.choice(len(bin_X), num_samples_needed, replace=True)
            sampled_X = bin_X[sampled_indices]
            sampled_y = bin_y[sampled_indices]
            
            # Shift windows forward and backward
            shifted_X_forward = np.roll(sampled_X, shift=1, axis=1)
            shifted_X_backward = np.roll(sampled_X, shift=-1, axis=1)
            
            augmented_X.append(np.concatenate([bin_X, sampled_X, shifted_X_forward, shifted_X_backward], axis=0)[:max_bin_size])
            augmented_y.append(np.concatenate([bin_y, sampled_y, sampled_y, sampled_y], axis=0)[:max_bin_size])
        else:
            augmented_X.append(bin_X)
            augmented_y.append(bin_y)
    
    augmented_X = np.concatenate(augmented_X, axis=0)
    augmented_y = np.concatenate(augmented_y, axis=0)
    
    return augmented_X, augmented_y


# +
# Augment the data
X_train_augmented, y_train_augmented = augment_data(X_train, y_train, bin_indices, max_bin_size)

print("Augmented training shape:", X_train_augmented.shape, y_train_augmented.shape)

# +
# Re-bin the augmented data
bin_indices_augmented = np.digitize(y_train_augmented, bins[:-1]) - 1

# Print bin counts after augmentation
bin_counts = np.bincount(bin_indices_augmented)
print("Bin counts after augmentation:", bin_counts)
# -

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_augmented, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_augmented, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader with reduced batch size
batch_size = 300  # Reduced batch size
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
        loss = torch.pow(torch.sigmoid(torch.abs(self.beta * F.l1_loss(inputs, targets, reduction=self.reduction))), self.gamma) * F.l1_loss(inputs, targets, reduction=self.reduction)
        return loss

# Instantiate the loss function and optimizer
criterion = FocalR(beta=1.0, gamma=2.0)
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 300
early_stop_patience = 10
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
# -

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

torch.save(model.state_dict(), 'D:/ITC/Thesis/Scripts/results/results/models/lstm_tshift_pytorch.pth')
