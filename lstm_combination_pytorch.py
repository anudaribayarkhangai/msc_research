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

### Sliding window function for a single dataframe
def sliding_window(df, window_size, step_size):
    data = df.values
    dates = df.columns
    num_samples, num_timestamps = data.shape

    windows = []
    targets = []
    window_dates = []

    for i in range(num_samples):
        for start_idx in range(0, num_timestamps - window_size, step_size):
            end_idx = start_idx + window_size
            if end_idx < num_timestamps:
                window = data[i, start_idx:end_idx]
                target = data[i, end_idx]  # Target is the next timestamp value
                windows.append(window)
                targets.append(target)
                window_dates.append(dates[end_idx])  # Use the date of the target value

    windows = np.array(windows)
    targets = np.array(targets)
    window_dates = np.array(window_dates)
    return windows, targets, window_dates


# Parameters
window_size = 24
step_size = 1

# Apply the sliding window to each dataframe and stack the results
all_windows = []
all_targets = []
all_dates = []

for df in normalized_dfs_2017_2021:
    windows, _, window_dates = sliding_window(df, window_size, step_size)
    all_windows.append(windows)
    all_dates.append(window_dates)

# Stack along the feature dimension (axis 1)
all_windows = np.stack(all_windows, axis=1)
all_dates = np.stack(all_dates, axis=1)

# The shape of the resulting array
print(all_windows.shape)  
print(all_dates.shape)

# Transpose the array to the correct shape
all_windows = np.transpose(all_windows, (0, 2, 1))
print(all_windows.shape)

# Split the data ino 80% training and 20% validation
X = all_windows[:, :, :]

# Ensure normalized_deforest_2017_2021 is a DataFrame
if isinstance(normalized_deforest_2017_2021, pd.DataFrame):
    _, targets, _ = sliding_window(normalized_deforest_2017_2021, window_size, step_size)
    all_targets.append(targets)
    all_targets = np.concatenate(all_targets, axis=0)
    
print(all_targets.shape)

y = all_targets

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


# Function to swap random time segments (4-12 columns representing 2-6 months) within corresponding periods in different years
def random_time_swap_biweeks(X_train, num_months=6):
    swapped_X = []
    num_samples, seq_length, num_features = X_train.shape
    
    # Each month is represented by 2 biweek columns
    biweeks_per_month = 2
    
    # Calculate the number of years in the dataset
    num_years = seq_length // (12 * biweeks_per_month)
    
    for i in range(num_samples):
        x_sample = X_train[i].copy()  # Copy the sample

        # Define the swap window length in biweeks (4-12 columns for 2-6 months)
        swap_length = np.random.randint(2, num_months + 1) * biweeks_per_month
        
        # Ensure that we can find two segments within the total sequence length
        max_start_idx = (12 * biweeks_per_month) - swap_length
        
        # Select a random starting point within a year for the swap
        start_idx_within_year = np.random.randint(0, max_start_idx + 1)
        
        # Perform the swap between corresponding periods in different years
        for year in range(num_years - 1):
            first_start_idx = year * 12 * biweeks_per_month + start_idx_within_year
            second_start_idx = (year + 1) * 12 * biweeks_per_month + start_idx_within_year
            
            # Perform the swap
            first_segment = x_sample[first_start_idx:first_start_idx + swap_length].copy()
            second_segment = x_sample[second_start_idx:second_start_idx + swap_length].copy()
            
            # Swap the segments
            x_sample[first_start_idx:first_start_idx + swap_length] = second_segment
            x_sample[second_start_idx:second_start_idx + swap_length] = first_segment
        
        swapped_X.append(x_sample)
    
    swapped_X = np.array(swapped_X)
    return swapped_X

# Modify augment_data function to first shift windows and then use random time swap
def augment_data_with_shift_and_time_swap(X_train, y_train, bin_indices, max_bin_size, num_months=6):
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
            
            # Apply random time swap for biweeks
            swapped_X = random_time_swap_biweeks(sampled_X, num_months=num_months)
            
            augmented_X.append(np.concatenate([bin_X, sampled_X, shifted_X_forward, shifted_X_backward, swapped_X], axis=0)[:max_bin_size])
            augmented_y.append(np.concatenate([bin_y, sampled_y, sampled_y, sampled_y, sampled_y], axis=0)[:max_bin_size])
        else:
            augmented_X.append(bin_X)
            augmented_y.append(bin_y)
    
    augmented_X = np.concatenate(augmented_X, axis=0)
    augmented_y = np.concatenate(augmented_y, axis=0)
    
    return augmented_X, augmented_y

# Apply data augmentation
X_train_augmented, y_train_augmented = augment_data_with_shift_and_time_swap(X_train, y_train, bin_indices, max_bin_size, num_months=6)

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

class FocalR(nn.Module):
    def __init__(self, beta, gamma, reduction='mean'):
        super(FocalR, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = torch.pow(F.sigmoid(torch.abs(self.beta*F.l1_loss(inputs, targets, reduction = self.reduction))), self.gamma) * F.l1_loss(inputs, targets, reduction = self.reduction)
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

torch.save(model.state_dict(), 'D:/ITC/Thesis/Scripts/results/models/lstm_loss_com.pth')