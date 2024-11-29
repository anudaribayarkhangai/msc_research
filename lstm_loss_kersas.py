import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

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

# Define the FocalR loss function in TensorFlow/Keras
class FocalRLoss(tf.keras.losses.Loss):
    def __init__(self, beta, gamma, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='FocalRLoss'):
        super().__init__(reduction=reduction, name=name)
        self.beta = beta
        self.gamma = gamma

    def call(self, y_true, y_pred):
        l1_loss = tf.abs(self.beta * tf.keras.losses.mean_absolute_error(y_true, y_pred))
        focal_loss = tf.pow(tf.sigmoid(l1_loss), self.gamma) * l1_loss
        return focal_loss

# Instantiate the custom loss function
focal_r_loss = FocalRLoss(beta=1.0, gamma=2.0)

model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Input layer
model.add(LSTM(units=150))  # LSTM layer
model.add(Dense(1))  # Dense layer for regression

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=focal_r_loss, metrics=['mae'])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=150, callbacks=[early_stop])

# Evaluate the model
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot MAE
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.save('D:/ITC/Thesis/Scripts/results/results/models/lstm_TK_loss_new_norm.keras')
