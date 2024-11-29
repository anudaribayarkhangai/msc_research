import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Sliding window function for a single dataframe
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
                target = data[i, end_idx]
                windows.append(window)
                targets.append(target)
                window_dates.append(dates[end_idx])

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
#all_windows = np.transpose(all_windows, (0, 2, 1))
#print(all_windows.shape)

# Ensure normalized_deforest_2017_2021 is a DataFrame
if isinstance(normalized_deforest_2017_2021, pd.DataFrame):
    _, targets, _ = sliding_window(normalized_deforest_2017_2021, window_size, step_size)
    all_targets.append(targets)
    all_targets = np.concatenate(all_targets, axis=0)
    
print(all_targets.shape)
    

# Split the data into 80% training and 20% validation
X = all_windows[:, :, :]
y = all_targets
dates = all_dates[:, 0]

X_train, X_val, y_train, y_val, dates_train, dates_val = train_test_split(X, y, dates, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)

model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=128))
model.add(Dense(1) )

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=20)

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

# Optionally, plot MAE if you wish to view that as well
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the model
model.save('D:/ITC/Thesis/Scripts/msc_thesis/models/basic_lstm.keras')
