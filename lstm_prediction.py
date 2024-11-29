import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model_path = 'D:/ITC/Thesis/Scripts/msc_thesis/models/basic_lstm_ref_norm_bins.keras'
model = load_model(model_path)

# Sliding window function for a single dataframe
def sliding_window(df, window_size, step_size):
    data = df.values
    dates = df.columns
    num_samples, num_timestamps = data.shape

    num_windows = (num_timestamps - window_size) // step_size
    windows = []
    window_dates = []

    for i in range(num_samples):
        for start_idx in range(0, num_timestamps - window_size, step_size):
            if end_idx = start_idx + window_size:
               window = data[i, start_idx:start_idx + window_size]
               windows.append(window)
               window_dates.append(dates[end_idx])

    windows = np.array(windows)
    window_dates = np.array(window_dates)
    return windows

# Parameters
window_size = 24
step_size = 1

# Apply the sliding window to each dataframe and stack the results
test_windows = []
test_targets = []

for df in normalized_dfs_2021_2023:
    windows, _ = sliding_window(df, window_size, step_size)
    test_windows.append(windows)
    test_dates.append(window_dates)
    
# Stack along the feature dimension (axis 1)
test_windows = np.stack(test_windows, axis=1)
test_dates = np.stack(test_dates, axis=1)

# The shape of the resulting array
print(test_windows.shape)
print(test_dates.shape)


# Transpose to match the input shape
test_windows = np.transpose(test_windows, (0, 2, 1))

# Prepare input (X_test) and target (y_test) variables for test data
X_test = test_windows[:, :, :]  # Input sequences

# Flatten the DataFrame
flattened_array = normalized_deforest_2022_2023.values.flatten()

print(flattened_array)

y_test = flattened_array

print("Test shape:", X_test.shape, y_test.shape)

# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Predict on the test set
y_pred = model.predict(X_test)

# Plot the actual vs predicted values for a subset
plt.figure(figsize=(10, 5))
plt.plot(y_test[:], label='Actual Values')
plt.plot(y_pred[:], label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Store predictions with their corresponding dates in a DataFrame
predictions = pd.DataFrame({
    'Date': dates_test,
    'Prediction': y_pred.flatten()
})

# Convert predictions to a NumPy array and reshape
predictions_array = predictions['Prediction'].values

# Reshape the array
num_windows = predictions.shape[0] // 48 
prediction_reshaped = predictions_array.reshape(num_windows, 48)

# Create a DataFrame with reshaped predictions and use the appropriate dates as column names
pred_dates = predictions['Date'].values[:48]
pred_date = pd.DataFrame(prediction_reshaped, columns=pred_dates)

# Display the reshaped DataFrame
print(pred_date.head())

# Save the DataFrame to a CSV file
csv_file_path = 'D:/ITC/Thesis/Scripts/results/results/basic_lstm_norm/prediction.csv'
pred_date.to_csv(csv_file_path, index=False)
print(f'Predicted output saved to {csv_file_path}')

# Concatenate the predicted values and the geometry DataFrame
df_combined = pd.concat([pred_date, geometry_df.reset_index(drop=True)], axis=1)

# Ensure the combined DataFrame is a GeoDataFrame
gdf_combined = gpd.GeoDataFrame(df_combined, geometry=geometry_df.geometry)

# Save the GeoDataFrame to a shapefile
output_shapefile = 'D:/ITC/Thesis/Scripts/results/results/basic_lstm_norm/prediction.shp'
gdf_combined.to_file(output_shapefile)

print(f"Predicted values with geometry saved to '{output_shapefile}'.")

# Example: Print the first few rows of the combined DataFrame
print(gdf_combined.head())