import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Read CSV files
csv_file_path = #prediction csv
pred_date = pd.read_csv(csv_file_path)

csv_file_path2 = #reference csv
true_date = pd.read_csv(csv_file_path2)

# Identify common columns
common_columns = [col for col in true_date.columns if col in pred_date.columns]

# Select columns based on the common columns
true_selected = true_date[common_columns]
pred_selected = pred_date[common_columns]

print(true_selected.shape)
print(pred_selected.shape)

# Initialize lists for combined R² calculation
all_true_values = []
all_pred_values = []

# Define bins
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

# Aggregate all true and predicted values
for col in common_columns:
    true_values = true_selected[col].values
    pred_values = pred_selected[col].values
    all_true_values.extend(true_values)
    all_pred_values.extend(pred_values)

# Convert to numpy arrays for easier indexing
all_true_values = np.array(all_true_values)
all_pred_values = np.array(all_pred_values)

# Initialize dictionaries for metrics
metrics = {'Bin': [], 'RMSE': [], 'R2': [], 'MBA': [], 'MSE': [], 'MBD': [], 'MAE': [], 'BinAccuracy': []}

# Calculate metrics for each bin
for i in range(len(bins) - 1):
    if i == len(bins) - 2:
        bin_mask = (all_true_values >= bins[i]) & (all_true_values <= bins[i + 1])
    else:
        bin_mask = (all_true_values >= bins[i]) & (all_true_values < bins[i + 1])
    
    bin_true_values = all_true_values[bin_mask]
    bin_pred_values = all_pred_values[bin_mask]
    
    if len(bin_true_values) > 0:
        metrics['Bin'].append(f'{bins[i]}-{bins[i+1]}')
        metrics['RMSE'].append(np.sqrt(mean_squared_error(bin_true_values, bin_pred_values)))
        metrics['R2'].append(r2_score(bin_true_values, bin_pred_values))
        metrics['MBA'].append(np.mean(bin_pred_values) / np.mean(bin_true_values) - 1)
        metrics['MSE'].append(mean_squared_error(bin_true_values, bin_pred_values))
        metrics['MBD'].append(np.mean(bin_pred_values - bin_true_values))
        metrics['MAE'].append(mean_absolute_error(bin_true_values, bin_pred_values))
        bin_accuracy = np.mean((bin_pred_values >= bins[i]) & (bin_pred_values < bins[i + 1]))
        metrics['BinAccuracy'].append(bin_accuracy)

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame(metrics)

# Save metrics DataFrame to CSV
metrics_df.to_csv(#results, index=False)

# Plot true vs predicted values for each bin with trend lines
for i in range(len(bins) - 1):
    if i == len(bins) - 2:
        bin_mask = (all_true_values >= bins[i]) & (all_true_values <= bins[i + 1])
    else:
        bin_mask = (all_true_values >= bins[i]) & (all_true_values < bins[i + 1])
    
    bin_true_values = all_true_values[bin_mask]
    bin_pred_values = all_pred_values[bin_mask]
    
    if len(bin_true_values) > 0:
        plt.figure()
        plt.scatter(bin_true_values, bin_pred_values, label=f'Actual vs Predicted for Bin {bins[i]}-{bins[i+1]}')
        z_bin = np.polyfit(bin_true_values, bin_pred_values, 1)
        p_bin = np.poly1d(z_bin)
        plt.plot(bin_true_values, p_bin(bin_true_values), "r--", label='Trend Line')
        plt.title(f'R² for Bin {bins[i]}-{bins[i+1]}: {r2_score(bin_true_values, bin_pred_values):.4f}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.legend()
        plt.show()