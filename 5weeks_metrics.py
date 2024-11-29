import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Read CSV files
csv_file_path = #prediction csv
pred_date = pd.read_csv(csv_file_path)

csv_file_path2 = #reference csv
true_date = pd.read_csv(csv_file_path2)

# Selected weeks for analysis
selected_weeks = ['0102', '1603', '0104', '1606', '0109', '1610', '0111']

def extract_full_date_id(column_name):
    # Assuming the date identifier is always at the end of the column name
    return column_name[-6:]

# Identify common columns
common_columns = [col for col in true_date.columns if col in pred_date.columns]

# Select columns based on the selected weeks
columns_to_select = [col for col in common_columns if any(week in col for week in selected_weeks)]
true_selected = true_date[columns_to_select]
pred_selected = pred_date[columns_to_select]

print(true_selected.shape)
print(pred_selected.shape)

# Initialize dictionaries for metrics
metrics = {'Date': [], 'RMSE': [], 'R2': [], 'MBA': [], 'MSE': [], 'MBD': [], 'MAE': []}

# Initialize lists for combined R² calculation
all_true_values = []
all_pred_values = []

# Calculate metrics for each selected week
for col in columns_to_select:
    true_values = true_selected[col].values
    pred_values = pred_selected[col].values
    
    # Append to combined lists
    all_true_values.extend(true_values)
    all_pred_values.extend(pred_values)
    
    # Extract full date identifier from column name
    date_id = extract_full_date_id(col)
    
    # Store date identifier and calculate/store metrics
    metrics['Date'].append(date_id)
    metrics['RMSE'].append(np.sqrt(mean_squared_error(true_values, pred_values)))
    metrics['R2'].append(r2_score(true_values, pred_values))
    metrics['MBA'].append(np.mean(pred_values) / np.mean(true_values) - 1)
    metrics['MSE'].append(mean_squared_error(true_values, pred_values))
    metrics['MBD'].append(np.mean(pred_values - true_values))
    metrics['MAE'].append(mean_absolute_error(true_values, pred_values))

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame(metrics)

# Save metrics DataFrame to CSV
metrics_df.to_csv(#result csv, index=False)

# Plot R² for each week with detailed precision and add a trend line
for i, col in enumerate(columns_to_select):
    plt.figure()
    actual = true_selected[col].values
    predicted = pred_selected[col].values
    
    # Scatter plot of actual vs predicted
    plt.scatter(actual, predicted, label='Actual vs Predicted')
    
    # Calculate coefficients for the trend line
    z = np.polyfit(actual, predicted, 1)
    p = np.poly1d(z)
    
    # Plotting the trend line
    plt.plot(actual, p(actual), "r--", label='Trend Line')
    
    # Plot details
    plt.title(f'R² for {col}: {metrics["R2"][i]:.4f}')  # Display R² with 4 decimal places
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.show()

# Calculate combined R² for all common columns
all_true_values_combined = []
all_pred_values_combined = []

for col in common_columns:
    true_values_combined = true_date[col].values
    pred_values_combined = pred_date[col].values
    
    all_true_values_combined.extend(true_values_combined)
    all_pred_values_combined.extend(pred_values_combined)

combined_r2 = r2_score(all_true_values_combined, all_pred_values_combined)

# Plot combined R² for all common columns
plt.figure()
plt.scatter(all_true_values_combined, all_pred_values_combined, label='Actual vs Predicted')
z_combined = np.polyfit(all_true_values_combined, all_pred_values_combined, 1)
p_combined = np.poly1d(z_combined)
plt.plot(all_true_values_combined, p_combined(all_true_values_combined), "r--", label='Trend Line')
plt.title(f'Combined R² for all columns: {combined_r2:.4f}')  # Display combined R² with 4 decimal places
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()