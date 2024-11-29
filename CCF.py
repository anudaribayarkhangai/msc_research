import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the datasets and transpose to get dates as index
df_original = pd.read_csv('D:/ITC/Thesis/Scripts/results/normalized_deforest_2022_2023_v2.csv', header=0).T
df_shift = pd.read_csv('D:/ITC/Thesis/Scripts/results/results/renormalize/normalized_timeshift.csv', header=0).T
df_swap = pd.read_csv('D:/ITC/Thesis/Scripts/results/results/renormalize/normalized_timeswap.csv', header=0).T
df_combined = pd.read_csv('D:/ITC/Thesis/Scripts/results/results/renormalize/normalized_timeshift_swap.csv', header=0).T

# Rename columns to be more descriptive for later processing
df_original.columns = ['Sample' + str(i) for i in range(df_original.shape[1])]
df_shift.columns = ['Sample' + str(i) for i in range(df_shift.shape[1])]
df_swap.columns = ['Sample' + str(i) for i in range(df_swap.shape[1])]
df_combined.columns = ['Sample' + str(i) for i in range(df_combined.shape[1])]

# Step 2: Average the samples to get a single time series for each dataset
original_series = df_original.mean(axis=1)
shift_series = df_shift.mean(axis=1)
swap_series = df_swap.mean(axis=1)
combined_series = df_combined.mean(axis=1)

# Step 3: Standardize each time series
original_series_std = (original_series - original_series.mean()) / original_series.std()
shift_series_std = (shift_series - shift_series.mean()) / shift_series.std()
swap_series_std = (swap_series - swap_series.mean()) / swap_series.std()
combined_series_std = (combined_series - combined_series.mean()) / combined_series.std()

# Step 4: Define the range of lags (e.g., -12 to +12 months)
lag_range = range(-24, 25)

# Step 5: Calculate cross-correlations at each lag for each DA method with the original
cross_corr_shift = [original_series_std.corr(shift_series_std.shift(lag)) for lag in lag_range]
cross_corr_swap = [original_series_std.corr(swap_series_std.shift(lag)) for lag in lag_range]
cross_corr_combined = [original_series_std.corr(combined_series_std.shift(lag)) for lag in lag_range]

# Step 6: Plot the Cross-Correlation Results
plt.figure(figsize=(12, 6))
plt.plot(lag_range, cross_corr_shift, marker='o', label='Temporal shift')
plt.plot(lag_range, cross_corr_swap, marker='o', label='Temporal swap')
plt.plot(lag_range, cross_corr_combined, marker='o', label='Combined')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Lag (biweeks)')
plt.ylabel('Cross-Correlation with Reference')
plt.title('Cross-Correlation with Reference Deforestation Data at Different Lags')
plt.legend()
plt.show()
