import os
import geopandas as gpd
import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import copy
import pickle


# Load the DataFrame list from the file
with open('D:/ITC/Thesis/data/new/all_shapefiles_list_new.pkl', 'rb') as file:
    all_shapefiles_list = pickle.load(file)

# Load the DataFrame list from the file
with open('D:/ITC/Thesis/Scripts/results/all_shapefiles_list.pkl', 'rb') as file:
    all_shapefiles_list = pickle.load(file)   

all_shapefiles_copy = copy.deepcopy(all_shapefiles_list)

# Lists to hold the divided DataFrames
dfs_2017_2021 = []
dfs_2018_2021 = []
dfs_2022_2023 = []
dfs_2021_2023 = []
deforest_2017_2021 = pd.DataFrame()
deforest_2018_2021 = pd.DataFrame()
deforest_2022_2023 = pd.DataFrame()
deforest_2021_2023 = pd.DataFrame()

# Function to filter columns based on year range
def filter_columns_by_year(df, start_year, end_year):
    year_pattern = re.compile(r'\d{6}$')
    filtered_columns = []
    for col in df.columns:
        match = year_pattern.search(col)
        if match:
            date_str = match.group()
            year = 2000 + int(date_str[-2:])
            if start_year <= year <= end_year:
                filtered_columns.append(col)
    return df[filtered_columns]

# Process each DataFrame in the list
for i, df in enumerate(all_shapefiles_list):
    df_2017_2021 = filter_columns_by_year(df, 2017, 2021)
    df_2018_2021 = filter_columns_by_year(df, 2018, 2021)
    df_2022_2023 = filter_columns_by_year(df, 2022, 2023)
    df_2021_2023 = filter_columns_by_year(df, 2021, 2023)
        
    if not df_2017_2021.empty:
        dfs_2017_2021.append(df_2017_2021)
        if i==10:
           deforest_2017_2021=df_2017_2021 
    
    if not df_2018_2021.empty:
        dfs_2018_2021.append(df_2018_2021)
        if i==10:
           deforest_2018_2021=df_2018_2021 
    
    if not df_2022_2023.empty:
        dfs_2022_2023.append(df_2022_2023)
        if i==10:
            deforest_2021_2023=df_2021_2023
        
    if not df_2021_2023.empty:
        dfs_2021_2023.append(df_2021_2023)
        if i==10:
            deforest_2022_2023=df_2022_2023
        
# Function to rename columns by extracting the ddmmyy part
def rename_columns(dfs_list):
    for df in dfs_list:
        df.columns = [re.search(r'\d{6}', col).group() if re.search(r'\d{6}', col) else col for col in df.columns]

# Rename columns for lists of DataFrames
rename_columns(dfs_2017_2021)
rename_columns(dfs_2018_2021)
rename_columns(dfs_2022_2023)
rename_columns(dfs_2021_2023)

# Rename columns for individual DataFrames if they are not empty
if not deforest_2017_2021.empty:
    rename_columns([deforest_2017_2021])
if not deforest_2018_2021.empty:
    rename_columns([deforest_2018_2021])
if not deforest_2022_2023.empty:
    rename_columns([deforest_2022_2023])
if not deforest_2021_2023.empty:
    rename_columns([deforest_2021_2023])
        
# Function to normalize a DataFrame
def normalize_dataframe(df):
    normalized_df = df.copy()
    # Set any value below 0 to 0
    normalized_df[normalized_df < 0] = 0
    max_val = normalized_df.max().max()
    # Normalize to ensure min is 0 and max is 1
    if max_val != 0:  # To avoid division by zero if all values are 0
        normalized_df = normalized_df / max_val
    return normalized_df


# Normalize each DataFrame in the 2017-2021 and 2022-2023 lists
normalized_dfs_2017_2021 = [normalize_dataframe(df) for df in dfs_2017_2021]
normalized_dfs_2018_2021 = [normalize_dataframe(df) for df in dfs_2018_2021]
normalized_dfs_2022_2023 = [normalize_dataframe(df) for df in dfs_2022_2023]
normalized_dfs_2021_2023 = [normalize_dataframe(df) for df in dfs_2021_2023]

# Normalization function per columns
def normalize_dataframe(df):
    normalized_df = df.copy()
    for column in normalized_df.columns:
        # Set any value below 0 to 0
        normalized_df[column] = normalized_df[column].clip(lower=0)
        max_val = normalized_df[column].max()
        # Normalize to ensure min is 0 and max is 1
        if max_val != 0:  # To avoid division by zero if all values are 0
            normalized_df[column] = normalized_df[column] / max_val
    return normalized_df

# Normalize the deforest DataFrames if they are not empty
normalized_deforest_2017_2021 = normalize_dataframe(deforest_2017_2021) if not deforest_2017_2021.empty else pd.DataFrame()
normalized_deforest_2018_2021 = normalize_dataframe(deforest_2018_2021) if not deforest_2018_2021.empty else pd.DataFrame()
normalized_deforest_2022_2023 = normalize_dataframe(deforest_2022_2023) if not deforest_2022_2023.empty else pd.DataFrame()
normalized_deforest_2021_2023 = normalize_dataframe(deforest_2021_2023) if not deforest_2021_2023.empty else pd.DataFrame()

      
# Serialize and save the DataFrame list to a file
import pickle
  
# Load the geometry DataFrame from the pickle file
pickle_file_path = 'D:/ITC/Thesis/Scripts/results/geometry_only_df.pkl'
geometry_df = pd.read_pickle(pickle_file_path)

csv_file_path = 'D:/ITC/Thesis/Scripts/results/deforest_2022_2023.csv'
deforest_2022_2023.to_csv(csv_file_path, index=False)

csv_file_path = 'D:/ITC/Thesis/Scripts/results/old_normalized_deforest_2022_2023.csv'
normalized_deforest_2022_2023.to_csv(csv_file_path, index=False)

csv_file_path = 'D:/ITC/Thesis/Scripts/results/old_normalized_deforest_2022_2023_v2.csv'
normalized_deforest_2022_2023.to_csv(csv_file_path, index=False)

# Concatenate the predicted values and the geometry DataFrame
df_combined = pd.concat([normalized_deforest_2022_2023, geometry_df.reset_index(drop=True)], axis=1)

# Ensure the combined DataFrame is a GeoDataFrame
gdf_combined = gpd.GeoDataFrame(df_combined, geometry=geometry_df.geometry)

# Save the GeoDataFrame to a shapefile
output_shapefile = 'D:/ITC/Thesis/Scripts/results/results/normalized_deforest/old_normalized_deforest_2022_2023.shp'
gdf_combined.to_file(output_shapefile)

print(f"Predicted values with geometry saved to '{output_shapefile}'.")

# Example: Print the first few rows of the combined DataFrame
print(gdf_combined.head())

