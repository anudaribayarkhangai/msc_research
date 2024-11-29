import os
import geopandas as gpd
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import copy
import pickle

# Directory containing the shapefiles
directory_path = 'D:/ITC/Thesis/data/BD_AMZ_25km_variaveis_1/data'

# List all shapefiles in the directory
shapefiles = [f for f in os.listdir(directory_path) if f.endswith('.shp')]

# Initialize the list to store the shapefiles
all_shapefiles_list = []

# Loop through the shapefiles and read them using GeoPandas
for shp_file in shapefiles:
    file_path = os.path.join(directory_path, shp_file)
    shp = gpd.read_file(file_path)
    all_shapefiles_list.append(shp)

# Create a deep copy of the list
all_shapefiles_copy = copy.deepcopy(all_shapefiles_list)
    
# Adding a new column to the dataframe at index 3
index = 3
if index < len(all_shapefiles_list):
    all_shapefiles_list[index]['DeAR010117'] = 0

# Sort columns of DataFrame at index 3
def sort_columns(df):
    # Define the desired order of the first columns
    initial_columns = ['id', 'col', 'row', 'geometry']
    # Extract the date columns starting with 'DeAR'
    date_columns = [col for col in df.columns if col.startswith('DeAR')]
    # Sort the date columns
    sorted_date_columns = sorted(date_columns, key=lambda x: datetime.strptime(x[4:], '%d%m%y'))
    # Keep other columns in their original order
    other_columns = [col for col in df.columns if col not in initial_columns + date_columns]
    # Combine initial columns with sorted date columns and other remaining columns
    sorted_columns = initial_columns + sorted_date_columns + other_columns
    # Reorder the DataFrame columns
    return df[sorted_columns]

# Apply sorting to the DataFrame at index 3
all_shapefiles_list[index] = sort_columns(all_shapefiles_list[index])


# Working with the DataFrame at index 4 in the copied list
index = 4
df = all_shapefiles_list[index]

# Generate dates for 1st and 16th of each month from 2017 to 2023
start_year = 2017
end_year = 2023

# Create a list of the 1st and 16th of each month
dates = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        dates.append(pd.Timestamp(year=year, month=month, day=1))
        dates.append(pd.Timestamp(year=year, month=month, day=16))

# Dictionary to map old columns to new columns
new_columns = {}
for col in df.columns:
    if col not in ['id', 'row', 'col', 'geometry']:  # Preserve 'id', 'row', and 'col'
        for date in dates:
            new_col_name = f'{col}{date.strftime("%d%m%y")}'
            new_columns[new_col_name] = col

# Create new columns with renamed headers and remove the original columns
for new_col, old_col in new_columns.items():
    df[new_col] = df[old_col]
df.drop(columns=[col for col in df.columns if col not in ['id', 'row', 'col', 'geometry'] + list(new_columns.keys())], inplace=True)

# Update the copied list with the modified DataFrame
all_shapefiles_list[index] = df

# Select the DataFrame at index 6
df = all_shapefiles_list[6]

# Identify the source column and target columns
source_col = 'DS310317'
target_cols = ['DS010117', 'DS160117', 'DS010217', 'DS160217']

# Duplicate the data from the source column to the target columns
for target_col in target_cols:
    df[target_col] = df[source_col]

# Optionally, remove the source column if no longer needed
# del df[source_col]

# Update the DataFrame at index 6
all_shapefiles_list[6] = df

# Function to generate target dates
def generate_target_dates(year, month):
    dates = []
    for i in range(3):  # Current month + next two months
        target_date = datetime(year, month, 1) + relativedelta(months=i)
        if 2017 <= target_date.year <= 2023:
            dates.append(target_date.strftime('01%m%y'))
            dates.append(target_date.replace(day=16).strftime('16%m%y'))
    return dates

# Function to process 'DSddmmyy' columns
def process_ds_columns(df):
    excluded_columns = {'row', 'col', 'id', 'geometry', 'DS010117', 'DS160117', 'DS010217', 'DS160217'}
    ds_columns = [col for col in df.columns if col.startswith('DS') and col not in excluded_columns
                  and (col[2:4] not in ['01', '16']) and (2017 <= int('20' + col[6:8]) <= 2023)]
    
    for col in ds_columns:
        day = int(col[2:4])
        month = int(col[4:6])
        year = 2000 + int(col[6:8])
        if year not in range(2017, 2024):
            continue
        source_date = datetime(year, month, day)
        target_columns = generate_target_dates(year, month)
        for target in target_columns:
            new_column_name = f'DS{target}'
            df[new_column_name] = df[col]
        del df[col]

# Process each DataFrame in the list
for i in range(len(all_shapefiles_list)):
    df = all_shapefiles_list[i]
    process_ds_columns(df)
    all_shapefiles_list[i] = df


def generate_date_range(start, end):
    # Generate dates from start to end for the 1st and 16th of each month
    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date.strftime('%d%m%y'))
        if current_date.day == 1:
            current_date = current_date.replace(day=16)
        else:
            current_date = (current_date.replace(day=1) + pd.DateOffset(months=1)).replace(day=1)
    return dates

# Generate the required date range
start_date = datetime(2017, 1, 1)
end_date = datetime(2018, 2, 16)
date_range = generate_date_range(start_date, end_date)

def update_dataframe_with_zeros(df, prefix, date_range):
    for date_str in date_range:
        column_name = f'{prefix}{date_str}'
        df[column_name] = 0

# Process DataFrame at index 11
df_11 = all_shapefiles_list[11]
update_dataframe_with_zeros(df_11, 'XArDS', date_range)
all_shapefiles_list[11] = df_11

# Process DataFrame at index 12
df_12 = all_shapefiles_list[12]
update_dataframe_with_zeros(df_12, 'XDeDS', date_range)
all_shapefiles_list[12] = df_12


def generate_dates(year, month):
    # This function generates the 1st and 16th of the month for the given month and the next two months
    date = datetime(year, month, 1)
    dates = []
    for m in range(0, 3):  # The given month and the next two months
        current_month = date + relativedelta(months=m)
        if current_month.year > 2023:
            break
        dates.append(current_month.strftime('%d%m%y'))
        mid_month = current_month.replace(day=16)
        dates.append(mid_month.strftime('%d%m%y'))
    return dates

def update_columns(df, prefix):
    # Months and their corresponding numerical values for the process
    months = {'03': 3, '06': 6, '09': 9, '12': 12}
    years = range(2017, 2024)  # Ensure the last year processed is 2023

    # Define excluded columns
    excluded_columns = {'id', 'row', 'col', 'geometry'}
    # Add dates from 010117 to 160218
    date_exclusions = generate_date_range(datetime(2017, 1, 1), datetime(2018, 2, 16))
    excluded_columns.update({f'{prefix}{date}' for date in date_exclusions})

    # Loop over each year and month combination
    for year in years:
        for month_str, month in months.items():
            old_column = f'{prefix}{month_str}{str(year)[2:]}'
            if old_column in df.columns and old_column not in excluded_columns:
                # Generate the appropriate new column names and copy data
                for date_str in generate_dates(year, month):
                    new_column = f'{prefix}{date_str}'
                    df[new_column] = df[old_column]
                # Delete the old column
                del df[old_column]

def generate_date_range(start, end):
    # Generate dates from start to end for the 1st and 16th of each month
    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date.strftime('%d%m%y'))
        if current_date.day == 1:
            current_date = current_date.replace(day=16)
        else:
            current_date = (current_date.replace(day=1) + pd.DateOffset(months=1)).replace(day=1)
    return dates

# Process DataFrames at index 11 and 12
for index in [11, 12]:
    df = all_shapefiles_list[index]
    prefix = 'XArDS' if index == 11 else 'XDeDS'
    update_columns(df, prefix)
    all_shapefiles_list[index] = df


def generate_date_range(start, end):
    # Generate dates from start to end for the 1st and 16th of each month
    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date.strftime('%d%m%y'))
        if current_date.day == 1:
            current_date = current_date.replace(day=16)
        else:
            current_date = (current_date.replace(day=1) + pd.DateOffset(months=1)).replace(day=1)
    return dates

def update_dataframe_with_zeros(df, prefix, date_range):
    for date_str in date_range:
        column_name = f'{prefix}{date_str}'
        df[column_name] = 0

def sort_columns(df):
    # Define the desired order of the first columns
    initial_columns = ['id', 'col', 'row', 'geometry']
    # Extract the date columns
    date_columns = [col for col in df.columns if col.startswith('nv')]
    # Sort the date columns
    sorted_date_columns = sorted(date_columns, key=lambda x: datetime.strptime(x[2:], '%d%m%y'))
    # Combine initial columns with sorted date columns
    sorted_columns = initial_columns + sorted_date_columns
    # Reorder the DataFrame columns
    return df[sorted_columns]

# Generate the required date range from 010117 to 160717
start_date = datetime(2017, 1, 1)
end_date = datetime(2017, 7, 16)
date_range = generate_date_range(start_date, end_date)

# Process DataFrame at index 9
df_9 = all_shapefiles_list[9]
update_dataframe_with_zeros(df_9, 'nv', date_range)
df_9 = sort_columns(df_9)
all_shapefiles_list[9] = df_9

df_10 = all_shapefiles_list[10]

# Drop the 'unique_id' column if it exists
if 'unique_id' in df_10.columns:
    df_10 = df_10.drop(columns=['unique_id'])

# Update the DataFrame at index 10
all_shapefiles_list[10] = df_10


def update_dataframe_with_specific_columns(df, prefix, date_list):
    for date_str in date_list:
        column_name = f'{prefix}{date_str}'
        df[column_name] = 0

def sort_columns(df):
    # Define the desired order of the first columns
    initial_columns = ['id', 'col', 'row', 'geometry']
    # Extract the date columns, keeping existing ones and new ones
    date_columns = [col for col in df.columns if col.startswith('XQ')]
    # Sort the date columns
    sorted_date_columns = sorted(date_columns, key=lambda x: datetime.strptime(x[2:], '%d%m%y'))
    # Combine initial columns with sorted date columns and other remaining columns
    other_columns = [col for col in df.columns if col not in initial_columns + sorted_date_columns]
    sorted_columns = initial_columns + sorted_date_columns + other_columns
    # Reorder the DataFrame columns
    return df[sorted_columns]

# Define the specific dates needed
specific_dates = ['010117', '160117', '010217', '160217']

# Process DataFrame at index 11
df_13 = all_shapefiles_list[13]
update_dataframe_with_specific_columns(df_13, 'XQ', specific_dates)
df_13 = sort_columns(df_13)
all_shapefiles_list[13] = df_13

# List of indices to be processed
indices_to_process = [5, 7, 8]

# Generate dates for 1st and 16th of each month from 2017 to 2023
start_year = 2017
end_year = 2023

# Create a list of the 1st and 16th of each month
dates = []
for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        dates.append(pd.Timestamp(year=year, month=month, day=1))
        dates.append(pd.Timestamp(year=year, month=month, day=16))

# Process each DataFrame at the specified indices
for index in indices_to_process:
    df = all_shapefiles_list[index]
    
    # Dictionary to map old columns to new columns
    new_columns = {}
    for col in df.columns:
        if col not in ['id', 'row', 'col', 'geometry']:  # Preserve 'id', 'row', and 'col'
            for date in dates:
                new_col_name = f'{col}{date.strftime("%d%m%y")}'
                new_columns[new_col_name] = col
    
    # Create new columns with renamed headers and remove the original columns
    for new_col, old_col in new_columns.items():
        df[new_col] = df[old_col]
    df.drop(columns=[col for col in df.columns if col not in ['id', 'row', 'col', 'geometry'] + list(new_columns.keys())], inplace=True)
    
    # Update the copied list with the modified DataFrame
    all_shapefiles_list[index] = df
    

def separate_attributes(df):
    """Separate attributes of a DataFrame based on unique name parts."""
    separated_dfs = {}
    for col in df.columns:
        if col not in ['id', 'col', 'row', 'geometry']:
            name_part = ''.join([i for i in col if not i.isdigit()])  # Remove digits to isolate the name
            if name_part not in separated_dfs:
                separated_dfs[name_part] = df[['id', 'col', 'row', 'geometry']].copy()
            separated_dfs[name_part][col] = df[col]
    return list(separated_dfs.values())

# Extract, separate, and replace for index 4
dfs_to_add_4 = separate_attributes(all_shapefiles_list[4])
all_shapefiles_list.pop(4)
for i, new_df in enumerate(dfs_to_add_4):
    all_shapefiles_list.insert(4 + i, new_df)

# Adjust index 8 to account for the shift after inserting new DataFrames at index 4
index_8_adjusted = 8 + len(dfs_to_add_4) - 1

# Extract, separate, and replace for index 8 (adjusted)
dfs_to_add_8 = separate_attributes(all_shapefiles_list[index_8_adjusted])
all_shapefiles_list.pop(index_8_adjusted)
for i, new_df in enumerate(dfs_to_add_8):
    all_shapefiles_list.insert(index_8_adjusted + i, new_df)
    
    
metadata = all_shapefiles_list[0][['id', 'col', 'row', 'geometry']]

for i in range(len(all_shapefiles_list)):
    if not all_shapefiles_list[i].empty:
        all_shapefiles_list[i] = all_shapefiles_list[i].drop(columns=['id', 'col', 'row', 'geometry'])
        

for var in list(globals().keys()):
    if var not in ['all_shapefiles_copy', 'all_shapefiles_list', 'metadata', 'pd']:
        del globals()[var]
        
# Serialize and save the DataFrame list to a file
with open('D:/ITC/Thesis/Scripts/results/all_shapefiles_list.pkl', 'wb') as file:
    pickle.dump(all_shapefiles_list, file)
    
# Extract the 'geometry' column from the DataFrame at index 1 of the original list
geometry_df = all_shapefiles_copy[1][['geometry']]

# Display the extracted DataFrame
print(geometry_df)


# Save the extracted DataFrame to a pickle file
geometry_df.to_pickle('D:/ITC/Thesis/Scripts/results/geometry_only_df.pkl')
