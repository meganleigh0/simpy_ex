import pandas as pd

# Ensure DateCount is in datetime format
df['DateCount'] = pd.to_datetime(df['DateCount'])

def get_initial_time(row, df):
    """
    Recursively traverse the parent chain for the same vehicle until reaching station 0.
    
    Parameters:
        row (pd.Series): The current row.
        df (pd.DataFrame): The full dataframe.
        
    Returns:
        pd.Timestamp or pd.NaT: The DateCount corresponding to station 0 for that vehicle.
    """
    # Base case: if this row is station 0, return its DateCount
    if row['Station'] == 0:
        return row['DateCount']
    
    # Get the station referenced in the Parent column of the current row.
    parent_station = row['Parent']
    
    # Filter the dataframe: same vehicle, station equals the parent's station,
    # and the DateCount is earlier than the current row's DateCount.
    subset = df[
        (df['Vehicle'] == row['Vehicle']) &
        (df['Station'] == parent_station) &
        (df['DateCount'] < row['DateCount'])
    ]
    
    # If no matching parent row is found, return NaT.
    if subset.empty:
        return pd.NaT
    
    # Otherwise, take the row with the most recent DateCount (i.e. the last occurrence)
    parent_row = subset.loc[subset['DateCount'].idxmax()]
    
    # Recursively call the function on this parent row.
    return get_initial_time(parent_row, df)

# Apply the recursive function to create a new column with the initial (station 0) time.
df['initial_station_time'] = df.apply(lambda row: get_initial_time(row, df), axis=1)