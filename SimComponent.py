import pandas as pd

# Ensure DateCount is in datetime format if needed
df['DateCount'] = pd.to_datetime(df['DateCount'])

def get_prev_station_time(row):
    # Filter for the same vehicle, where the station equals the current row's Parent
    # and the DateCount is earlier than the current row's DateCount.
    mask = (
        (df['Vehicle'] == row['Vehicle']) &
        (df['Station'] == row['Parent']) &
        (df['DateCount'] < row['DateCount'])
    )
    subset = df.loc[mask, 'DateCount']
    if not subset.empty:
        # Return the latest DateCount (i.e. the maximum)
        return subset.max()
    else:
        return pd.NaT

# Apply the function to create the new column
df['prev_station_time'] = df.apply(get_prev_station_time, axis=1)