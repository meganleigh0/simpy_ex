# --- Single-Cell Baseline Code for Project Proposal ---
# This script outlines a concise approach to load data, preprocess it,
# engineer features, build a model, and predict production throughput.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Data Loading & Basic Preprocessing
# Note: Update 'path_to_data.csv' to your actual CSV or data source.
data = pd.read_csv('path_to_data.csv')

# Columns are assumed as [VEHICLE, VARIANT, STATION, PROGRAM, SECTION, DAYS].
# Basic checks
data.dropna(inplace=True)  # Drop rows with missing values (if minimal)
data['DAYS'] = data['DAYS'].astype(float)  # Ensure numeric type

# 2. Feature Engineering
# Example: Create monthly throughput for plant-level predictions.
# Here we assume there's some "MONTH" column or we derive it from a date.
# If only DAYS are provided without an actual date, we might skip direct time-series
# and do a simpler approach or rely on additional date references.

# (A) Aggregate to get Plant-Level throughput (vehicles completed per month).
# Placeholder: If 'MONTH' is not in data, you would need to create or merge it.
# For demonstration, let's pretend we have a 'MONTH' column:
if 'MONTH' not in data.columns:
    # If no real month info, create a dummy month index as an example
    # This is purely illustrative. In reality, you'd have actual timestamps.
    data['MONTH'] = np.random.randint(1, 19, size=len(data))

plant_monthly = data.groupby('MONTH')['VEHICLE'].nunique().reset_index()
plant_monthly.columns = ['MONTH', 'PLANT_THROUGHPUT']

# (B) Aggregate station-level throughput.
station_monthly = data.groupby(['MONTH', 'STATION'])['VEHICLE'].nunique().reset_index()
station_monthly.columns = ['MONTH', 'STATION', 'STATION_THROUGHPUT']

# (C) Aggregate total days per vehicle (for vehicle-level cycle time).
vehicle_days = data.groupby('VEHICLE')['DAYS'].sum().reset_index()
vehicle_days.columns = ['VEHICLE', 'TOTAL_DAYS']

# 3. Simple Modeling Approach (Example: Predict Plant-Level Throughput)
# We'll train a simple regression model using basic numeric features 
# (like month index) to predict future throughput.

# Merge any additional features you deem relevant. For now, we only use MONTH as a feature.
X = plant_monthly[['MONTH']]
y = plant_monthly['PLANT_THROUGHPUT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"Plant-Level Prediction RMSE: {rmse:.2f}")

# 4. Extending to Station-Level
# In practice, you'd build a similar model or group-based approach for each station,
# or build a multi-output model. Example grouping approach:
station_level_models = {}
stations = station_monthly['STATION'].unique()

for st in stations:
    st_data = station_monthly[station_monthly['STATION'] == st]
    X_st = st_data[['MONTH']]
    y_st = st_data['STATION_THROUGHPUT']
    if len(X_st) < 5:  # minimal check to skip very small data
        continue
    
    X_st_train, X_st_test, y_st_train, y_st_test = train_test_split(
        X_st, y_st, test_size=0.2, random_state=42
    )
    
    st_model = RandomForestRegressor(n_estimators=50, random_state=42)
    st_model.fit(X_st_train, y_st_train)
    st_preds = st_model.predict(X_st_test)
    st_rmse = mean_squared_error(y_st_test, st_preds, squared=False)
    
    station_level_models[st] = st_model
    print(f"Station {st} RMSE: {st_rmse:.2f}")

# 5. Vehicle-Level (Cycle Time Prediction)
# Example: predict total days a vehicle spends in production.
# In a more advanced scenario, you'd have advanced features (vehicle type, station path, etc.).
# For demonstration, let's just randomly assign features to a single regression.

# Create some dummy features for the vehicle (e.g., the variant or program).
vehicle_days = vehicle_days.merge(data[['VEHICLE','VARIANT','PROGRAM']].drop_duplicates(),
                                  on='VEHICLE',
                                  how='left')

# Encode variant/program or any categorical features if needed
vehicle_days['VARIANT_CODE'] = vehicle_days['VARIANT'].factorize()[0]
vehicle_days['PROGRAM_CODE'] = vehicle_days['PROGRAM'].factorize()[0]

X_v = vehicle_days[['VARIANT_CODE','PROGRAM_CODE']]
y_v = vehicle_days['TOTAL_DAYS']

X_v_train, X_v_test, y_v_train, y_v_test = train_test_split(X_v, y_v, test_size=0.2, random_state=42)
vehicle_model = RandomForestRegressor(n_estimators=50, random_state=42)
vehicle_model.fit(X_v_train, y_v_train)

v_preds = vehicle_model.predict(X_v_test)
v_rmse = mean_squared_error(y_v_test, v_preds, squared=False)
print(f"Vehicle-Level Cycle Time RMSE: {v_rmse:.2f}")

# 6. Next Steps:
# - Integrate real date/time fields to build advanced time-series models.
# - Refine feature engineering for better accuracy (e.g., station sequence logic).
# - Validate and deploy production-level pipelines for monthly scheduling forecasts.





import pandas as pd

# 1. Load your data. Update 'path_to_data.csv' as needed.
df = pd.read_csv('path_to_data.csv')

# 2. Ensure DAYS is numeric and drop rows with missing or invalid data if necessary.
df['DAYS'] = pd.to_numeric(df['DAYS'], errors='coerce')
df.dropna(subset=['DAYS'], inplace=True)

# 3. Calculate the average time each vehicle spends at each station.
avg_time_per_station = df.groupby('STATION')['DAYS'].mean().reset_index()
avg_time_per_station.columns = ['STATION', 'AVG_DAYS']

# 4. Sort stations by descending average time.
avg_time_per_station.sort_values('AVG_DAYS', ascending=False, inplace=True)

# 5. Identify bottlenecks. Here, any station whose average days exceed 
#    (mean + 1 standard deviation) is flagged as a bottleneck.
mean_avg_days = avg_time_per_station['AVG_DAYS'].mean()
std_avg_days  = avg_time_per_station['AVG_DAYS'].std()
threshold = mean_avg_days + std_avg_days

bottlenecks = avg_time_per_station[avg_time_per_station['AVG_DAYS'] > threshold]

# 6. Print results
print("=== Average Days per Station (descending order) ===")
print(avg_time_per_station)

print("\n=== Identified Bottlenecks (above threshold) ===")
print(bottlenecks)
