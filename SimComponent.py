import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# 1. Read in or have your DataFrames
# ---------------------------------------------------------------------
# Example structure (replace with your actual data loading)
# current_status_df has columns: [Vehicle, Section, Station, Date, Variant, Number]
# station_df has columns: [Station, average_days, median_days, Variant]

# current_status_df = pd.read_csv("current_status.csv")
# station_df = pd.read_csv("station_history.csv")

# For illustration, let's make some tiny mock data
current_status_df = pd.DataFrame({
    'Vehicle': ['VIN123', 'VIN456', 'VIN789'],
    'Section': ['Paint', 'Assembly', 'Assembly'],
    'Station': ['STN1', 'STN2', 'STN3'],
    'Date': pd.to_datetime(['2025-02-01', '2025-02-02', '2025-02-03']),
    'Variant': ['V1', 'V2', 'V1'],
    'Number': [2, 1, 3]  # days so far at that station
})

station_df = pd.DataFrame({
    'Station': ['STN1', 'STN2', 'STN3'],
    'Variant': ['V1', 'V2', 'V1'],
    'average_days': [3.2, 5.5, 4.1],
    'median_days': [3, 5, 4]
})

# ---------------------------------------------------------------------
# 2. Merge DataFrames to get historical features (avg, median, etc.)
# ---------------------------------------------------------------------
# We can do a left join from current_status_df to station_df
merged_df = pd.merge(
    current_status_df, 
    station_df, 
    how='left',
    on=['Station', 'Variant']
)

# The merged DataFrame now contains all columns from current_status_df
# plus average_days, median_days from the station_df.

# merged_df might look like:
#   Vehicle Section   Station  Date        Variant Number  average_days  median_days
# 0 VIN123  Paint     STN1     2025-02-01 V1      2       3.2           3
# 1 VIN456  Assembly  STN2     2025-02-02 V2      1       5.5           5
# 2 VIN789  Assembly  STN3     2025-02-03 V1      3       4.1           4

# ---------------------------------------------------------------------
# 3. Create a historical dataset with known completion times
#    for training a supervised model
# ---------------------------------------------------------------------
# In real use, you'll have a historical dataset with completion info.
# For demonstration, let's create a hypothetical "historical_data_df":
#   Each row is a vehicle in some station snapshot, along with the final
#   days_to_completion that actually happened. This is your 'label'.
historical_data_df = pd.DataFrame({
    'Vehicle': ['VIN0001', 'VIN0002', 'VIN0003', 'VIN0004', 'VIN0005'],
    'Section': ['Paint', 'Paint', 'Assembly', 'Assembly', 'Assembly'],
    'Station': ['STN1', 'STN1', 'STN2', 'STN3', 'STN2'],
    'Date': pd.to_datetime(['2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05']),
    'Variant': ['V1', 'V1', 'V2', 'V1', 'V2'],
    'Number': [1, 2, 1, 1, 2],  # days so far at that station
    # "days_to_completion" is how many more days from this point until final completion
    'days_to_completion': [5, 4, 10, 3, 9]
})

# Merge in the average/median station info
historical_merged_df = pd.merge(
    historical_data_df,
    station_df,
    how='left',
    on=['Station', 'Variant']
)

# ---------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------
# For example, we can keep:
#   - 'Number' (days at current station so far)
#   - 'average_days', 'median_days'
#   - Possibly convert 'Section' and 'Station' into categorical dummies
#   - 'Variant' also as categorical
#   - Exclude 'Vehicle' and 'Date' for now
#   - 'days_to_completion' is our target

# Simple example:
feature_cols = ['Number', 'average_days', 'median_days', 'Section', 'Station', 'Variant']
target_col = 'days_to_completion'

# Convert categorical columns to dummies (one-hot encoding)
categorical_cols = ['Section', 'Station', 'Variant']
historical_merged_df = pd.get_dummies(historical_merged_df, columns=categorical_cols, drop_first=True)

X = historical_merged_df[[c for c in historical_merged_df.columns if c not in ['Vehicle','Date','days_to_completion']]]
y = historical_merged_df[target_col]

# ---------------------------------------------------------------------
# 5. Train/Test Split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------
# 6. Train a Regression Model
# ---------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# 7. Evaluate the Model
# ---------------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"MAE on Test: {mae:.2f} days")
print(f"R^2 on Test: {r2:.3f}")

# ---------------------------------------------------------------------
# 8. Predict for Current Vehicles
# ---------------------------------------------------------------------
# Now, we want to predict 'days_to_completion' for vehicles in current_status_df (merged_df).
# We must replicate the same feature engineering steps we did on the training set
current_features_df = merged_df.copy()

# Convert the same categorical columns to dummies
# We must ensure the columns are consistent with the training dummy columns
current_features_df = pd.get_dummies(current_features_df, columns=categorical_cols, drop_first=True)

# Some columns might not exist if the category wasn't in the new data (or vice versa).
# A safe approach is to reindex the new data's columns to match the training data's columns:
missing_cols = set(X.columns) - set(current_features_df.columns)
for col in missing_cols:
    current_features_df[col] = 0
extra_cols = set(current_features_df.columns) - set(X.columns)
current_features_df.drop(extra_cols, axis=1, inplace=True)

current_features_df = current_features_df[X.columns]  # ensure same column order

# Predict
current_predictions = model.predict(current_features_df)

# Attach predictions back to the original merged_df
merged_df['estimated_days_to_completion'] = current_predictions

# We can also estimate a completion date by adding the predicted days to the current date
# (for demonstration, we’ll just add to the 'Date' column in merged_df)
merged_df['estimated_completion_date'] = merged_df['Date'] + pd.to_timedelta(merged_df['estimated_days_to_completion'].round(), unit='D')

# ---------------------------------------------------------------------
# 9. Examine the Results
# ---------------------------------------------------------------------
print(merged_df[['Vehicle','Station','Variant','Date','estimated_days_to_completion','estimated_completion_date']])