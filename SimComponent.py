# --------------------------------------------------
# All-in-one Python code for:
# 1. Reading data with columns ["VEHICLE", "COMPLETION_DATE"]
# 2. Preprocessing and grouping by month
# 3. Plotting monthly production counts (vehicles completed per month)
# 4. Time series forecasting using Prophet
# 5. Splitting data for model evaluation
# 6. Computing forecast metrics
# --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------
# 1. Read and preprocess data
# ------------------------
# Replace 'your_data.csv' with the actual path to your dataset
df = pd.read_csv('your_data.csv')  

# Convert COMPLETION_DATE to datetime
df['COMPLETION_DATE'] = pd.to_datetime(df['COMPLETION_DATE'])

# ------------------------
# 2. Group by month and count vehicles
# ------------------------
# Group by month (start of each month) to get monthly totals
df_monthly = (
    df.groupby(pd.Grouper(key='COMPLETION_DATE', freq='MS'))
      .size()
      .reset_index(name='count')
)

# ------------------------
# 3. Plot monthly production counts
# ------------------------
plt.figure(figsize=(10, 5))
plt.plot(df_monthly['COMPLETION_DATE'], df_monthly['count'], marker='o')
plt.title('Monthly Vehicle Completion Counts')
plt.xlabel('Month')
plt.ylabel('Number of Vehicles Completed')
plt.grid(True)
plt.show()

# Let's also print the top 5 months with highest outputs
top_months = df_monthly.nlargest(5, 'count')
print("Top 5 months with highest completion counts:")
print(top_months)

# ------------------------
# 4. Prepare data for Prophet
# ------------------------
# Prophet expects 'ds' and 'y' as column names
df_monthly_prophet = df_monthly.rename(columns={'COMPLETION_DATE': 'ds', 'count': 'y'})

# ------------------------
# 5. Split data into train/test for evaluation
# ------------------------
# For simplicity, let's take the last 3 months as the test set
split_index = len(df_monthly_prophet) - 3
train_data = df_monthly_prophet.iloc[:split_index]
test_data = df_monthly_prophet.iloc[split_index:].copy()

# ------------------------
# 6. Build and fit the Prophet model
# ------------------------
model = Prophet(seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
model.fit(train_data)

# ------------------------
# 7. Forecast on the test period
# ------------------------
# We'll create a future dataframe that covers the test period 
# (and possibly further if we want to see beyond the test window)
future_dates = model.make_future_dataframe(
    periods=3,  # 3 months beyond the training set
    freq='MS'   # monthly start
)
forecast = model.predict(future_dates)

# Extract only the rows corresponding to the test set for evaluation
test_forecast = forecast[forecast['ds'].isin(test_data['ds'])].copy()

# ------------------------
# 8. Evaluation on the test set
# ------------------------
# Merge actual values with forecast
evaluation_df = pd.merge(
    test_data[['ds', 'y']],
    test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
    on='ds',
    how='left'
)

# Calculate metrics
mse = mean_squared_error(evaluation_df['y'], evaluation_df['yhat'])
rmse = np.sqrt(mse)
mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])

print("\nEvaluation on Test Set:")
print("-----------------------")
print(f"Test MSE:  {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE:  {mae:.2f}")

print("\nTest Data vs. Forecast:")
print(evaluation_df)

# ------------------------
# 9. Plot the forecast for entire dataset (including future)
# ------------------------
fig1 = model.plot(forecast)
plt.title("Prophet Forecast (Training + Test + Future)")
plt.xlabel("Month")
plt.ylabel("Vehicles Completed")
plt.show()

# If you want an extended forecast beyond just the test set,
# you can specify a larger `periods` above, e.g. periods=6 or 12
# to forecast 6 or 12 months into the future.