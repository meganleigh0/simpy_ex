import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data
df = pd.read_csv("plant_data.csv")

# 2. Summarize or pivot by vehicle
pivot_df = (
    df.groupby(['VEHICLE','VARIANT','SECTION'], as_index=False)
      .agg({'NUM_DAYS_SPENT':'sum'})
    .pivot_table(index=['VEHICLE','VARIANT'],
                 columns='SECTION',
                 values='NUM_DAYS_SPENT',
                 fill_value=0)
    .reset_index()
)

pivot_df.columns.name = None
pivot_df.rename(columns={'HP1': 'HP1_time',
                         'TP1': 'TP1_time',
                         'HP3': 'HP3_time',
                         'TP3': 'TP3_time',
                         'V': 'V_time'},
                inplace=True)

pivot_df['total_time'] = pivot_df[['HP1_time','TP1_time','HP3_time','TP3_time','V_time']].sum(axis=1)

# 3. Separate completed vs. incomplete
completed_vehicles_df = pivot_df[
    (pivot_df['HP1_time']>0)&
    (pivot_df['TP1_time']>0)&
    (pivot_df['HP3_time']>0)&
    (pivot_df['TP3_time']>0)&
    (pivot_df['V_time']>0)
].copy()

incomplete_df = pivot_df[~pivot_df.index.isin(completed_vehicles_df.index)].copy()

# 4. Train a simple model on completed vehicles
completed_vehicles_df['VARIANT_encoded'] = completed_vehicles_df['VARIANT'].astype('category').cat.codes
X = completed_vehicles_df[['VARIANT_encoded','HP1_time','TP1_time','HP3_time','TP3_time','V_time']]
y = completed_vehicles_df['total_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 5. Predict for incomplete vehicles
incomplete_df['VARIANT_encoded'] = incomplete_df['VARIANT'].astype('category').cat.codes
incomplete_df['spent_so_far'] = incomplete_df[['HP1_time','TP1_time','HP3_time','TP3_time','V_time']].sum(axis=1)
X_incomplete = incomplete_df[['VARIANT_encoded','HP1_time','TP1_time','HP3_time','TP3_time','V_time']]

incomplete_df['predicted_total_time'] = model.predict(X_incomplete)
incomplete_df['remaining_time'] = incomplete_df['predicted_total_time'] - incomplete_df['spent_so_far']
incomplete_df['remaining_time'] = incomplete_df['remaining_time'].clip(lower=0)

# For demonstration, assume "today" is a known date
today = pd.to_datetime("2025-01-01")
incomplete_df['estimated_completion_date'] = incomplete_df.apply(
    lambda row: today + pd.Timedelta(days=row['remaining_time']),
    axis=1
)

# 6. Combine everything
# Let's unify the 'completed_vehicles_df' and 'incomplete_df' with an 'estimated_completion_date'

# For completed vehicles, assume their actual finishing date = some known date column
# If you only have total time, you might not have actual finish dates. As a placeholder:
# We'll just define them as finished on some earlier day.
completed_vehicles_df['estimated_completion_date'] = today

full_df = pd.concat([completed_vehicles_df, incomplete_df], ignore_index=True)

# 7. Group by month, variant for next 3 months
full_df['completion_month'] = full_df['estimated_completion_date'].dt.to_period('M')

schedule_summary = (
    full_df.groupby(['completion_month','VARIANT'])
    .agg(Qty=('VEHICLE','count'))
    .reset_index()
    .sort_values(['completion_month','VARIANT'])
)

# Print or save schedule
print(schedule_summary)