# =========================
# ALL-IN-ONE PYTHON EXAMPLE
# =========================
# This example demonstrates:
#  1) Loading the historical dataset.
#  2) Creating a boxplot of time spent (Number_Days_Spent) by Section.
#  3) Training a simple predictive model to estimate completion time 
#     from VARIANT, QTY, MONTH (and then deriving whether schedule can be met).
#  4) Visualizing the prediction versus schedule.
#
# NOTE:
#  - Adjust file paths, column names, and model hyperparameters as needed.
#  - In practice you’ll want to refine data cleaning, feature engineering,
#    and modeling for your specific scenario.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------------
# 1) LOAD AND PREP DATA
# ------------------------------------------------------------------------
# Replace 'your_data.csv' with your actual dataset filename or path
# If you already have the data in a DataFrame named df, skip the read_csv.
df = pd.read_csv('your_data.csv')

# Example columns assumed:
# df.columns --> ['VEHICLE', 'SECTION', 'STATION', 'DATE', 'VARIANT', 'NUMBER_DAYS_SPENT', ...]
# If you have more columns, adjust as needed.

# Make sure DATE is in datetime form if you need it
df['DATE'] = pd.to_datetime(df['DATE'])

# For demonstration, create a 'MONTH' feature (the month of the date)
df['MONTH'] = df['DATE'].dt.month

# If QTY doesn't exist in your dataset but you plan to use it, 
# you might add or load it from somewhere. For now, we’ll assume
# there’s a 'QTY' column or we set it artificially:
if 'QTY' not in df.columns:
    # Dummy example if QTY not present:
    df['QTY'] = np.random.randint(1, 10, len(df))

# ------------------------------------------------------------------------
# 2) CREATE BOX PLOT OF TIME SPENT BY SECTION
# ------------------------------------------------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(x='SECTION', y='NUMBER_DAYS_SPENT', data=df)
plt.title('Box Plot of Time Spent by Section')
plt.xlabel('Section')
plt.ylabel('Number of Days Spent')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------
# 3) BUILD A SIMPLE PREDICTIVE MODEL
# ------------------------------------------------------------------------
# We’ll predict NUMBER_DAYS_SPENT based on (VARIANT, QTY, MONTH).
# Then from predicted days, we can infer if schedule is met.

# Filter to columns we need
model_df = df[['VARIANT','QTY','MONTH','NUMBER_DAYS_SPENT']].dropna()

# Label-encode VARIANT if it's categorical
le = LabelEncoder()
model_df['VARIANT'] = le.fit_transform(model_df['VARIANT'])

# Features (X) and target (y)
X = model_df[['VARIANT','QTY','MONTH']]
y = model_df['NUMBER_DAYS_SPENT']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor (feel free to tune hyperparameters)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Evaluate quickly on test set (basic check)
test_preds = reg.predict(X_test)
mae = np.mean(abs(test_preds - y_test))
print(f"Test MAE (Mean Absolute Error): {mae:.2f} days")

# ------------------------------------------------------------------------
# 4) FUNCTION TO PREDICT COMPLETION & VISUALIZE SCHEDULE
# ------------------------------------------------------------------------
def predict_schedule(variant_input, qty_input, month_input, planned_days):
    """
    Given a new input (VARIANT, QTY, MONTH) and a planned schedule in days,
    this predicts how many days the operation will actually take, 
    then compares to the planned schedule.
    
    Returns predicted_days and a statement on whether schedule is likely met.
    """
    # Convert variant input to label-encoded value
    variant_transformed = le.transform([variant_input])[0] \
                           if variant_input in le.classes_ else 0
    
    # Prepare the feature vector for prediction
    X_new = np.array([[variant_transformed, qty_input, month_input]])
    predicted_days = reg.predict(X_new)[0]
    
    # Compare to planned
    if predicted_days <= planned_days:
        result = f"Likely to meet schedule (Predicted {predicted_days:.1f} <= Planned {planned_days})"
    else:
        result = f"Unlikely to meet schedule (Predicted {predicted_days:.1f} > Planned {planned_days})"
    
    return predicted_days, result

# Example usage for a hypothetical scenario:
variant_test = "A"   # some variant existing in your data
qty_test     = 5
month_test   = 9
planned_days = 6

pred_days, result_str = predict_schedule(variant_test, qty_test, month_test, planned_days)
print(f"\n---PREDICTION RESULT---\nVariant: {variant_test}, QTY: {qty_test}, Month: {month_test}")
print(f"Predicted Completion: {pred_days:.1f} days")
print(result_str)

# OPTIONAL: Quick bar chart to illustrate
labels = ['Planned', 'Predicted']
values = [planned_days, pred_days]

plt.figure(figsize=(4,4))
sns.barplot(x=labels, y=values, palette="Blues_d")
plt.title(f"Schedule vs Prediction (Variant={variant_test}, QTY={qty_test}, Month={month_test})")
plt.ylabel("Days")
for i, v in enumerate(values):
    plt.text(i, v+0.1, f"{v:.1f}", ha='center', color='black', fontweight='bold')
plt.ylim(0, max(values)+2)
plt.tight_layout()
plt.show()