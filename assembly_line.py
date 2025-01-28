# ============================================
# ALL-IN-ONE PYTHON EXAMPLE (NO QTY COLUMN)
# ============================================
#
# This example:
#  1) Loads a historical dataset.
#  2) Creates a boxplot of time spent (NUMBER_DAYS_SPENT) by SECTION.
#  3) Builds a simple predictive model to estimate days spent 
#     based on (VARIANT, MONTH).
#  4) Demonstrates how to check if a planned schedule is likely met,
#     with a simple visualization of predicted vs. planned days.
#
# Adjust file paths, column names, and parameters as needed.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------
# 1) LOAD DATA AND PREP
# --------------------------------------------------------
# Replace 'your_data.csv' with the actual CSV file of your 89,893 records
df = pd.read_csv('your_data.csv')

# Columns assumed: ['VEHICLE','SECTION','STATION','DATE','VARIANT','NUMBER_DAYS_SPENT', ...]
# Convert DATE to datetime if it isn't already
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# Extract numeric month from DATE (1 to 12)
df['MONTH'] = df['DATE'].dt.month

# Drop rows where we cannot extract necessary info
df.dropna(subset=['VARIANT','MONTH','NUMBER_DAYS_SPENT','SECTION'], inplace=True)

# --------------------------------------------------------
# 2) BOX PLOT OF TIME SPENT BY SECTION
# --------------------------------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='SECTION', y='NUMBER_DAYS_SPENT')
plt.title('Box Plot of Time Spent by Section')
plt.xlabel('Section')
plt.ylabel('Number of Days Spent')
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 3) BUILD A SIMPLE PREDICTIVE MODEL
# --------------------------------------------------------
# We want to predict NUMBER_DAYS_SPENT from (VARIANT, MONTH).
model_df = df[['VARIANT','MONTH','NUMBER_DAYS_SPENT']].copy()

# Label-encode the VARIANT feature (categorical)
le = LabelEncoder()
model_df['VARIANT'] = le.fit_transform(model_df['VARIANT'])

# Features X and target y
X = model_df[['VARIANT', 'MONTH']]
y = model_df['NUMBER_DAYS_SPENT']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use a Random Forest Regressor (could be replaced or tuned as desired)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Quick evaluation
y_pred = reg.predict(X_test)
mae = np.mean(np.abs(y_pred - y_test))
print(f"Test MAE (Mean Absolute Error): {mae:.2f} days")

# --------------------------------------------------------
# 4) FUNCTION TO PREDICT AND VISUALIZE SCHEDULE CHECK
# --------------------------------------------------------
def predict_schedule(variant_input, month_input, planned_days):
    """
    Given VARIANT, MONTH, and a planned schedule (in days),
    predict how many days it will actually take using the model,
    then determine if schedule is likely to be met.
    """
    # Transform variant using the same LabelEncoder
    if variant_input in le.classes_:
        variant_encoded = le.transform([variant_input])[0]
    else:
        # If variant not in training set, fallback to 0 or handle differently
        variant_encoded = 0
    
    # Prepare feature vector
    X_new = np.array([[variant_encoded, month_input]])
    
    # Predict
    pred_days = reg.predict(X_new)[0]
    
    # Compare to planned
    if pred_days <= planned_days:
        result_text = (f"Likely to meet schedule "
                       f"(Predicted {pred_days:.1f} <= Planned {planned_days})")
    else:
        result_text = (f"Unlikely to meet schedule "
                       f"(Predicted {pred_days:.1f} > Planned {planned_days})")
    
    # Visualization (bar chart)
    plt.figure(figsize=(4,4))
    labels = ['Planned', 'Predicted']
    values = [planned_days, pred_days]
    sns.barplot(x=labels, y=values, palette="Blues_d")
    plt.title(f"Schedule vs Prediction\n(VARIANT={variant_input}, MONTH={month_input})")
    plt.ylabel("Days")
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center', color='black', fontweight='bold')
    plt.ylim(0, max(values) + 2)
    plt.tight_layout()
    plt.show()
    
    return pred_days, result_text

# EXAMPLE USAGE:
variant_test  = "A"   # Must be one of the VARIANTs in the dataset
month_test    = 9     # e.g. September
planned_days  = 5     # Suppose we planned 5 days

predicted_days, schedule_outcome = predict_schedule(variant_test, month_test, planned_days)
print(f"Predicted {predicted_days:.1f} days\n{schedule_outcome}")