# ==========================================================
# ALL-IN-ONE PYTHON EXAMPLE: HISTORICAL PREDICTION BASELINE
# ==========================================================
#
# This script shows how to:
#   1) Load your historical dataset of station-level records
#   2) Convert it into a "section-level" summary by finding the
#      min and max dates per (VEHICLE, SECTION, VARIANT)
#   3) Compute how many calendar days each VEHICLE spent in that
#      section (as a “historical baseline”)
#   4) Create a box plot of total days (by SECTION)
#   5) Build a predictive model (train/test split + validation)
#      that estimates total days from SECTION and VARIANT
#
# NOTE:
#   - Adjust file paths, column names, and model details
#     (e.g. hyperparameters) to suit your data and needs.
#   - The code is deliberately verbose for illustration.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# 1) LOAD THE DATA
#    --------------
df = pd.read_csv('your_data.csv')  # Replace with your actual path

# Columns assumed:
#   [ "VEHICLE", "SECTION", "STATION", "DATE", "VARIANT", "NUMBER_DAYS_SPENT", ...]
# Convert DATE to datetime (if not already)
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# Drop rows that might be missing critical info
df.dropna(subset=['VEHICLE','SECTION','VARIANT','DATE','NUMBER_DAYS_SPENT'], inplace=True)

# 2) BUILD SECTION-LEVEL SUMMARY
#    ---------------------------
# We want, for each (VEHICLE, SECTION, VARIANT), the min DATE (start), max DATE (end),
# plus a total measure of days spent. We’ll define:
#   - CALENDAR_DAYS = (max_date - min_date).days + 1
#   - SUM_DAYS_SPENT = sum(NUMBER_DAYS_SPENT) across the log entries
# Then we have a single record per vehicle-section-variant.

grouped = (
    df.groupby(['VEHICLE','SECTION','VARIANT'], as_index=False)
      .agg(
        min_date = ('DATE','min'),
        max_date = ('DATE','max'),
        sum_days_spent = ('NUMBER_DAYS_SPENT','sum')
      )
)

# Compute total calendar days from min to max
grouped['calendar_days'] = (grouped['max_date'] - grouped['min_date']).dt.days + 1

# This “grouped” DataFrame is your “historical baseline” by section.
# Some vehicles may appear multiple times if they re-enter a section,
# but typically you’ll have 1 row per (vehicle-section-variant).

# 3) VISUALIZE: BOX PLOT OF TOTAL DAYS BY SECTION
#    --------------------------------------------
# We'll visualize "calendar_days" by section as an example
plt.figure(figsize=(8,6))
sns.boxplot(data=grouped, x='SECTION', y='calendar_days')
plt.title('Box Plot of Total Calendar Days by SECTION')
plt.xlabel('SECTION')
plt.ylabel('Total Calendar Days (min->max)')
plt.tight_layout()
plt.show()

# 4) BUILD A PREDICTIVE MODEL
#    ------------------------
# We want to predict how many total calendar days a vehicle might spend
# in a given SECTION, given (SECTION, VARIANT). We could also incorporate
# sum_days_spent or other features as well, but let’s keep it simple.

# Prepare modeling DataFrame
model_df = grouped[['SECTION','VARIANT','calendar_days']].copy()

# Encode the categorical features SECTION, VARIANT
section_encoder = LabelEncoder()
variant_encoder = LabelEncoder()

model_df['SECTION_enc'] = section_encoder.fit_transform(model_df['SECTION'])
model_df['VARIANT_enc'] = variant_encoder.fit_transform(model_df['VARIANT'])

# Features X and target y
X = model_df[['SECTION_enc','VARIANT_enc']]
y = model_df['calendar_days']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Train a RandomForestRegressor (example choice)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# 5) VALIDATION
#    ----------
# Predict on the test set and compute a simple MAE
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test Mean Absolute Error: {mae:.2f} days")

# 6) EXAMPLE PREDICTION FUNCTION
#    ---------------------------
def predict_section_days(section_input, variant_input):
    """
    Given a SECTION and VARIANT, predict how many calendar days it might take
    (based on our historical baseline model).
    """
    # Encode using the same label encoders
    if section_input in section_encoder.classes_:
        sec_val = section_encoder.transform([section_input])[0]
    else:
        # If new/unseen, fallback or handle
        sec_val = 0
    
    if variant_input in variant_encoder.classes_:
        var_val = variant_encoder.transform([variant_input])[0]
    else:
        var_val = 0

    # Predict
    X_new = np.array([[sec_val, var_val]])
    days_pred = reg.predict(X_new)[0]
    return days_pred

# Example usage:
test_section = "Vehicle"  # e.g., 'Vehicle' if that's one of your sections
test_variant = "A"
predicted_calendar_days = predict_section_days(test_section, test_variant)
print(f"Predicted days for SECTION={test_section}, VARIANT={test_variant}: "
      f"{predicted_calendar_days:.1f}")