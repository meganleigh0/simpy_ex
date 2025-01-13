###############################################################################
# Single-cell example: 
# Train multiple regression models to predict how many days (Number) a 
# vehicle variant spends at each station, then visualize predictions with Plotly.
###############################################################################

import pandas as pd
import numpy as np

# 1. Data Loading and Preparation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 2. Models and Metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# 3. Visualization
import plotly.graph_objects as go

# ---------------------------
#   A) LOAD YOUR DATA
# ---------------------------
# Replace with your real data file or DataFrame
df = pd.DataFrame({
    "Vehicle": ["V1","V1","V2","V2","V3","V3"],
    "Section": ["S1","S1","S2","S2","S1","S2"],
    "Station": ["StA","StB","StA","StB","StB","StA"],
    "Date": pd.date_range("2021-01-01", periods=6, freq="D"),
    "Variant": ["Type1","Type1","Type2","Type2","Type1","Type2"],
    "Number": [3, 5, 2, 6, 8, 4]  # Number of days the vehicle was at the Station
})

# Convert Date to datetime (already done above, but here for completeness)
df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------
#   B) FEATURE ENGINEERING
# ---------------------------
# Let's assume "Number" (the number of days) is our target.
# We'll encode categorical features. For brevity, we’ll keep only 
# relevant columns for modeling and drop or ignore others as an example.

# Example: encode "Variant" and "Station" using one-hot encoding.
cat_features = ["Variant", "Station"]
encoder = OneHotEncoder(drop='first', sparse=False)
encoded = encoder.fit_transform(df[cat_features])

# Create a new DataFrame for the encoded columns
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_features))

# Combine original DataFrame with the encoded data
df_encoded = pd.concat([df, encoded_df], axis=1)

# Define X (features) and y (target)
# Here, y is the "Number" of days at the station
y = df_encoded["Number"]

# We'll drop columns that are not suitable as direct features (Vehicle, Section, Date, etc.)
# You might keep "Date" if you want to engineer day-of-week/month features, etc.
X = df_encoded.drop(["Number", "Vehicle", "Section", "Date", "Variant", "Station"], axis=1)

# Train/test split (random for example; if time-based, use chronological split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
#   C) MODEL TRAINING
# ---------------------------
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=50, use_label_encoder=False, eval_metric='rmse', random_state=42),
    "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(16,8), max_iter=500, random_state=42)
}

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    results[model_name] = {"MSE": mse, "R^2": r2}

results_df = pd.DataFrame(results).T
print("Model Comparison:")
print(results_df)

# ---------------------------
#   D) PREDICTIONS FOR PLOTTING
# ---------------------------
# Store each model's predictions in a DataFrame
predictions_df = pd.DataFrame({
    "TrueDays": y_test.reset_index(drop=True),
    "Pred_Linear": models["Linear"].predict(X_test),
    "Pred_RF": models["RandomForest"].predict(X_test),
    "Pred_XGBoost": models["XGBoost"].predict(X_test),
    "Pred_NeuralNetwork": models["NeuralNetwork"].predict(X_test),
})

# ---------------------------
#   E) PLOTLY VISUALIZATION WITH DROPDOWN
# ---------------------------
fig = go.Figure()

# Always show the "TrueDays" trace
fig.add_trace(go.Scatter(
    x=predictions_df.index,
    y=predictions_df["TrueDays"],
    mode='lines',
    name='True Days'
))

# Add a trace for each model; make them initially invisible
model_cols = ["Pred_Linear", "Pred_RF", "Pred_XGBoost", "Pred_NeuralNetwork"]
for col in model_cols:
    fig.add_trace(go.Scatter(
        x=predictions_df.index,
        y=predictions_df[col],
        mode='lines',
        name=col,
        visible=False
    ))

# Create dropdown that toggles which model's prediction trace is visible
updatemenus = [
    dict(
        buttons=[
            dict(label="Linear", 
                 method="update",
                 args=[{"visible": [True, True, False, False, False]}, 
                       {"title": "Linear vs. True"}]),
            dict(label="RandomForest", 
                 method="update",
                 args=[{"visible": [True, False, True, False, False]}, 
                       {"title": "RandomForest vs. True"}]),
            dict(label="XGBoost", 
                 method="update",
                 args=[{"visible": [True, False, False, True, False]}, 
                       {"title": "XGBoost vs. True"}]),
            dict(label="NeuralNetwork", 
                 method="update",
                 args=[{"visible": [True, False, False, False, True]}, 
                       {"title": "NeuralNetwork vs. True"}]),
        ],
        direction="down",
        showactive=True,
        x=0,  # x-position of the dropdown
        y=1.1 # y-position of the dropdown
    )
]

fig.update_layout(
    updatemenus=updatemenus,
    title="Days at Station: Predictions vs. True",
    xaxis_title="Test Sample Index",
    yaxis_title="Number of Days"
)

fig.show()
