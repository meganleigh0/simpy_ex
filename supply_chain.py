###############################################################################
# SINGLE-CELL COMPREHENSIVE EXAMPLE
# 1) Train multiple models on data including both Station & Variant.
# 2) Plotly figure with triple dropdown: Station, Variant, Model.
#    Shows True vs. Predicted "days at station" for that subset.
# 3) Second Plotly bar chart: Multiply "max station time" by user-defined
#    quantity for each Variant => "Total Predicted Days".
###############################################################################

import pandas as pd
import numpy as np

# Modeling imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Plotly for interactive visuals
import plotly.graph_objects as go

# ---------------------------
#   A) CREATE / LOAD DATA
# ---------------------------
# Example data with multiple variants & stations:
df = pd.DataFrame({
    "Vehicle":  ["V1","V1","V1","V1","V2","V2","V2","V2","V3","V3","V3","V3"],
    "Section":  ["S1","S1","S2","S3","S1","S1","S2","S3","S1","S2","S3","S3"],
    "Station":  ["StA","StB","StC","StD","StA","StB","StC","StD","StA","StC","StD","StE"],
    "Date":     pd.date_range("2021-01-01", periods=12, freq="D"),
    "Variant":  ["Type1","Type1","Type1","Type1",
                 "Type2","Type2","Type2","Type2",
                 "Type1","Type1","Type2","Type2"],
    "Number":   [ 3,   5,   2,   4,   8,   6,   10, 12,    7,   9,   3,   15]
})

df["Date"] = pd.to_datetime(df["Date"])

# ---------------------------
#   B) FEATURE ENGINEERING
# ---------------------------
# We want to predict "Number" (days at station).
# We'll encode both 'Station' and 'Variant' so the models can learn from both.
cat_features = ["Station", "Variant"]

# OneHotEncoder (for sklearn 0.24.2, use get_feature_names not get_feature_names_out)
encoder = OneHotEncoder(drop='first', sparse=False)
encoded = encoder.fit_transform(df[cat_features])
feature_names = encoder.get_feature_names(cat_features)
encoded_df = pd.DataFrame(encoded, columns=feature_names)

df_encoded = pd.concat([df, encoded_df], axis=1)

# Our target
y = df_encoded["Number"]

# We'll drop columns that are non-numeric or not used directly as features
X = df_encoded.drop(["Number", "Vehicle", "Section", "Date", "Station", "Variant"], axis=1)

# We'll keep track of Station & Variant in separate arrays for filtering / plotting
stations = df_encoded["Station"]
variants = df_encoded["Variant"]

# Train/test split
X_train, X_test, y_train, y_test, stations_train, stations_test, variants_train, variants_test = train_test_split(
    X, y, stations, variants, test_size=0.5, random_state=42
)

# ---------------------------
#   C) TRAIN MULTIPLE MODELS
# ---------------------------
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=20, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=20, use_label_encoder=False, eval_metric='rmse', random_state=42),
    "NeuralNet": MLPRegressor(hidden_layer_sizes=(16,), max_iter=200, random_state=42)
}

results = {}
predictions_dict = {}  # We'll store a DataFrame of test predictions for each model

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2  = r2_score(y_test, y_pred)
    results[model_name] = {"RMSE": rmse, "R^2": r2}

    # Build a DF that includes Station, Variant, TrueDays, PredDays for test set
    preds_df = pd.DataFrame({
        "Station": stations_test.values,
        "Variant": variants_test.values,
        "TrueDays": y_test.values,
        "PredDays": y_pred
    })
    predictions_dict[model_name] = preds_df

results_df = pd.DataFrame(results).T
print("Model Comparison (RMSE, R^2):\n", results_df, "\n")

# If the predictions look too constant, it may be due to limited data or 
# insufficient variability in features. Including both Station & Variant 
# helps, but real-world data should be larger & more detailed.

# ---------------------------
#   D) INTERACTIVE PLOT:
#      Triple dropdown: Station, Variant, Model
# ---------------------------
# We'll create two traces (True, Pred) for each (Station, Variant, Model).
# Then have 3 dropdowns letting the user pick which combination is shown.

unique_stations = sorted(stations_test.unique())
unique_variants = sorted(variants_test.unique())
model_names = list(models.keys())

fig = go.Figure()
trace_indices = {}  # Maps (station, variant, model) -> (idx_true, idx_pred)

for stn in unique_stations:
    for var in unique_variants:
        for mdl in model_names:
            # Filter the predictions for that model, station, and variant
            subset = predictions_dict[mdl]
            mask = (subset["Station"] == stn) & (subset["Variant"] == var)
            sub = subset[mask].reset_index(drop=True)

            # Sort by index or anything relevant
            # We'll just keep the order they appear
            idx_true = len(fig.data)
            fig.add_trace(go.Scatter(
                x=sub.index,
                y=sub["TrueDays"],
                mode='lines+markers',
                name=f"{stn}-{var}-True-{mdl}",
                visible=False
            ))
            idx_pred = len(fig.data)
            fig.add_trace(go.Scatter(
                x=sub.index,
                y=sub["PredDays"],
                mode='lines+markers',
                name=f"{stn}-{var}-Pred-{mdl}",
                visible=False
            ))
            trace_indices[(stn, var, mdl)] = (idx_true, idx_pred)

def make_visible_arrays(selected_station, selected_variant, selected_model):
    """Return a boolean list controlling which traces are visible."""
    n_traces = len(fig.data)
    visible = [False] * n_traces
    # Activate the pair for the selected combination
    (idx_true, idx_pred) = trace_indices[(selected_station, selected_variant, selected_model)]
    visible[idx_true] = True
    visible[idx_pred] = True
    return visible

# Initialize with the first station, variant, model
initial_station = unique_stations[0]
initial_variant = unique_variants[0]
initial_model = model_names[0]
initial_visibility = make_visible_arrays(initial_station, initial_variant, initial_model)
for i, v in enumerate(initial_visibility):
    fig.data[i].visible = v

# Build the dropdowns
station_buttons = []
for stn in unique_stations:
    station_buttons.append(
        dict(
            label=stn,
            method="update",
            args=[
                # We keep the same variant/model as initial on station change
                {"visible": make_visible_arrays(stn, initial_variant, initial_model)},
                {"title": f"Station={stn}, Variant={initial_variant}, Model={initial_model}"}
            ]
        )
    )

variant_buttons = []
for var in unique_variants:
    variant_buttons.append(
        dict(
            label=var,
            method="update",
            args=[
                {"visible": make_visible_arrays(initial_station, var, initial_model)},
                {"title": f"Station={initial_station}, Variant={var}, Model={initial_model}"}
            ]
        )
    )

model_buttons = []
for mdl in model_names:
    model_buttons.append(
        dict(
            label=mdl,
            method="update",
            args=[
                {"visible": make_visible_arrays(initial_station, initial_variant, mdl)},
                {"title": f"Station={initial_station}, Variant={initial_variant}, Model={mdl}"}
            ]
        )
    )

updatemenus = [
    dict(
        buttons=station_buttons,
        direction="down",
        showactive=True,
        x=0.0,
        y=1.15,
        xanchor="left",
        yanchor="top"
    ),
    dict(
        buttons=variant_buttons,
        direction="down",
        showactive=True,
        x=0.25,
        y=1.15,
        xanchor="left",
        yanchor="top"
    ),
    dict(
        buttons=model_buttons,
        direction="down",
        showactive=True,
        x=0.50,
        y=1.15,
        xanchor="left",
        yanchor="top"
    )
]

fig.update_layout(
    updatemenus=updatemenus,
    title=f"Station={initial_station}, Variant={initial_variant}, Model={initial_model}",
    xaxis_title="Index",
    yaxis_title="Days (Number)"
)

fig.show()

# ---------------------------
#   E) SECOND CHART:
#      Max station time * quantity for each Variant
# ---------------------------
# 1) Let the user pick which model's predictions to use.
#    (In a real app, you'd let them pick from a dropdown. For simplicity here, 
#     we do a dropdown in the chart or just pick one by default.)
chosen_model = "RandomForest"

df_preds = predictions_dict[chosen_model]

# 2) Suppose the user provides a dictionary of how many units for each variant
qty_dict = {
    "Type1": 5,
    "Type2": 2
}

# 3) Compute each vehicle's bottleneck (max station time).
#    Then group by variant to get an average or max across vehicles of that variant.
#    We'll do average here:
df_bottleneck = df_preds.groupby(["Variant", "Station"]).mean().reset_index()
# Actually, to truly get per-vehicle max, we need "Vehicle" in the test set.
# For simplicity with this small data, let's do a simpler approach:
# we can do groupby(["Variant","Station"]) to get average station time, then
# pick the max across stations for that variant. 
# Real logic depends on how your process is structured.

avg_by_variant_station = df_preds.groupby(["Variant","Station"])["PredDays"].mean().reset_index()
# pivot or groupby to find the max station time per variant
bottleneck_per_variant = avg_by_variant_station.groupby("Variant")["PredDays"].max().reset_index()
bottleneck_per_variant.columns = ["Variant","MaxStationTime"]

# 4) Multiply by user-provided quantity
bottleneck_per_variant["Qty"] = bottleneck_per_variant["Variant"].map(qty_dict).fillna(0)
bottleneck_per_variant["TotalDays"] = bottleneck_per_variant["MaxStationTime"] * bottleneck_per_variant["Qty"]

# 5) Plot a bar chart showing "TotalDays" for each variant
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=bottleneck_per_variant["Variant"],
    y=bottleneck_per_variant["TotalDays"],
    text=(
        "MaxStationTime="
        + bottleneck_per_variant["MaxStationTime"].round(1).astype(str)
        + "<br>Qty="
        + bottleneck_per_variant["Qty"].astype(int).astype(str)
    ),
    hoverinfo="text",
    name="Total Predicted Days"
))

fig2.update_layout(
    title=f"Total Predicted Days (Model={chosen_model})\n(Qty × Max Station Time)",
    xaxis_title="Variant",
    yaxis_title="Total Days",
    showlegend=False
)

fig2.show()
