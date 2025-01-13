# Suppose each station is parallel, and total build time = maximum station time
# among all stations for that vehicle. We'll do this for one chosen model
# (e.g., RandomForest).

chosen_model = "RandomForest"
preds_for_chosen = predictions_dict[chosen_model]

# Group by vehicle, get the max predicted days across stations in the test set
max_times = preds_for_chosen.groupby("Vehicle")["PredDays"].max().reset_index()
max_times.columns = ["Vehicle", "MaxPredictedDays"]
print(f"Max predicted days per vehicle (Model={chosen_model}):")
print(max_times)
print()

# If it was a sequential process, you'd do sum() instead of max().

# ---------------------------
#   E) INTERACTIVE PLOT (STATION & MODEL DROPDOWNS)
# ---------------------------
# We'll create a separate trace for each (Station, Model).
# Then 2 dropdowns: one picks STATION, one picks MODEL.
# We set 'visible' to True only if the trace matches both selected station & model.

# 1) Collect all unique stations
unique_stations = sorted(stations_test.unique())
model_names = list(models.keys())

# We'll build a list of traces for Plotly. Each trace is a line of TrueDays vs. index,
# plus a line of PredDays vs. index. Actually, to keep it simpler:
# We'll do *one trace* per (Station, Model) with lines+markers for True & Pred together.
# Or we'll do two traces per combination. But let's do two lines (True vs. Pred) in one "combo."

fig = go.Figure()

trace_indices = {}  # Will store (station, model) -> [trace_index_true, trace_index_pred]

for stn in unique_stations:
    for mdl in model_names:
        # Filter predictions_dict[mdl] to station=stn
        subset = predictions_dict[mdl][predictions_dict[mdl]["Station"] == stn].copy()
        # We'll sort by Vehicle or index just for consistent plotting
        subset = subset.sort_values("Vehicle").reset_index(drop=True)

        # Add "TrueDays" trace
        trace_true = go.Scatter(
            x=subset.index,  # or subset["Vehicle"]
            y=subset["TrueDays"],
            mode='lines+markers',
            name=f"{stn}-True-{mdl}",
            visible=False  # we'll control visibility with dropdown
        )
        # Add "PredDays" trace
        trace_pred = go.Scatter(
            x=subset.index,
            y=subset["PredDays"],
            mode='lines+markers',
            name=f"{stn}-Pred-{mdl}",
            visible=False
        )

        # Add them to figure
        idx_true = len(fig.data)
        fig.add_trace(trace_true)
        idx_pred = len(fig.data)
        fig.add_trace(trace_pred)

        # Store the pair of indices
        trace_indices[(stn, mdl)] = (idx_true, idx_pred)

# 2) Build the 'visible' matrix for each station and model choice
def make_visible_arrays(selected_station, selected_model):
    """Return a list of booleans, specifying which traces are visible under
    the current (station, model) selection."""
    n_traces = len(fig.data)
    visible = [False] * n_traces

    # Get the trace indices for that combination
    idx_true, idx_pred = trace_indices[(selected_station, selected_model)]
    # Mark them visible
    visible[idx_true] = True
    visible[idx_pred] = True

    return visible

# We'll initialize with the first station and model
initial_station = unique_stations[0]
initial_model = model_names[0]
initial_visibility = make_visible_arrays(initial_station, initial_model)
for i, v in enumerate(initial_visibility):
    fig.data[i].visible = v

# 3) Build the dropdowns for Station and Model
station_buttons = []
for stn in unique_stations:
    station_buttons.append(
        dict(
            label=stn,
            method="update",
            args=[
                # We'll keep the model the same, only station changes.
                {"visible": make_visible_arrays(stn, initial_model)},
                {"title": f"Station={stn}, Model={initial_model}"}
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
                {"visible": make_visible_arrays(initial_station, mdl)},
                {"title": f"Station={initial_station}, Model={mdl}"}
            ]
        )
    )

# 4) We'll create two updatemenus (one for station, one for model)
updatemenus = [
    dict(
        buttons=station_buttons,
        direction="down",
        showactive=True,
        x=0.0,
        y=1.15,
        xanchor="left",
        yanchor="top",
        pad={"r": 10}
    ),
    dict(
        buttons=model_buttons,
        direction="down",
        showactive=True,
        x=0.25,
        y=1.15,
        xanchor="left",
        yanchor="top",
        pad={"r": 10}
    )
]

fig.update_layout(
    updatemenus=updatemenus,
    title=f"Station={initial_station}, Model={initial_model}",
    xaxis_title="Index (sorted by Vehicle)",
    yaxis_title="Days at Station"
)

fig.show()
