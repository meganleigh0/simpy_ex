# ---------------------------------------------------------
# EVMS CHART (HTML ONLY) – FIXED FOR ARRAY-SAFE OPERATIONS
# ---------------------------------------------------------

# Monthly rollup by month-period
month_totals = (
    cobra.groupby([cobra[DATE_COL].dt.to_period("M"), COSTSET_COL])[HOURS_COL]
         .sum()
         .unstack(fill_value=0.0)
)

# Ensure all cost sets exist
for cs in COST_SETS:
    if cs not in month_totals.columns:
        month_totals[cs] = 0.0

ev = month_totals.sort_index().copy()

# Vector-safe divisions
ev["CPI_month"] = np.where(ev["ACWP"] == 0, np.nan, ev["BCWP"] / ev["ACWP"])
ev["SPI_month"] = np.where(ev["BCWS"] == 0, np.nan, ev["BCWP"] / ev["BCWS"])

# Cumulative cost sets
ev["ACWP_cum"] = ev["ACWP"].cumsum()
ev["BCWP_cum"] = ev["BCWP"].cumsum()
ev["BCWS_cum"] = ev["BCWS"].cumsum()

# Vector-safe cumulative indices
ev["CPI_cum"] = np.where(ev["ACWP_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["ACWP_cum"])
ev["SPI_cum"] = np.where(ev["BCWS_cum"] == 0, np.nan, ev["BCWP_cum"] / ev["BCWS_cum"])

# Convert month-period index to string
ev["Month"] = ev.index.to_timestamp().strftime("%b-%y")

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["CPI_month"], name="Monthly CPI", mode="markers"))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["SPI_month"], name="Monthly SPI", mode="markers"))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["CPI_cum"], name="Cumulative CPI", mode="lines"))
fig.add_trace(go.Scatter(x=ev["Month"], y=ev["SPI_cum"], name="Cumulative SPI", mode="lines"))

fig.update_layout(title="EV Indices (SPI / CPI)", template="plotly_white")

# Save HTML only
fig.write_html(EV_PLOT_HTML)