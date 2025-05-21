import plotly.graph_objects as go

fig = go.Figure()

# Add stacked bar traces for each metric
for metric_name in df_plot['metric'].unique():
    filtered = df_plot[df_plot['metric'] == metric_name]
    fig.add_trace(go.Bar(
        x=filtered["make_or_buy"] + " - " + filtered["source"],
        y=filtered["percent"],
        name=metric_name.replace("percent_", "").capitalize().replace("_", " "),
    ))

# Add line trace for total_parts
# We’ll group it to match the same x-axis
line_df = df_plot.drop_duplicates(subset=["make_or_buy", "source", "total_parts"])
fig.add_trace(go.Scatter(
    x=line_df["make_or_buy"] + " - " + line_df["source"],
    y=line_df["total_parts"],
    mode="lines+markers",
    name="Total Parts",
    yaxis="y2"
))

# Update layout for dual axis and styling
fig.update_layout(
    title=f"Parts Matching Overview - Snapshot {snapshot_to_plot}",
    barmode="stack",
    yaxis=dict(title="Percent Matched", range=[0, 1]),
    yaxis2=dict(title="Total Parts", overlaying="y", side="right"),
    xaxis=dict(title="Make/Buy by Source"),
    legend=dict(x=1.05, y=1),
    height=500
)

fig.show()
