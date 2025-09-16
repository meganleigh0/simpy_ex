import plotly.express as px

fig = px.scatter(
    df_anon,
    title="Anonymized Daily Status",
    x="DATE",
    y="VEHICLE_ANON",
    color="STATION_ANON",
)

fig.update_layout(
    width=1200,
    height=800,
    title=dict(x=0.5, xanchor="center", font=dict(size=20)),  # Centered title
    xaxis=dict(
        title="Date",
        tickformat="%b %Y",     # Month-Year formatting
        dtick="M3",             # show ticks every 3 months
        tickangle=-30,          # slanted labels for readability
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
    ),
    yaxis=dict(
        title="Product",
        showgrid=True,
        gridcolor="lightgrey",
        zeroline=False,
    ),
    legend=dict(
        title="Stations",
        orientation="v",
        yanchor="top",
        y=1.0,
        xanchor="left",
        x=1.02,                # move legend outside chart
        bgcolor="rgba(255,255,255,0.6)"
    ),
    plot_bgcolor="white",
    margin=dict(l=150, r=200, t=80, b=80)
)

fig.show()