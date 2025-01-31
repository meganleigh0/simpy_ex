import plotly.graph_objects as go

# Extract unique sections
unique_sections = CURRENT_DATA['SECTION'].unique()

# Create figure
fig = go.Figure()

# Add one scatter trace per SECTION
for i, section in enumerate(unique_sections):
    subset = CURRENT_DATA[CURRENT_DATA['SECTION'] == section]
    
    fig.add_trace(
        go.Scatter(
            x=subset['STATION'],
            y=subset['NUMBER'],
            mode='markers',
            name=f"Section {section}",
            # Only show the first section by default
            visible=True if i == 0 else False
        )
    )

# Build dropdown buttons, one per SECTION
buttons = []
for i, section in enumerate(unique_sections):
    # For each button, we turn on only the ith trace (and turn off the others).
    visible_array = [False] * len(unique_sections)
    visible_array[i] = True
    
    buttons.append(
        dict(
            label=f"Section {section}",
            method="update",
            args=[
                {"visible": visible_array},
                {"title": f"Current Data for Section {section}"}
            ]
        )
    )

# Add dropdown to figure
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=buttons,
            x=0.0,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )
    ],
    title=f"Current Data for Section {unique_sections[0]}",
    xaxis_title="Station",
    yaxis_title="Number"
)

fig.show()
