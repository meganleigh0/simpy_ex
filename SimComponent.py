import pandas as pd
import plotly.express as px
from datetime import date

# ------------------------------------------------------------------
# 1) Simulate / Load Your Main Data Frame
# ------------------------------------------------------------------
# In practice, replace this with your real loading logic, e.g.:
# df = pd.read_csv("floor_data.csv") or a database query
data = {
    'vehicle': ['VIN100','VIN101','VIN102','VIN103','VIN104','VIN105','VIN106'],
    'section': ['Paint','Paint','Assembly','Assembly','Weld','Weld','Weld'],
    'station': ['Station A','Station A','Station B','Station C','Station D','Station D','Station D'],
    'date': [
        '2025-01-31','2025-01-31','2025-01-31',
        '2025-01-31','2025-01-31','2025-01-31','2025-01-31'
    ],
    'variant': ['Program D','Program D','Program A','Program C','Program D','Program A','Program B'],
    'number':  [ 32,         33,         11,         7,         35,         19,         4         ]
}
df = pd.DataFrame(data)

# Convert 'date' to an actual date type for filtering
df['date'] = pd.to_datetime(df['date']).dt.date

# ------------------------------------------------------------------
# 2) Filter to "What's on the floor TODAY"
# ------------------------------------------------------------------
today = date.today()  # Real-time
df_today = df[df['date'] == today].copy()

# ------------------------------------------------------------------
# 3) Compute Unique Vehicle Counts per (Section, Station, Variant)
# ------------------------------------------------------------------
# If you want the count of unique vehicles in each station/section/variant:
grouped = (
    df_today
    .groupby(['section', 'station', 'variant'])['vehicle']
    .nunique()  # number of distinct vehicles
    .reset_index(name='unique_vehicle_count')
)

# ------------------------------------------------------------------
# 4) Plotly Visualization
#    - We facet by SECTION so each "section" is a separate panel.
#    - X-axis is STATION, color is VARIANT, and the height is the count of unique vehicles.
# ------------------------------------------------------------------
fig = px.bar(
    grouped,
    x='station',
    y='unique_vehicle_count',
    color='variant',
    facet_col='section',            # one subplot per section
    barmode='group',                # group bars side-by-side
    text='unique_vehicle_count',    # show counts on bars
    title=f"Unique Variant Counts per Station/Section for {today.isoformat()}"
)

# Optional styling tweaks:
fig.update_layout(
    font=dict(size=12),
    paper_bgcolor='white',
    plot_bgcolor='white',
    height=600,
    legend_title_text='Variant'
)
fig.update_xaxes(title="Station")
fig.update_yaxes(title="Count of Unique Vehicles")
fig.show()