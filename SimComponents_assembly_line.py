import pandas as pd
import plotly.express as px

# ===== 1) Summarize CURRENT_DATA by variant  =====
# We'll count how many distinct VEHICLEs are currently on the floor per variant
current_counts = (
    CURRENT_DATA
    .groupby('VARIANT')['VEHICLE']
    .nunique()  # or .count(), depending on your preference
    .reset_index(name='count_current')
)

# ===== 2) Summarize RECORVER_SCHEDULE by variant across months =====
# Melt the schedule to go from wide (Jan 2025, Feb 2025, ...) to long format
schedule_long = pd.melt(
    RECORVER_SCHEDULE,
    id_vars='VARIANT',
    var_name='Month',
    value_name='Monthly_Schedule'
)

# Sum across all months to get the total yearly schedule per variant
schedule_sums = (
    schedule_long
    .groupby('VARIANT')['Monthly_Schedule']
    .sum()
    .reset_index(name='count_schedule')
)

# ===== 3) Merge current_counts with schedule_sums =====
compare_df = pd.merge(current_counts, schedule_sums, on='VARIANT', how='outer').fillna(0)

# ===== 4) Plotly grouped bar chart =====
fig = px.bar(
    compare_df,
    x='VARIANT',
    y=['count_current', 'count_schedule'],
    barmode='group',
    title='Current Production Status vs. Yearly Schedule'
)

fig.update_layout(
    xaxis_title='Variant',
    yaxis_title='Number of Vehicles',
    legend_title='',
    template='plotly_white'
)

fig.show()

