import pandas as pd
import plotly.express as px

# ===== 1) Summarize CURRENT_DATA by variant =====
# Count how many distinct VEHICLEs (or rows) exist per variant.
current_counts = (
    CURRENT_DATA
    .groupby('VARIANT')['VEHICLE']
    .nunique()  # or .count(), depending on your logic
    .reset_index(name='count_current')
)

# ===== 2) Convert RECORVER_SCHEDULE from *cumulative* to *monthly increments* =====
df_schedule = RECORVER_SCHEDULE.copy()

# Identify the month columns (excluding 'VARIANT')
month_cols = [c for c in df_schedule.columns if c != 'VARIANT']

# (Optional) Ensure month_cols are in chronological order, e.g. if columns are strings like 'Jan 2025', 'Feb 2025'...
# month_cols = sorted(month_cols, key=lambda x: pd.to_datetime(x, format='%b %Y'))

# For each variant, transform each month’s *cumulative* number into that month’s *increment*
for i, col in enumerate(month_cols):
    if i == 0:
        # The first column (e.g. January) stays as is, because there's no previous month
        continue
    else:
        # Subtract the previous month’s cumulative count to get just the increment
        df_schedule[col] = df_schedule[col] - df_schedule[month_cols[i - 1]]

# ===== 3) Sum across all monthly increments to get each variant’s total annual schedule =====
df_schedule['count_schedule'] = df_schedule[month_cols].sum(axis=1)

# ===== 4) Merge current_counts with df_schedule =====
compare_df = pd.merge(
    current_counts,
    df_schedule[['VARIANT', 'count_schedule']],  # we only need these two columns from df_schedule
    on='VARIANT', 
    how='outer'
).fillna(0)

# ===== 5) Plotly grouped bar chart comparing current vs. scheduled =====
fig = px.bar(
    compare_df,
    x='VARIANT',
    y=['count_current', 'count_schedule'],
    barmode='group',
    title='Current Production vs. Yearly Schedule (using monthly increments)'
)

fig.update_layout(
    xaxis_title='Variant',
    yaxis_title='Number of Vehicles',
    legend_title='',
    template='plotly_white'
)

fig.show()
