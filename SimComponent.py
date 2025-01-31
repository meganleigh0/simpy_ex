import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime

# ------------------------------------------------------------------
# 1) SIMULATE LOADING YOUR DATA
# ------------------------------------------------------------------
# Replace with actual code that pulls your data: e.g. df = pd.read_csv("my_report.csv")
data = {
    'Vehicle':  ['VIN111', 'VIN222', 'VIN333', 'VIN444', 'VIN555', 'VIN666', 'VIN777'],
    'Section':  ['Paint', 'Paint', 'Assembly', 'Assembly', 'Weld', 'Weld', 'Assembly'],
    'Station':  ['Station A','Station A','Station B','Station C','Station D','Station D','Station C'],
    'Date':     ['2025-01-31','2025-01-31','2025-01-31','2025-01-31','2025-01-31','2025-01-31','2025-01-31'],
    'Variant':  ['Program X','Program X','Program Y','Program Y','Program Z','Program X','Program Z'],
    'Number':   [  5,          3,          2,          4,          1,          2,          3        ]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date']).dt.date  # Ensure it's a proper date

# ------------------------------------------------------------------
# 2) PER-VARIANT MONTHLY GOALS
# ------------------------------------------------------------------
# Example dictionary of monthly goals (replace with your actual data or a database lookup)
monthly_goals = {
    'Program X': 120,   # e.g. goal for the month
    'Program Y':  90,
    'Program Z': 150,
    # Add more if needed...
}

# ------------------------------------------------------------------
# 3) SEGMENT DATA FOR TODAY AND MONTH-TO-DATE
# ------------------------------------------------------------------
today = date.today()  # In a real environment, this might be the current date
df_today = df[df['Date'] == today]

# Month-to-date: everything from the 1st of this month to current date
first_of_month = today.replace(day=1)
df_month = df[(df['Date'] >= first_of_month) & (df['Date'] <= today)]

# ------------------------------------------------------------------
# 4) COMPUTE KEY SUMMARIES
# ------------------------------------------------------------------
# A) Station distribution (Today's data)
station_dist = df_today.groupby('Station', as_index=False)['Number'].sum()

# B) Monthly progress by variant (merge with monthly goals)
monthly_prog = df_month.groupby('Variant', as_index=False)['Number'].sum()
monthly_prog.rename(columns={'Number': 'CurrentMonthTotal'}, inplace=True)

# Only keep variants that appear in the data or exist in monthly_goals
all_variants = set(monthly_prog['Variant']).union(set(monthly_goals.keys()))
monthly_prog = monthly_prog.set_index('Variant').reindex(all_variants, fill_value=0).reset_index()

# Add a 'Goal' column by mapping from monthly_goals
monthly_prog['Goal'] = monthly_prog['Variant'].map(monthly_goals).fillna(0)

# C) Pie chart data: distribution of variants (Today)
variant_dist_today = df_today.groupby('Variant', as_index=False)['Number'].sum()

# D) Daily trend for the current month (line chart)
daily_trend = (
    df_month.groupby(['Date', 'Variant'], as_index=False)['Number'].sum()
    .sort_values('Date')
)

# ------------------------------------------------------------------
# 5) BUILD MULTI-PANEL DASHBOARD
# ------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Monthly Progress vs. Goals (by Variant)",
        f"Station Distribution (Today: {today.isoformat()})",
        "Variant Distribution (Today)",
        "Daily Trend (Month-to-Date)"
    ],
    specs=[
        [{"type": "xy"},       {"type": "xy"}],      # row 1
        [{"type": "domain"},   {"type": "xy"}]       # row 2
    ]
)

# ---- SUBPLOT (1,1): Bar Chart of Monthly Progress vs Goal ----
fig.add_trace(
    go.Bar(
        x=monthly_prog['Variant'],
        y=monthly_prog['CurrentMonthTotal'],
        name='Current MTD',
        marker_color='royalblue',
        text=monthly_prog['CurrentMonthTotal'],
        textposition='auto'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        x=monthly_prog['Variant'],
        y=monthly_prog['Goal'],
        name='Monthly Goal',
        marker_color='lightgray',
        text=monthly_prog['Goal'],
        textposition='auto'
    ),
    row=1, col=1
)

# Stacked or grouped bars?
# Here we do grouped. If you want stacked, use barmode='stack' in layout.

# ---- SUBPLOT (1,2): Station Distribution (Today) ----
fig.add_trace(
    go.Bar(
        x=station_dist['Station'],
        y=station_dist['Number'],
        marker_color='indianred',
        text=station_dist['Number'],
        textposition='auto',
        name="Station Distribution"
    ),
    row=1, col=2
)

# ---- SUBPLOT (2,1): Pie Chart of Variant Distribution (Today) ----
fig.add_trace(
    go.Pie(
        labels=variant_dist_today['Variant'],
        values=variant_dist_today['Number'],
        hoverinfo='label+value+percent',
        textinfo='value+percent',
        hole=0.4,  # donut style
        marker=dict(line=dict(color='#000000', width=1))
    ),
    row=2, col=1
)

# ---- SUBPLOT (2,2): Daily Trend (Line Chart), Colored by Variant ----
# We’ll build separate lines for each variant
for variant in daily_trend['Variant'].unique():
    df_var = daily_trend[daily_trend['Variant'] == variant]
    fig.add_trace(
        go.Scatter(
            x=df_var['Date'],
            y=df_var['Number'],
            mode='lines+markers',
            name=str(variant),
            text=df_var['Number'],
            textposition='top center'
        ),
        row=2, col=2
    )

# ------------------------------------------------------------------
# 6) UPDATE LAYOUT & DISPLAY
# ------------------------------------------------------------------
fig.update_layout(
    title_text=f"<b>Executive Floor Dashboard</b> - {today.isoformat()}",
    barmode='group',  # so monthly progress vs. goal bars appear side-by-side
    paper_bgcolor='#f9f9f9',
    plot_bgcolor='#f9f9f9',
    font=dict(family="Arial, sans-serif", size=12),
    showlegend=True,
    height=900
)

# Label axes for bar/line subplots
fig.update_xaxes(title_text="Variant", row=1, col=1)
fig.update_yaxes(title_text="Units (Month-to-Date)", row=1, col=1)
fig.update_xaxes(title_text="Station", row=1, col=2)
fig.update_yaxes(title_text="Units (Today)", row=1, col=2)
fig.update_xaxes(title_text="Date", row=2, col=2)
fig.update_yaxes(title_text="Units per Day", row=2, col=2)

fig.show()