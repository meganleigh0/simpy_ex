import plotly.graph_objects as go

def plot_bom_comparison_snapshot(combined_df, snapshot_date):
    total_ebom_parts = combined_df['PART_NUMBER'].nunique()
    matched_mbom_tc = combined_df['Match_EBOM_MBOM_TC'].sum()
    matched_mbom_oracle = combined_df['Match_EBOM_MBOM_Oracle'].sum()

    fig = go.Figure(data=[
        go.Bar(name='MBOM TeamCenter', x=[snapshot_date], y=[matched_mbom_tc / total_ebom_parts * 100]),
        go.Bar(name='MBOM Oracle', x=[snapshot_date], y=[matched_mbom_oracle / total_ebom_parts * 100])
    ])
    fig.update_layout(
        title='MBOM Coverage vs EBOM',
        yaxis_title='% of EBOM Parts Matched',
        barmode='group'
    )
    fig.show()
    
    
plot_bom_comparison_snapshot(combined_df, "2025-03-27")  # example date

def extract_weekly_metrics(combined_df, snapshot_date):
    total_parts = combined_df['PART_NUMBER'].nunique()
    matched_tc = combined_df['Match_EBOM_MBOM_TC'].sum()
    matched_oracle = combined_df['Match_EBOM_MBOM_Oracle'].sum()

    return {
        "snapshot_date": snapshot_date,
        "total_ebom_parts": total_parts,
        "matched_mbom_tc": matched_tc,
        "matched_mbom_oracle": matched_oracle,
        "percent_mbom_tc": matched_tc / total_parts * 100,
        "percent_mbom_oracle": matched_oracle / total_parts * 100
    }
    
import pandas as pd
import os

def save_weekly_metrics(metrics, file_path='mbom_progress_tracking.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=metrics.keys())
    
    df = df.append(metrics, ignore_index=True)
    df.to_csv(file_path, index=False)
    
    
def plot_progress_over_time(file_path='mbom_progress_tracking.csv'):
    df = pd.read_csv(file_path)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['snapshot_date'], y=df['percent_mbom_tc'],
                             mode='lines+markers', name='MBOM TeamCenter'))
    fig.add_trace(go.Scatter(x=df['snapshot_date'], y=df['percent_mbom_oracle'],
                             mode='lines+markers', name='MBOM Oracle'))

    fig.update_layout(
        title='MBOM Coverage Progress Over Time',
        xaxis_title='Date',
        yaxis_title='% of EBOM Parts Matched',
        yaxis_range=[0, 100]
    )
    fig.show()
    
    
metrics = extract_weekly_metrics(combined_df, "YYYY-MM-DD")
save_weekly_metrics(metrics)





import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# === CONFIG ===
snapshot_date = datetime.now().strftime("%Y-%m-%d")
output_path = "/Volumes/your_catalog/your_schema/mbom_progress_tracking.csv"  # <-- update this path
make_or_buy_filter = None  # set to "Make" or "Buy" to filter, or None for all

# === INPUT: COMBINED_DF from compare_bom_data() ===
# Assumes `combined_df` is already created with columns:
# ['PART_NUMBER', 'Quantity_ebom', 'Make or Buy_ebom', 'Quantity_mbom_tc', 'Make or Buy_mbom_tc',
#  'Quantity', 'Make or Buy', 'Match_EBOM_MBOM_TC', 'Match_EBOM_MBOM_Oracle', 'Match_MBOM_TC_MBOM_Oracle']

# === 1. Apply Make/Buy Filter if needed ===
if make_or_buy_filter:
    filtered_df = combined_df[combined_df['Make or Buy_ebom'] == make_or_buy_filter]
else:
    filtered_df = combined_df.copy()

# === 2. Calculate Weekly Metrics ===
total_parts = filtered_df['PART_NUMBER'].nunique()
matched_tc = filtered_df['Match_EBOM_MBOM_TC'].sum()
matched_oracle = filtered_df['Match_EBOM_MBOM_Oracle'].sum()

metrics = {
    "snapshot_date": snapshot_date,
    "make_or_buy": make_or_buy_filter or "All",
    "total_ebom_parts": total_parts,
    "matched_mbom_tc": matched_tc,
    "matched_mbom_oracle": matched_oracle,
    "percent_mbom_tc": matched_tc / total_parts * 100 if total_parts > 0 else 0,
    "percent_mbom_oracle": matched_oracle / total_parts * 100 if total_parts > 0 else 0
}

# === 3. Save Metrics to Volume (DBFS) ===
if os.path.exists(output_path):
    history_df = pd.read_csv(output_path)
else:
    history_df = pd.DataFrame(columns=metrics.keys())

history_df = pd.concat([history_df, pd.DataFrame([metrics])], ignore_index=True)
history_df.to_csv(output_path, index=False)

# === 4. Plot Snapshot Bar (Current Week) ===
fig1 = go.Figure(data=[
    go.Bar(name='MBOM TeamCenter', x=[snapshot_date], y=[metrics['percent_mbom_tc']]),
    go.Bar(name='MBOM Oracle', x=[snapshot_date], y=[metrics['percent_mbom_oracle']])
])
fig1.update_layout(
    title=f'MBOM Coverage vs EBOM ({metrics["make_or_buy"]})',
    yaxis_title='% of EBOM Parts Matched',
    barmode='group'
)
fig1.show()

# === 5. Plot Historical Trend ===
history_df['snapshot_date'] = pd.to_datetime(history_df['snapshot_date'])
filtered_history = history_df[history_df['make_or_buy'] == metrics["make_or_buy"]]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=filtered_history['snapshot_date'], y=filtered_history['percent_mbom_tc'],
                          mode='lines+markers', name='MBOM TeamCenter'))
fig2.add_trace(go.Scatter(x=filtered_history['snapshot_date'], y=filtered_history['percent_mbom_oracle'],
                          mode='lines+markers', name='MBOM Oracle'))
fig2.update_layout(
    title=f'MBOM Progress Over Time ({metrics["make_or_buy"]})',
    xaxis_title='Date',
    yaxis_title='% of EBOM Parts Matched',
    yaxis_range=[0, 100]
)
fig2.show()


