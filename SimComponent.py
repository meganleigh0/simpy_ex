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
