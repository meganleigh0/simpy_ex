import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# === 1. Plot and return snapshot completion stats ===
def plot_bom_completion_by_make_buy(combined_df, snapshot_date):
    bars = []
    records = []

    for source, match_col in [('TeamCenter', 'Match_EBOM_MBOM_TC'), ('Oracle', 'Match_EBOM_MBOM_Oracle')]:
        for mob in ['Make', 'Buy']:
            ebom_filtered = combined_df[combined_df['Make or Buy_ebom'] == mob]
            total_parts = ebom_filtered['PART_NUMBER'].nunique()
            matched_parts = ebom_filtered[ebom_filtered[match_col]]['PART_NUMBER'].nunique()
            percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

            label = f'{source} - {mob}'
            bars.append({'label': label, 'value': percent_matched})
            records.append({
                'snapshot_date': snapshot_date,
                'source': source,
                'make_or_buy': mob,
                'percent_matched': percent_matched
            })

    # Plot snapshot chart
    fig = go.Figure(data=[
        go.Bar(x=[bar['label']], y=[bar['value']], name=bar['label']) for bar in bars
    ])
    fig.update_layout(
        title=f'MBOM Completion by Make/Buy vs EBOM - {snapshot_date}',
        yaxis_title='% of EBOM Parts Matched',
        xaxis_title='Source - Make or Buy',
        yaxis=dict(range=[0, 100]),
        barmode='group'
    )
    fig.show()

    return pd.DataFrame(records)


# === 2. Save snapshot to CSV (or Delta if needed) ===
def save_bom_completion_snapshot(snapshot_df, log_path="/Volumes/your_catalog/your_schema/bom_completion_log.csv"):
    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        combined_df = pd.concat([log_df, snapshot_df], ignore_index=True)
    else:
        combined_df = snapshot_df

    combined_df.to_csv(log_path, index=False)


# === 3. Plot historical progress over time ===
def plot_bom_completion_progress_over_time(log_path="/Volumes/your_catalog/your_schema/bom_completion_log.csv"):
    if not os.path.exists(log_path):
        print("No history log found.")
        return

    df = pd.read_csv(log_path)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

    fig = go.Figure()
    for (source, mob), group_df in df.groupby(['source', 'make_or_buy']):
        label = f"{source} - {mob}"
        fig.add_trace(go.Scatter(
            x=group_df['snapshot_date'],
            y=group_df['percent_matched'],
            mode='lines+markers',
            name=label
        ))

    fig.update_layout(
        title='MBOM Completion Progress Over Time by Make/Buy',
        xaxis_title='Date',
        yaxis_title='% of EBOM Parts Matched',
        yaxis=dict(range=[0, 100])
    )
    fig.show()