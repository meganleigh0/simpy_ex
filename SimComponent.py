import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

def log_and_plot_bom_completion_over_time(combined_df, log_path="/Volumes/your_catalog/your_schema/bom_completion_log.csv"):
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    
    # Build snapshot records
    records = []
    sources = {
        'TeamCenter': 'Match_EBOM_MBOM_TC',
        'Oracle': 'Match_EBOM_MBOM_Oracle'
    }

    for source_name, match_col in sources.items():
        for mob in ['Make', 'Buy']:
            ebom_filtered = combined_df[combined_df['Make or Buy_ebom'] == mob]
            total_parts = ebom_filtered['PART_NUMBER'].nunique()
            matched_parts = ebom_filtered[ebom_filtered[match_col]]['PART_NUMBER'].nunique()
            percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

            records.append({
                'snapshot_date': snapshot_date,
                'source': source_name,
                'make_or_buy': mob,
                'percent_matched': percent_matched
            })

    snapshot_df = pd.DataFrame(records)

    # Append to historical log
    if os.path.exists(log_path):
        history_df = pd.read_csv(log_path)
        history_df = pd.concat([history_df, snapshot_df], ignore_index=True)
    else:
        history_df = snapshot_df

    # Save updated log
    history_df.to_csv(log_path, index=False)

    # Plot trend over time
    history_df['snapshot_date'] = pd.to_datetime(history_df['snapshot_date'])
    fig = go.Figure()

    for (source, mob), group_df in history_df.groupby(['source', 'make_or_buy']):
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