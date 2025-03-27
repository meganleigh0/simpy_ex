import plotly.graph_objects as go
import pandas as pd

# Plot and return snapshot completion stats
def plot_bom_completion_by_make_buy(combined_df, snapshot_date):
    bars = []
    records = []

    for source, match_col in [('TeamCenter', 'Match_EBOM_MBOM_TC'), ('Oracle', 'Match_EBOM_MBOM_Oracle')]:
        for mob in ['Make', 'Buy']:
            ebom_filtered = combined_df[combined_df['Make or Buy_ebom'] == mob]

            if ebom_filtered.empty:
                continue  # Skip if there are no EBOM parts with this Make/Buy

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