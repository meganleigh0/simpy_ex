import plotly.graph_objects as go

def plot_bom_completion_by_make_buy(combined_df, snapshot_date):
    bars = []

    for source, match_col in [('TeamCenter', 'Match_EBOM_MBOM_TC'), ('Oracle', 'Match_EBOM_MBOM_Oracle')]:
        for mob in ['Make', 'Buy']:
            ebom_filtered = combined_df[combined_df['Make or Buy_ebom'] == mob]
            total_parts = ebom_filtered['PART_NUMBER'].nunique()

            matched_parts = ebom_filtered[ebom_filtered[match_col]]['PART_NUMBER'].nunique()
            percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

            bars.append({
                'label': f'{source} - {mob}',
                'value': percent_matched
            })

    # Plot using Plotly
    fig = go.Figure(data=[
        go.Bar(name=bar['label'], x=[bar['label']], y=[bar['value']]) for bar in bars
    ])

    fig.update_layout(
        title=f'MBOM Completion by Make/Buy vs EBOM - {snapshot_date}',
        yaxis_title='% of EBOM Parts Matched',
        xaxis_title='Source - Make or Buy',
        yaxis_range=[0, 100],
        barmode='group'
    )
    fig.show()