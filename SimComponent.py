import plotly.graph_objects as go

def plot_bom_comparison_snapshot_by_make_buy(combined_df, snapshot_date):
    # Filter to only EBOM part numbers
    total_parts_ebom = combined_df[['PART_NUMBER', 'Make or Buy_ebom']].drop_duplicates()
    
    # Count total EBOM parts by Make/Buy
    ebom_counts = total_parts_ebom.groupby('Make or Buy_ebom').agg(total_ebom_parts=('PART_NUMBER', 'nunique')).reset_index()

    # Count matched MBOM TeamCenter by Make/Buy
    matched_tc = combined_df[combined_df['Match_EBOM_MBOM_TC']]
    matched_tc_counts = matched_tc.groupby('Make or Buy_ebom').agg(matched_mbom_tc=('PART_NUMBER', 'nunique')).reset_index()

    # Count matched MBOM Oracle by Make/Buy
    matched_oracle = combined_df[combined_df['Match_EBOM_MBOM_Oracle']]
    matched_oracle_counts = matched_oracle.groupby('Make or Buy_ebom').agg(matched_mbom_oracle=('PART_NUMBER', 'nunique')).reset_index()

    # Merge and calculate percentages
    summary = ebom_counts.merge(matched_tc_counts, on='Make or Buy_ebom', how='left') \
                         .merge(matched_oracle_counts, on='Make or Buy_ebom', how='left')
    summary.fillna(0, inplace=True)
    summary['percent_mbom_tc'] = (summary['matched_mbom_tc'] / summary['total_ebom_parts']) * 100
    summary['percent_mbom_oracle'] = (summary['matched_mbom_oracle'] / summary['total_ebom_parts']) * 100

    # Plot
    fig = go.Figure(data=[
        go.Bar(name='MBOM TeamCenter', x=summary['Make or Buy_ebom'], y=summary['percent_mbom_tc']),
        go.Bar(name='MBOM Oracle', x=summary['Make or Buy_ebom'], y=summary['percent_mbom_oracle'])
    ])
    fig.update_layout(
        title=f'MBOM Coverage vs EBOM by Make/Buy - Snapshot: {snapshot_date}',
        yaxis_title='% of EBOM Parts Matched',
        xaxis_title='Make or Buy',
        barmode='group',
        yaxis_range=[0, 100]
    )
    fig.show()