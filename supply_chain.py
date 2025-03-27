import pandas as pd
import plotly.graph_objects as go

def calculate_and_plot_bom_completion_detailed(combined_df, snapshot_date, variant_id="Variant_1"):
    """
    Detailed completion calculation based on the logic in Data Analysis #1 and #2:
    - Match: Same part number and quantity
    - Quantity Mismatch: Same part number, different quantity
    - Missing: In EBOM but not in MBOM
    """
    records = []
    bars = []

    for source, qty_col, match_flag in [('TeamCenter', 'Quantity_mbom_tc', 'Match_EBOM_MBOM_TC'),
                                        ('Oracle', 'Quantity', 'Match_EBOM_MBOM_Oracle')]:
        for mob in ['Make', 'Buy']:
            # Filter EBOM parts for current Make/Buy category
            ebom_parts = combined_df[combined_df['Make or Buy_ebom'] == mob]

            total_parts = ebom_parts['PART_NUMBER'].nunique()

            # Match: Same part number and quantity
            matched_parts = ebom_parts[ebom_parts[match_flag] == True]['PART_NUMBER'].nunique()

            # Quantity mismatch: Part is found but quantity is different
            qty_mismatch_parts = ebom_parts[
                (ebom_parts[qty_col].notnull()) & (ebom_parts[match_flag] == False)
            ]['PART_NUMBER'].nunique()

            # Missing in MBOM: No entry in MBOM at all
            missing_parts = ebom_parts[ebom_parts[qty_col].isnull()]['PART_NUMBER'].nunique()

            # Percent completion = exact matches / total EBOM parts
            percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

            label = f"{source} - {mob}"
            bars.append({'label': label, 'value': percent_matched})
            records.append({
                'snapshot_date': snapshot_date,
                'variant_id': variant_id,
                'source': source,
                'make_or_buy': mob,
                'total_parts': total_parts,
                'matched_parts': matched_parts,
                'quantity_mismatches': qty_mismatch_parts,
                'missing_parts': missing_parts,
                'percent_matched': percent_matched
            })

    # Plot snapshot bar chart
    fig = go.Figure(data=[
        go.Bar(x=[bar['label']], y=[bar['value']], name=bar['label']) for bar in bars
    ])
    fig.update_layout(
        title=f'EBOM to MBOM Completion (Exact Match Only) - {snapshot_date}',
        yaxis_title='% Matched (Exact Quantity)',
        xaxis_title='Source - Make or Buy',
        yaxis=dict(range=[0, 100]),
        barmode='group'
    )
    fig.show()

    return pd.DataFrame(records)