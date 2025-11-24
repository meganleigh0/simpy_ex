# ------------------------------------------------------------
# ORG-LEVEL PLOTLY VISUALS: PART COUNTS + SEP HOURS ONLY
# ------------------------------------------------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# 1. DATA SOURCES
df_xm30 = xm30_mbom_merged_2.copy()   # XM30 merged dataset
df_sep  = v3_scr_lim_ang_makes.copy() # SEP dataset (update if needed)

# detect Make/Buy column
mb_candidates = [c for c in df_xm30.columns if 'Make/Buy' in c]
if not mb_candidates:
    raise KeyError("No column containing 'Make/Buy' found in XM30 dataset.")
mb_col = mb_candidates[0]

# 2. GROUPED METRICS PER ORG

# XM30 part counts by Make/Buy
xm30_counts = (
    df_xm30.groupby(['Org', mb_col], as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'XM30_Parts'})
)

# SEP part counts by Make/Buy
sep_counts = (
    df_sep.groupby(['Org', mb_col], as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'SEP_Parts'})
)

# SEP actual hours
sep_hours = (
    df_sep.groupby('Org', as_index=False)['CWS']
    .sum()
    .rename(columns={'CWS': 'SEP_Hours'})
)

# Combine metrics
org_metrics = (
    xm30_counts
        .merge(sep_counts, on=['Org', mb_col], how='outer')
        .merge(sep_hours, on='Org', how='left')
)

# ensure numeric for filtering/plotting
for col in ['XM30_Parts', 'SEP_Parts', 'SEP_Hours']:
    if col in org_metrics.columns:
        org_metrics[col] = org_metrics[col].fillna(0)

# 3. CREATE VISUALS PER ORG (SKIP COMPLETELY EMPTY ORGS)

orgs = org_metrics['Org'].dropna().unique()

for org in orgs:
    df_o = org_metrics[org_metrics['Org'] == org].copy()

    # filter out Make/Buy rows where both XM30 and SEP parts are 0
    df_o_parts = df_o[
        (df_o['XM30_Parts'] > 0) | (df_o['SEP_Parts'] > 0)
    ]

    # SEP hours value (same for each row for this org)
    sep_hours_val = df_o['SEP_Hours'].iloc[0] if not df_o.empty else 0

    # if no parts and no hours, skip this org entirely
    if df_o_parts.empty and sep_hours_val == 0:
        continue

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{org} – Part Counts (Make/Buy)",
            f"{org} – SEP Labor Hours"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # LEFT: Part counts (XM30 vs SEP)
    fig.add_trace(
        go.Bar(
            x=df_o_parts[mb_col],
            y=df_o_parts['XM30_Parts'],
            name="XM30 Parts"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df_o_parts[mb_col],
            y=df_o_parts['SEP_Parts'],
            name="SEP Parts"
        ),
        row=1, col=1
    )

    # RIGHT: SEP Hours (only if > 0 so bar isn't flat)
    if sep_hours_val > 0:
        fig.add_trace(
            go.Bar(
                x=['SEP Hours'],
                y=[sep_hours_val],
                name='SEP Hours'
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=f"{org} – XM30 & SEP Summary",
        barmode='group',
        height=520,
        width=1100,
        showlegend=True
    )

    fig.update_xaxes(title_text="Make / Buy", row=1, col=1)
    fig.update_yaxes(title_text="Number of Parts", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text="Labor Hours (SEP)", row=1, col=2)

    fig.show()
