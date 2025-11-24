import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ------------------------------------------------------------
# 1. DEFINE DATA SOURCES
# ------------------------------------------------------------
df_xm30 = xm30_mbom_merged_2.copy()               # XM30 merged dataset
df_sep  = v3_scr_lim_ang_makes.copy()             # SEP dataset (update if needed)

# detect Make/Buy column
mb_col = [c for c in df_xm30.columns if 'Make/Buy' in c][0]

# ------------------------------------------------------------
# 2. CALCULATE 80% PREDICTED HOURS FOR XM30
# ------------------------------------------------------------
df_xm30['Pred_Hours'] = df_xm30['CWS'] * df_xm30['Children'] * 0.80

# ------------------------------------------------------------
# 3. GROUPED METRICS PER ORG
# ------------------------------------------------------------

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
    df_sep.groupby(['Org'], as_index=False)['CWS']
    .sum()
    .rename(columns={'CWS': 'SEP_Hours'})
)

# XM30 predicted hours (80%)
xm30_hours = (
    df_xm30.groupby(['Org'], as_index=False)['Pred_Hours']
    .sum()
    .rename(columns={'Pred_Hours': 'XM30_Pred_Hours'})
)

# Combine metrics
org_metrics = (
    xm30_counts
        .merge(sep_counts, on=['Org', mb_col], how='outer')
        .merge(sep_hours, on='Org', how='left')
        .merge(xm30_hours, on='Org', how='left')
)

# ------------------------------------------------------------
# 4. CREATE VISUALS PER ORG
# ------------------------------------------------------------
orgs = org_metrics['Org'].unique()

for org in orgs:
    df_o = org_metrics[org_metrics['Org'] == org]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"{org} – Part Counts (Make/Buy)",
            f"{org} – Hours (SEP vs XM30 Predicted)"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # --- LEFT: Part counts ---
    fig.add_trace(
        go.Bar(
            x=df_o[mb_col],
            y=df_o['XM30_Parts'],
            name="XM30 Parts",
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=df_o[mb_col],
            y=df_o['SEP_Parts'],
            name="SEP Parts",
            marker_color='#ff7f0e'
        ),
        row=1, col=1
    )

    # --- RIGHT: Hours ---
    fig.add_trace(
        go.Bar(
            x=['SEP Hours'],
            y=[df_o['SEP_Hours'].iloc[0]],
            name='SEP Hours',
            marker_color='#2ca02c'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=['XM30 Pred Hours'],
            y=[df_o['XM30_Pred_Hours'].iloc[0]],
            name='XM30 Pred. Hours (80%)',
            marker_color='#d62728'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title=f"{org} – XM30 & SEP Summary",
        barmode='group',
        height=500,
        width=1100,
        showlegend=True
    )

    fig.show()
