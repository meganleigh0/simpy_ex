# ------------------------------------------------------------
# Pies & Bars for Make/Buy Part Distributions and SEP Hours
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- 1. Source dataframes ----
df_xm30 = xm30_mbom_merged_2.copy()       # XM30 merged dataset
df_sep  = v3_scr_lim_ang_makes.copy()     # SEP dataset (change if needed)

# Detect Make/Buy column name
mb_candidates = [c for c in df_xm30.columns if 'Make/Buy' in c]
if not mb_candidates:
    raise KeyError("No column containing 'Make/Buy' found in XM30 dataset.")
mb_col = mb_candidates[0]

# Standardize Make/Buy labels into 'Make' / 'Buy'
def clean_mb(s):
    s = s.astype(str).str.strip().str.upper()
    out = np.where(s == 'MAKE', 'Make',
          np.where(s == 'BUY',  'Buy',  'Other'))
    return pd.Series(out, index=s.index)

df_xm30['MB_group'] = clean_mb(df_xm30[mb_col])
df_sep['MB_group']  = clean_mb(df_sep[mb_col])

# Keep only Make/Buy (drop "Other", if any)
df_xm30 = df_xm30[df_xm30['MB_group'].isin(['Make', 'Buy'])]
df_sep  = df_sep[df_sep['MB_group'].isin(['Make', 'Buy'])]

# ------------------------------------------------------------
# 2. Aggregations for pies
# ------------------------------------------------------------

def agg_parts_by_org(df, mb_value):
    """Unique part counts by Org for a specific Make/Buy value."""
    sub = df[df['MB_group'] == mb_value]
    if sub.empty:
        return pd.DataFrame(columns=['Org', 'Num_Parts'])
    out = (sub
           .groupby('Org', as_index=False)['Part_Number']
           .nunique()
           .rename(columns={'Part_Number': 'Num_Parts'}))
    out = out[(out['Org'].notna()) & (out['Num_Parts'] > 0)]
    return out

# XM30 parts
xm30_make_org = agg_parts_by_org(df_xm30, 'Make')
xm30_buy_org  = agg_parts_by_org(df_xm30, 'Buy')

# SEP parts
sep_make_org = agg_parts_by_org(df_sep, 'Make')
sep_buy_org  = agg_parts_by_org(df_sep, 'Buy')

# SEP hours by Org (sum of CWS)
sep_hours_org = (
    df_sep.groupby('Org', as_index=False)['CWS']
    .sum()
    .rename(columns={'CWS': 'SEP_Hours'})
)
sep_hours_org = sep_hours_org[(sep_hours_org['Org'].notna()) &
                              (sep_hours_org['SEP_Hours'] > 0)]

# ------------------------------------------------------------
# 3. PIE CHART GRID
# ------------------------------------------------------------
fig_pies = make_subplots(
    rows=3, cols=2,
    specs=[
        [{"type": "domain"}, {"type": "domain"}],
        [{"type": "domain"}, {"type": "domain"}],
        [{"type": "domain"}, {"type": None}]
    ],
    subplot_titles=(
        "XM30 – MAKES: Parts by Org",
        "XM30 – BUYS: Parts by Org",
        "SEP – MAKES: Parts by Org",
        "SEP – BUYS: Parts by Org",
        "SEP – Hours by Org",
        ""
    )
)

# XM30 Make
if not xm30_make_org.empty:
    fig_pies.add_trace(
        go.Pie(
            labels=xm30_make_org['Org'],
            values=xm30_make_org['Num_Parts'],
            hole=0.4,
            name="XM30 Make Parts"
        ),
        row=1, col=1
    )

# XM30 Buy
if not xm30_buy_org.empty:
    fig_pies.add_trace(
        go.Pie(
            labels=xm30_buy_org['Org'],
            values=xm30_buy_org['Num_Parts'],
            hole=0.4,
            name="XM30 Buy Parts"
        ),
        row=1, col=2
    )

# SEP Make
if not sep_make_org.empty:
    fig_pies.add_trace(
        go.Pie(
            labels=sep_make_org['Org'],
            values=sep_make_org['Num_Parts'],
            hole=0.4,
            name="SEP Make Parts"
        ),
        row=2, col=1
    )

# SEP Buy
if not sep_buy_org.empty:
    fig_pies.add_trace(
        go.Pie(
            labels=sep_buy_org['Org'],
            values=sep_buy_org['Num_Parts'],
            hole=0.4,
            name="SEP Buy Parts"
        ),
        row=2, col=2
    )

# SEP Hours
if not sep_hours_org.empty:
    fig_pies.add_trace(
        go.Pie(
            labels=sep_hours_org['Org'],
            values=sep_hours_org['SEP_Hours'],
            hole=0.4,
            name="SEP Hours"
        ),
        row=3, col=1
    )

# Text = label + percent + raw value
fig_pies.update_traces(textinfo='label+percent+value')

fig_pies.update_layout(
    title_text="XM30 vs SEP – Make / Buy Part Distributions and SEP Hours by Org",
    height=1100,
    width=1100,
    legend_title="Org"
)

fig_pies.show()

# ------------------------------------------------------------
# 4. BAR CHARTS (my “must have” views)
# ------------------------------------------------------------

# XM30 parts by Org & Make/Buy
xm30_parts_org_mb = (
    df_xm30
    .groupby(['Org', 'MB_group'], as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'Num_Parts'})
)
xm30_parts_org_mb = xm30_parts_org_mb[xm30_parts_org_mb['Num_Parts'] > 0]

fig_bar_xm30 = px.bar(
    xm30_parts_org_mb,
    x='Org',
    y='Num_Parts',
    color='MB_group',
    barmode='group',
    text='Num_Parts',
    title='XM30 – Number of Parts by Org and Make/Buy'
)
fig_bar_xm30.update_traces(textposition='outside')
fig_bar_xm30.update_layout(yaxis_title='Number of Parts', xaxis_title='Org',
                           legend_title='Make/Buy')
fig_bar_xm30.show()

# SEP parts by Org & Make/Buy
sep_parts_org_mb = (
    df_sep
    .groupby(['Org', 'MB_group'], as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'Num_Parts'})
)
sep_parts_org_mb = sep_parts_org_mb[sep_parts_org_mb['Num_Parts'] > 0]

fig_bar_sep = px.bar(
    sep_parts_org_mb,
    x='Org',
    y='Num_Parts',
    color='MB_group',
    barmode='group',
    text='Num_Parts',
    title='SEP – Number of Parts by Org and Make/Buy'
)
fig_bar_sep.update_traces(textposition='outside')
fig_bar_sep.update_layout(yaxis_title='Number of Parts', xaxis_title='Org',
                          legend_title='Make/Buy')
fig_bar_sep.show()

# SEP Hours by Org
fig_bar_sep_hours = px.bar(
    sep_hours_org,
    x='Org',
    y='SEP_Hours',
    text='SEP_Hours',
    title='SEP – Total Labor Hours (CWS) by Org'
)
fig_bar_sep_hours.update_traces(texttemplate='%{text:.1f}', textposition='outside')
fig_bar_sep_hours.update_layout(yaxis_title='Labor Hours (SEP)', xaxis_title='Org')
fig_bar_sep_hours.show()