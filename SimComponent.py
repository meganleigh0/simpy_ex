# ------------------------------------------------------------
# PIE CHARTS: XM30 vs SEP – PARTS BY ORG AND MAKE/BUY
# ------------------------------------------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ---- 1. Source dataframes ----
df_xm30 = xm30_mbom_merged_2.copy()
df_sep  = v3_scr_lim_ang_makes.copy()   # change if your SEP df has a different name

# Detect Make/Buy column in XM30 (and assume same name in SEP)
mb_candidates = [c for c in df_xm30.columns if 'Make/Buy' in c]
if not mb_candidates:
    raise KeyError("No column containing 'Make/Buy' found in XM30 dataset.")
mb_col = mb_candidates[0]

# ---- 2. Aggregate unique part counts ----

# XM30 by Org
xm30_org = (
    df_xm30.groupby('Org', as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'Num_Parts'})
)
xm30_org = xm30_org[(xm30_org['Org'].notna()) & (xm30_org['Num_Parts'] > 0)]

# XM30 by Make/Buy
xm30_mb = (
    df_xm30.groupby(mb_col, as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'Num_Parts'})
)
xm30_mb = xm30_mb[(xm30_mb[mb_col].notna()) & (xm30_mb['Num_Parts'] > 0)]

# SEP by Org
sep_org = (
    df_sep.groupby('Org', as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'Num_Parts'})
)
sep_org = sep_org[(sep_org['Org'].notna()) & (sep_org['Num_Parts'] > 0)]

# SEP by Make/Buy
sep_mb = (
    df_sep.groupby(mb_col, as_index=False)['Part_Number']
    .nunique()
    .rename(columns={'Part_Number': 'Num_Parts'})
)
sep_mb = sep_mb[(sep_mb[mb_col].notna()) & (sep_mb['Num_Parts'] > 0)]

# ---- 3. Build 2x2 grid of pies ----
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "domain"}, {"type": "domain"}],
           [{"type": "domain"}, {"type": "domain"}]],
    subplot_titles=(
        "XM30 – Parts by Org",
        "XM30 – Parts by Make/Buy",
        "SEP – Parts by Org",
        "SEP – Parts by Make/Buy"
    )
)

# Row 1, Col 1: XM30 by Org
if not xm30_org.empty:
    fig.add_trace(
        go.Pie(
            labels=xm30_org['Org'],
            values=xm30_org['Num_Parts'],
            name="XM30 Parts by Org",
            hole=0.3
        ),
        row=1, col=1
    )

# Row 1, Col 2: XM30 by Make/Buy
if not xm30_mb.empty:
    fig.add_trace(
        go.Pie(
            labels=xm30_mb[mb_col],
            values=xm30_mb['Num_Parts'],
            name="XM30 Parts by Make/Buy",
            hole=0.3
        ),
        row=1, col=2
    )

# Row 2, Col 1: SEP by Org
if not sep_org.empty:
    fig.add_trace(
        go.Pie(
            labels=sep_org['Org'],
            values=sep_org['Num_Parts'],
            name="SEP Parts by Org",
            hole=0.3
        ),
        row=2, col=1
    )

# Row 2, Col 2: SEP by Make/Buy
if not sep_mb.empty:
    fig.add_trace(
        go.Pie(
            labels=sep_mb[mb_col],
            values=sep_mb['Num_Parts'],
            name="SEP Parts by Make/Buy",
            hole=0.3
        ),
        row=2, col=2
    )

fig.update_traces(textinfo='percent+label')

fig.update_layout(
    title_text="XM30 vs SEP – Distribution of Parts by Org and Make/Buy",
    height=800,
    width=1000,
    legend_title="Category"
)

fig.show()