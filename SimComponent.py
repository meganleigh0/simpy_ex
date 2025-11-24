import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Helper functions to find dataframes and clean Make/Buy
# ------------------------------------------------------------
def first_existing_df(names):
    """Return the first dataframe from globals() that exists and is not None."""
    g = globals()
    for n in names:
        if n in g and isinstance(g[n], pd.DataFrame):
            return g[n].copy()
    return None

def detect_make_buy_col(df):
    candidates = [c for c in df.columns if 'Make/Buy' in c]
    if not candidates:
        raise KeyError("No column containing 'Make/Buy' found in dataframe.")
    return candidates[0]

def clean_mb_series(s):
    s = s.astype(str).str.strip().str.upper()
    out = np.where(s == 'MAKE', 'Make',
          np.where(s == 'BUY',  'Buy',  'Other'))
    return pd.Series(out, index=s.index)

def summarize_bom(df, program, source):
    """Return rows: Program, Source, MB_group, Num_Parts."""
    if df is None:
        return pd.DataFrame(columns=['Program','Source','MB_group','Num_Parts'])
    mb_col = detect_make_buy_col(df)
    tmp = df.copy()
    tmp['MB_group'] = clean_mb_series(tmp[mb_col])
    # group by Make/Buy; use unique Part_Number counts
    grp = (
        tmp.groupby('MB_group', as_index=False)['Part_Number']
           .nunique()
           .rename(columns={'Part_Number': 'Num_Parts'})
    )
    grp['Program'] = program
    grp['Source'] = source
    # reorder columns
    return grp[['Program','Source','MB_group','Num_Parts']]

# ------------------------------------------------------------
# 1. Locate the BOM dataframes in your notebook
# ------------------------------------------------------------
xm30_oracle_df = first_existing_df(['xm30_oracle_mbom', 'xm30_oracle_mbom_merged'])
xm30_tc_df     = first_existing_df(['xm30_tc_mbom', 'xm30_tc_mbom_merged'])
sep_oracle_df  = first_existing_df(['sep_oracle_mbom', 'v3_oracle_mbom', 'sep_oracle_mbom_merged'])
sep_tc_df      = first_existing_df(['sep_tc_mbom', 'v3_tc_mbom', 'sep_tc_mbom_merged'])

# ------------------------------------------------------------
# 2. Build the combined summary table
# ------------------------------------------------------------
summary_list = []
summary_list.append(summarize_bom(xm30_oracle_df, 'XM30', 'Oracle'))
summary_list.append(summarize_bom(xm30_tc_df,     'XM30', 'Teamcenter'))
summary_list.append(summarize_bom(sep_oracle_df,  'SEP',  'Oracle'))
summary_list.append(summarize_bom(sep_tc_df,      'SEP',  'Teamcenter'))

summary = pd.concat(summary_list, ignore_index=True)
summary = summary[summary['Num_Parts'] > 0]  # remove empties

display(summary)  # optional: see the numeric table defining your universe

# ------------------------------------------------------------
# 3. PIE CHARTS – Make vs Buy for each Program & Source
# ------------------------------------------------------------
fig_pies = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "domain"}, {"type": "domain"}],
           [{"type": "domain"}, {"type": "domain"}]],
    subplot_titles=(
        "XM30 – Oracle (Make vs Buy)",
        "XM30 – Teamcenter (Make vs Buy)",
        "SEP – Oracle (Make vs Buy)",
        "SEP – Teamcenter (Make vs Buy)"
    )
)

def add_pie(program, source, row, col):
    sub = summary[(summary['Program'] == program) &
                  (summary['Source'] == source)]
    if sub.empty:
        return
    fig_pies.add_trace(
        go.Pie(
            labels=sub['MB_group'],
            values=sub['Num_Parts'],
            name=f"{program} {source}",
            hole=0.4
        ),
        row=row, col=col
    )

add_pie('XM30', 'Oracle',      1, 1)
add_pie('XM30', 'Teamcenter',  1, 2)
add_pie('SEP',  'Oracle',      2, 1)
add_pie('SEP',  'Teamcenter',  2, 2)

# show label + percent + raw value
fig_pies.update_traces(textinfo='label+percent+value')

fig_pies.update_layout(
    title_text="XM30 & SEP – Make vs Buy Distribution by Source (Oracle vs Teamcenter)",
    height=800,
    width=1000
)

fig_pies.show()

# ------------------------------------------------------------
# 4. BAR CHARTS – High-level “universe” views
# ------------------------------------------------------------

# 4a. Total parts by Program & Source (regardless of Make/Buy)
prog_source_totals = (
    summary.groupby(['Program','Source'], as_index=False)['Num_Parts']
           .sum()
)

fig_bar1 = px.bar(
    prog_source_totals,
    x='Program',
    y='Num_Parts',
    color='Source',
    barmode='group',
    text='Num_Parts',
    title='Total Unique Parts by Program and Source (Oracle vs Teamcenter)'
)
fig_bar1.update_traces(textposition='outside')
fig_bar1.update_layout(yaxis_title='Number of Unique Parts', xaxis_title='Program')
fig_bar1.show()

# 4b. Parts by Program, Source, and Make/Buy (stacked)
fig_bar2 = px.bar(
    summary,
    x='Program',
    y='Num_Parts',
    color='MB_group',
    pattern_shape='Source',
    barmode='stack',
    text='Num_Parts',
    title='Parts by Program, Source, and Make/Buy'
)
fig_bar2.update_traces(textposition='inside')
fig_bar2.update_layout(
    yaxis_title='Number of Unique Parts',
    xaxis_title='Program',
    legend_title='Make/Buy (pattern = Source)'
)
fig_bar2.show()
