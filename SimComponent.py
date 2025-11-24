# Plotly visuals for XM30 – Sep hours by Org and Make/Buy part counts
import plotly.express as px
import pandas as pd

# -------------------------------------------------------------------
# 1) Sep labor hours by Org (using CWS labor hours on xm30_mbom_merged_2)
# -------------------------------------------------------------------

# If your merged XM30 dataframe has a different name, update here:
df_xm30 = xm30_mbom_merged_2.copy()

# If there is a specific September-hours column, put that name here.
# Otherwise we’ll treat CWS as the September hours baseline.
hours_col = 'CWS'   # change to e.g. 'Sep_hours' if you have that column

sep_hours_by_org = (
    df_xm30
        .groupby('Org', as_index=False)[hours_col]
        .sum()
        .rename(columns={hours_col: 'Sep_hours'})
        .sort_values('Sep_hours', ascending=False)
)

# Optional: enforce a logical Org order if you mainly use these sites
org_order = ['TLH', 'SCR', 'LIM', 'GRW', 'COP']
sep_hours_by_org['Org'] = pd.Categorical(
    sep_hours_by_org['Org'],
    categories=org_order + [o for o in sep_hours_by_org['Org'].unique() if o not in org_order],
    ordered=True
)
sep_hours_by_org = sep_hours_by_org.sort_values('Org')

fig_hours = px.bar(
    sep_hours_by_org,
    x='Org',
    y='Sep_hours',
    text='Sep_hours',
    title='XM30 Sep Labor Hours by Org (CWS-based)'
)
fig_hours.update_traces(texttemplate='%{text:.0f}', textposition='outside')
fig_hours.update_layout(
    xaxis_title='Org',
    yaxis_title='Labor Hours (Sep)',
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

fig_hours.show()

# -------------------------------------------------------------------
# 2) Part counts by Make / Buy by Org for XM30
# -------------------------------------------------------------------

# Figure out which Make/Buy column exists in your merged XM30 df
if 'Make/Buy_Oracle' in df_xm30.columns:
    mb_col = 'Make/Buy_Oracle'
elif 'Make/Buy' in df_xm30.columns:
    mb_col = 'Make/Buy'
else:
    raise KeyError("No Make/Buy column found – expected 'Make/Buy_Oracle' or 'Make/Buy'.")

parts_mb_org = (
    df_xm30
        .groupby(['Org', mb_col], as_index=False)['Part_Number']
        .nunique()
        .rename(columns={'Part_Number': 'Num_parts'})
)

# keep Org ordering consistent with the first chart
parts_mb_org['Org'] = pd.Categorical(
    parts_mb_org['Org'],
    categories=sep_hours_by_org['Org'].cat.categories,
    ordered=True
)
parts_mb_org = parts_mb_org.sort_values(['Org', mb_col])

fig_parts = px.bar(
    parts_mb_org,
    x='Org',
    y='Num_parts',
    color=mb_col,
    barmode='stack',      # change to 'group' if you prefer side-by-side bars
    text='Num_parts',
    title='XM30 Parts by Make/Buy and Org'
)
fig_parts.update_traces(texttemplate='%{text}', textposition='inside')
fig_parts.update_layout(
    xaxis_title='Org',
    yaxis_title='Number of Unique Parts',
    legend_title='Make / Buy'
)

fig_parts.show()