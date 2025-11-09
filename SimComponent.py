import pandas as pd

df = xm30_cobra_export_weekly_extract  # your DataFrame from the notebook

# Make the pivot: rows=CHG#, columns=COST-SET, values=sum(HOURS)
pivot = (
    df.pivot_table(
        index='CHG#',
        columns='COST-SET',
        values='HOURS',
        aggfunc='sum',
        fill_value=0
    )
    # optional: order the COST-SET columns to match Excel
    .reindex(columns=['ACWP','BCWP','BCWS','ETC'], fill_value=0)
    .sort_index()
)

# optional grand total row
pivot.loc['TOTAL'] = pivot.sum(numeric_only=True)

pivot

import pandas as pd

df = xm30_cobra_export_weekly_extract.copy()

# Ensure DATE is datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Sort then build cumulative HOURS per CHG# & COST-SET
df = df.sort_values(['CHG#','COST-SET','DATE'])
df['HOURS_CUM'] = df.groupby(['CHG#','COST-SET'])['HOURS'].cumsum()

# ---- pick your cutoff, or omit to use all available dates ----
# cutoff = pd.Timestamp('2025-01-31')
# df = df[df['DATE'] <= cutoff]
# --------------------------------------------------------------

cum_pivot = (
    df.pivot_table(
        index='CHG#',
        columns='COST-SET',
        values='HOURS_CUM',
        aggfunc='last',      # last cumulative value per group = total to date
        fill_value=0
    )
    .reindex(columns=['ACWP','BCWP','BCWS','ETC'], fill_value=0)
    .sort_index()
)

cum_pivot.loc['TOTAL'] = cum_pivot.sum(numeric_only=True)

cum_pivot