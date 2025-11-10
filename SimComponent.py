# === Cost Performance (CPI) by CHG#: CTD and YTD ==============================
import pandas as pd
import numpy as np
from datetime import datetime

# Reuse existing variables if present; set safe defaults
if 'df' not in globals():
    raise NameError("Expected in-memory `df` with DATE, COST-SET, HOURS, CHG#.")

GROUP_COL = globals().get('GROUP_COL', 'CHG#')
ANCHOR    = globals().get('ANCHOR', datetime.now())

# Ensure DATE is datetime and trim any future-dated rows
df_use = df.copy()
df_use['DATE'] = pd.to_datetime(df_use['DATE'], errors='coerce')
df_use = df_use[df_use['DATE'] <= ANCHOR]

# Windows
ytd_start = datetime(ANCHOR.year, 1, 1)

def rollup_acwp_bcwp(dframe):
    """Return ACWP & BCWP sums by GROUP_COL; ensure missing cost-sets are 0."""
    g = (dframe
         .groupby([GROUP_COL, 'COST-SET'], dropna=False)['HOURS']
         .sum()
         .unstack(fill_value=0.0))
    for k in ['ACWP', 'BCWP']:
        if k not in g.columns:
            g[k] = 0.0
    return g[['ACWP','BCWP']].astype(float)

# CTD and YTD totals
ctd = rollup_acwp_bcwp(df_use)
ytd = rollup_acwp_bcwp(df_use[(df_use['DATE'] >= ytd_start)])

def safe_ratio(num, den):
    return (num / den.replace({0: np.nan}))

# CPI = BCWP / ACWP
cpi_ctd = safe_ratio(ctd['BCWP'], ctd['ACWP']).round(2)
cpi_ytd = safe_ratio(ytd['BCWP'], ytd['ACWP']).round(2)

cost_performance_tbl = pd.DataFrame({
    'CTD': cpi_ctd,
    'YTD': cpi_ytd
})
cost_performance_tbl.index.name = None

# TOTAL row (ratio of sums, not average of ratios)
tot_ctd_acwp = ctd['ACWP'].sum()
tot_ctd_bcwp = ctd['BCWP'].sum()
tot_ytd_acwp = ytd['ACWP'].sum()
tot_ytd_bcwp = ytd['BCWP'].sum()

tot_row = pd.Series({
    'CTD': np.nan if np.isclose(tot_ctd_acwp, 0) else round(tot_ctd_bcwp / tot_ctd_acwp, 2),
    'YTD': np.nan if np.isclose(tot_ytd_acwp, 0) else round(tot_ytd_bcwp / tot_ytd_acwp, 2)
}, name='TOTAL')

cost_performance_tbl = pd.concat([cost_performance_tbl, tot_row.to_frame().T])

print(f"Anchor: {ANCHOR:%Y-%m-%d} | YTD start: {ytd_start:%Y-%m-%d}")
print("\n=== Cost Performance (CPI: CTD & YTD) by CHG# ===")
display(cost_performance_tbl)
# ============================================================================== 