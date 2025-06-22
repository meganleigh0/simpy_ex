###############################################################################
#  UPDATE BurdenRateImport.xlsx WITH THE 9 LINES FROM OTHER RATES             #
###############################################################################
import re
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Assumes you already ran something like:                                  
#         other_rates   = RATE_SUM_DATA.parse("OTHER RATES", skiprows=5, ...)
#         BURDEN_RATE   = pd.read_excel("data/BurdenRateImport.xlsx")          
#     If not, load them first.                                                 
# ─────────────────────────────────────────────────────────────────────────────

# 1)  DECLUTTER the “OTHER RATES” sheet  ──────────────────────────────────────
label_col = other_rates.columns[0]                                # first col → index
other_rates.columns = [re.sub(r"^#\s*", "", c).strip()            # “# CY2022” → “CY2022”
                       for c in other_rates.columns]
other_rates = other_rates.set_index(label_col).dropna(how="all")  # tidy blanks

year_cols  = [c for c in other_rates.columns if re.fullmatch(r"CY\d{4}", c)]
other_rates = (other_rates[year_cols]           # keep just CY20XX numbers
                           .ffill(axis=1))      # forward-fill → last value “sticks”

# 2)  QUICK DICTIONARIES THAT TELL THE SCRIPT **WHAT** TO TOUCH  ─────────────
#
# Every row in OTHER RATES is matched to:
#   • one-or-more “Burden Pool” values in BurdenRateImport  (CSSC, GDLS,…)
#   • one-or-more cost-element **columns** to overwrite      (G&A CSSC, REORD …)
# If your naming conventions ever change, just tweak these two dicts.

POOL_RULES = {                 # any of these phrases  ➜  write into these pools
    "CSSC":          ["CSSC"],
    "GDLS":          ["GDLS"],
    "GENERAL DYN":   ["GDLS"],
    "DIVISION":      ["GDLS"],
    "GDLS & CSSC":   ["GDLS", "CSSC"],          # apply to both
}

COLUMN_RULES = {               # phrase that appears in label  ➜  target-column hints
    "G & A":            ["G&A", "GRA"],         # picks up “G&A CSSC”, “GRA G & A” …
    "REORDER POINT":    ["REORD"],
    "OVERHEAD":         ["OVERHEAD", "PROC O/H", "OH"],
    "PROCUREMENT":      ["PCURE", "PROCURE"],
    "MAJOR END-ITEM":   ["MAJOR END"],
    "SUPPORT":          ["SUPPORT"],
    "CONTL TEST":       ["CONTL", "TEST"],
}

# 3)  MAKE LIFE EASY: create helpers that turn a row-label into (pools, cols) ─
def pools_for(label: str) -> list[str]:
    pools = []
    for key, val in POOL_RULES.items():
        if key in label:
            pools.extend(val)
    return pools or ["CSSC", "GDLS"]            # default → hit both sets

def columns_for(label: str) -> list[str]:
    hits = []
    for key, hints in COLUMN_RULES.items():
        if key in label:
            for h in hints:
                hits.extend([c for c in BURDEN_RATE.columns
                             if h in c.upper()])
    return list(dict.fromkeys(hits))            # drop dupes, keep order

# 4)  MAIN LOOP – push each OTHER RATES line into every matching cell ─────────
for label, rates in other_rates.iterrows():
    label_u = label.upper()

    dest_pools   = pools_for(label_u)
    dest_columns = columns_for(label_u)
    if not dest_columns:
        # Nothing in this row maps to a column we know about – skip quietly
        continue

    # pre-build a dict {year:int → rate:float}, forward-filled
    cy_rate = {int(c[2:]): rates[c] for c in year_cols}
    last_year  = max(k for k, v in cy_rate.items() if pd.notna(v))
    last_value = cy_rate[last_year]
    for y in range(last_year+1, BURDEN_RATE["Date"].max()+1):
        cy_rate[y] = last_value                 # carry last value forward

    # now slam the numbers in
    for pool in dest_pools:
        mask_pool = BURDEN_RATE["Burden Pool"].str.upper() == pool
        for col in dest_columns:
            # update *all* years present in BURDEN_RATE for this pool
            for idx, row in BURDEN_RATE[mask_pool].iterrows():
                yr = int(row["Date"])
                if yr in cy_rate:
                    BURDEN_RATE.at[idx, col] = cy_rate[yr]

# 5)  HOUSE-KEEPING – round the numbers exactly as you asked last time ───────
for c in BURDEN_RATE.select_dtypes(include="number").columns:
    BURDEN_RATE[c] = (BURDEN_RATE[c].round(6) if c.startswith("COM")
                      else BURDEN_RATE[c].round(5))

# 6)  SAVE A CLEAN COPY  ──────────────────────────────────────────────────────
out_path = Path("data") / "BurdenRateImport_updated.xlsx"
BURDEN_RATE.to_excel(out_path, index=False)
print(f"✅  Burden rates written to {out_path}")