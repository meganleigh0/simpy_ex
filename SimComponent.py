# ----------------- ONE-CELL PIPELINE: update BurdenRateImport from OTHER RATES -----------------
import re
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# 1. Helper dictionaries – adjust if your column names ever change
# ────────────────────────────────────────────────────────────────────────────────
POOL_KEYWORDS = {             # text to look for  ➜  Burden Pool in BURDEN_RATE
    "CSSC": "CSSC",
    "GDLS": "GDLS",
    "DIVISION": "GDLS"        # “Division General & Adm” rows belong to GDLS pool
}

COST_COL_KEYWORDS = {         # text to look for  ➜  cost-element column to update
    "G & A":            "GRA G & A",
    "REORDER POINT":    "ROP",
    "OVERHEAD RATE":    "PROC O/H",
    "PROCUREMENT RATE": "PCURE GDLS",
    "MAJOR END-ITEM":   "MAJOR END ITEM",
    "SUPPORT RATE":     "SUPPORT",
    "CONTL TEST":       "CONTL TEST RATE"
}

# if you only want to write into columns that are called out in your
# cost_element_burdening_matrix, keep the intersection:
valid_cost_columns = set(cost_element_df.columns) & set(BURDEN_RATE.columns)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Tidy OTHER RATES so we have a clean index + CY20XX numeric columns
# ────────────────────────────────────────────────────────────────────────────────
other_rates = (
    other_rates
      .dropna(how="all", axis=0)                # remove empty rows
      .dropna(how="all", axis=1)                # remove empty cols
)
# drop the leading “# ” that Jupyter shows and make sure we keep CY columns only
other_rates.columns = [
    re.sub(r"^#\s*", "", c).strip()             # “# CY2022” ➜ “CY2022”
    for c in other_rates.columns
]
year_cols = [c for c in other_rates.columns if re.fullmatch(r"CY\d{4}", c)]
other_rates = other_rates.set_index(other_rates.columns[0])   # first col is the label

# ────────────────────────────────────────────────────────────────────────────────
# 3. Main loop – for every line in OTHER RATES push values to BURDEN_RATE
# ────────────────────────────────────────────────────────────────────────────────
for label, row in other_rates[year_cols].iterrows():

    # which burden pool(s) does this line affect?
    pools = {
        POOL_KEYWORDS[k] for k in POOL_KEYWORDS if k in label.upper()
    } or {"GDLS"}   # default to GDLS if no keyword found

    # which cost-element column should be updated?
    dest_col = next((
        COST_COL_KEYWORDS[k] for k in COST_COL_KEYWORDS if k in label.upper()
    ), None)
    if dest_col is None or dest_col not in valid_cost_columns:
        # Nothing to update – skip quietly
        continue

    for pool in pools:
        for cy_col in year_cols:                       # e.g. “CY2022”
            yr = int(cy_col[2:])                       # 2022 -> integer
            mask = (
                (BURDEN_RATE["Burden Pool"].str.upper() == pool)
                & (BURDEN_RATE["Date"] == yr)
            )
            BURDEN_RATE.loc[mask, dest_col] = row[cy_col]

# ────────────────────────────────────────────────────────────────────────────────
# 4. Optional rounding (uncomment / tweak to taste)
# ────────────────────────────────────────────────────────────────────────────────
for col in valid_cost_columns:
    if col == "COM":                                       # six decimal places
        BURDEN_RATE[col] = BURDEN_RATE[col].round(6)
    else:                                                  # five elsewhere
        BURDEN_RATE[col] = BURDEN_RATE[col].round(5)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Save a clean copy so the original file stays untouched
# ────────────────────────────────────────────────────────────────────────────────
out_path = Path("data") / "BurdenRateImport_updated.xlsx"
BURDEN_RATE.to_excel(out_path, index=False)

print(f"✅ Burden rates updated and written to {out_path}")
# --------------------------------------------------------------------------------