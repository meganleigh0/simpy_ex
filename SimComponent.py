import re
import pandas as pd
# -----------------------------------------------------------
# 1.  READ BOTH TABLES
# -----------------------------------------------------------
# “OTHER RATES” – already in memory from Data Wrangler
other_rates = RATE_SUM_DATA.parse("OTHER RATES", skiprows=5).dropna(how="all")

# “BURDEN RATE IMPORT” – the master file you need to update
BURDEN_RATE = pd.read_excel("data/BurdenRateImport.xlsx", sheet_name="<<TODO‑sheet‑name>>")

# -----------------------------------------------------------
# 2.  CLEAN COLUMN NAMES SO WE CAN WORK WITH THEM
# -----------------------------------------------------------
# Keep only the CY20xx columns and turn them into a pure four‑digit year label
other_rates.columns = [
    re.sub(r"#\s*CY", "", c).strip() if "CY" in c else c
    for c in other_rates.columns
]
year_cols = [c for c in other_rates.columns if c.isdigit()]          # ['2022','2023',…]

# -----------------------------------------------------------
# 3.  MAP DESCRIPTIONS → TARGET COLUMNS IN BURDEN_RATE
# -----------------------------------------------------------
# A manual dictionary is the most reliable (you only have ~8‑10 rows)
DESC_TO_COL = {
    # description fragment (ANY part of other_rates['Unnamed: 1']) : BURDEN_RATE column
    "CSSC G & A":            "G&A CSSC",
    "DIVISION GENERAL":      "G&A GDAMS",
    "GDLS PROCUREMENT":      "Proc O/H G",
    "FREIGHT":               "Proc O/H C",
    "MAJOR END‑ITEM":        "G&A MEI",
    "SUPPORT RATE":          "Spare Alloc",
    # add / tweak as needed …
}

def match_target(desc: str) -> str | None:
    """Return the BURDEN_RATE column that matches the description line."""
    for key, col in DESC_TO_COL.items():
        if key.lower() in str(desc).lower():
            return col
    return None   # anything unmatched will be ignored

other_rates["target_col"] = other_rates.iloc[:, 1].apply(match_target)   # assumes “Unnamed: 1” is the desc. column
other_rates = other_rates.dropna(subset=["target_col"])                  # keep only rows we can place

# -----------------------------------------------------------
# 4.  MELT TO LONG FORMAT AND MERGE INTO THE MASTER
# -----------------------------------------------------------
long_rates = other_rates.melt(
    id_vars=["target_col"],           # what stays wide
    value_vars=year_cols,             # what becomes long
    var_name="Date",                  # new year column (matches BURDEN_RATE["Date"])
    value_name="rate"
).astype({"Date": int})               # make sure year is an int

# iterate row‑by‑row to write numbers into BURDEN_RATE
for row in long_rates.itertuples(index=False):
    yr_mask = BURDEN_RATE["Date"] == row.Date
    BURDEN_RATE.loc[yr_mask, row.target_col] = row.rate

# -----------------------------------------------------------
# 5.  FILL FUTURE YEARS WITH THE LAST KNOWN VALUE
# -----------------------------------------------------------
for col in long_rates["target_col"].unique():
    BURDEN_RATE[col] = pd.to_numeric(BURDEN_RATE[col], errors="coerce")   # ensure numeric
    BURDEN_RATE[col] = BURDEN_RATE[col].ffill()                           # forward‑fill down the sheet

# -----------------------------------------------------------
# 6.  SAVE
# -----------------------------------------------------------
BURDEN_RATE.to_excel("data/BurdenRateImport_updated.xlsx", index=False)