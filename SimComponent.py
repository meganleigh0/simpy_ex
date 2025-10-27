# === ONE CELL: Estimate XM30/M10 hours from SEP hours-per-part, by (Usr Org, Make/Buy) ===
# Requires: pandas, numpy, matplotlib

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 1) Load exactly-as-is (Excel or CSV) ----
FILE_PATH = "your_file.xlsx"   # <-- change this
def _read_any(path):
    p = path.lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    elif p.endswith(".csv"):
        return pd.read_csv(path)
    # try excel then csv
    try:
        return pd.read_excel(path)
    except:
        return pd.read_csv(path)

df = _read_any(FILE_PATH).copy()

# ---- 2) Standardize the expected columns (use your exact names) ----
expected = ["Usr Org","Make/Buy","CWS","Part-Number_sep","Part-Number_xm30","Part-Number_m10"]
# If someone uppercases/lowercases headers in Excel, normalize lightly:
norm = {c: re.sub(r"\s+", " ", str(c)).strip() for c in df.columns}
df.columns = [norm[c] if c in norm else c for c in df.columns]

# Keep only the columns we need, in order
df = df[expected].copy()

# ---- 3) Types & cleaning ----
num_cols = ["CWS","Part-Number_sep","Part-Number_xm30","Part-Number_m10"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["Make/Buy"] = df["Make/Buy"].astype(str).str.upper().str.strip()

# ---- 4) Core idea: use SEP hours-per-part as the rate for each group ----
# Base hours per part (SEP). If SEP part count is 0 or NaN, set NaN and backfill from group medians.
df["hpp_sep_raw"] = np.where(df["Part-Number_sep"] > 0, df["CWS"] / df["Part-Number_sep"], np.nan)

# Group medians to backfill missing/zero cases:
g1 = df.groupby(["Usr Org","Make/Buy"])["hpp_sep_raw"].median()
g2 = df.groupby(["Make/Buy"])["hpp_sep_raw"].median()
global_med = df["hpp_sep_raw"].median()

def fill_hpp(row):
    if not np.isnan(row["hpp_sep_raw"]):
        return row["hpp_sep_raw"]
    key1 = (row["Usr Org"], row["Make/Buy"])
    if key1 in g1 and not np.isnan(g1.loc[key1]):
        return g1.loc[key1]
    if row["Make/Buy"] in g2 and not np.isnan(g2.loc[row["Make/Buy"]]):
        return g2.loc[row["Make/Buy"]]
    return global_med

df["hpp_sep"] = df.apply(fill_hpp, axis=1)

# ---- 5) Estimate hours for XM30 and M10 using each row's hpp_sep ----
df["Est_Hours_XM30"] = (df["hpp_sep"] * df["Part-Number_xm30"]).fillna(0)
df["Est_Hours_M10"]  = (df["hpp_sep"] * df["Part-Number_m10"]).fillna(0)

# Guard against tiny negatives from numeric noise
df["Est_Hours_XM30"] = df["Est_Hours_XM30"].clip(lower=0)
df["Est_Hours_M10"]  = df["Est_Hours_M10"].clip(lower=0)

# ---- 6) Nice per-row view ----
view_cols = [
    "Usr Org","Make/Buy",
    "CWS","Part-Number_sep","hpp_sep",
    "Part-Number_xm30","Est_Hours_XM30",
    "Part-Number_m10","Est_Hours_M10"
]
print("\n=== Per-row SEP rate and estimated hours (rounded) ===")
print(df[view_cols].round(2).to_string(index=False))

# ---- 7) Rollups by (Usr Org, Make/Buy) and by Usr Org ----
grp_cols = ["Usr Org","Make/Buy"]
rollup = (df
          .groupby(grp_cols, dropna=False)[["CWS","Est_Hours_XM30","Est_Hours_M10"]]
          .sum()
          .reset_index())
print("\n=== Rollup by (Usr Org, Make/Buy) ===")
print(rollup.round(2).to_string(index=False))

rollup_org = (df
              .groupby("Usr Org", dropna=False)[["CWS","Est_Hours_XM30","Est_Hours_M10"]]
              .sum()
              .reset_index())
print("\n=== Rollup by Usr Org ===")
print(rollup_org.round(2).to_string(index=False))

# ---- 8) Quick visuals: Estimated XM30/M10 hours by Usr Org ----
plt.figure(figsize=(8,4))
plt.title("Estimated Hours by Usr Org — XM30")
plt.bar(rollup_org["Usr Org"].astype(str), rollup_org["Est_Hours_XM30"])
plt.ylabel("Estimated Hours (XM30)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.title("Estimated Hours by Usr Org — M10")
plt.bar(rollup_org["Usr Org"].astype(str), rollup_org["Est_Hours_M10"])
plt.ylabel("Estimated Hours (M10)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# (Optional) Save to file:
# df.to_excel("hours_estimates_output.xlsx", index=False)