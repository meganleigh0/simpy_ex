# ===== ONE-CELL ANALYSIS: Load -> Clean -> Estimate -> Evaluate -> Plot =====
# Requirements: pandas, numpy, scikit-learn, matplotlib
# If needed in Databricks: %pip install pandas numpy scikit-learn matplotlib

import os, re, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------
# 1) LOAD YOUR DATA "AS IS"
# -------------------------
# >>>> CHANGE THIS TO YOUR FILE <<<<
FILE_PATH = "your_file.xlsx"  # e.g., "hours_table.xlsx" or "hours_table.csv"

def _read_any(path):
    name = path.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(path)
    elif name.endswith(".csv"):
        return pd.read_csv(path)
    else:
        # Try excel first, then csv
        try:
            return pd.read_excel(path)
        except:
            return pd.read_csv(path)

df_raw = _read_any(FILE_PATH).copy()

# ------------------------------------------------
# 2) STANDARDIZE COLUMN NAMES & MAP EXPECTED NAMES
# ------------------------------------------------
def norm(s): 
    return re.sub(r"\s+", " ", str(s)).strip()

df = df_raw.copy()
df.columns = [norm(c) for c in df.columns]

# Expected names (case-insensitive); we’ll find best matches
expect_map = {
    "Usr Org": ["usr org","user org","usrorg","org"],
    "Make/Buy": ["make/buy","make buy","make_or_buy","makebuy"],
    "CWS": ["cws","hours","sep hours","sep_cws","c w s"],
    "Part-Number_sep": ["part-number_sep","part number sep","sep parts","sep count","sep"],
    "Part-Number_xm30": ["part-number_xm30","part number xm30","xm30 parts","xm30 count","xm30"],
    "Part-Number_m10": ["part-number_m10","part number m10","m10 parts","m10 count","m10"],
}

def find_col(possible_names, cols):
    cols_lower = {c.lower(): c for c in cols}
    for alias in possible_names:
        if alias in cols_lower: 
            return cols_lower[alias]
    # fuzzy: look for exact token match ignoring punctuation/space
    norm_cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for alias in possible_names:
        k = re.sub(r"[^a-z0-9]", "", alias.lower())
        if k in norm_cols:
            return norm_cols[k]
    # final: partial contains
    for alias in possible_names:
        for c in cols:
            if alias.replace(" ","") in c.lower().replace(" ",""):
                return c
    return None

colmap = {}
for canonical, aliases in expect_map.items():
    found = find_col(aliases+[canonical], df.columns)
    if found is None:
        raise ValueError(f"Could not locate expected column for '{canonical}'. Columns present: {list(df.columns)}")
    colmap[canonical] = found

df = df.rename(columns={v:k for k,v in colmap.items()})

# Keep only needed columns, preserve their order
df = df[["Usr Org","Make/Buy","CWS","Part-Number_sep","Part-Number_xm30","Part-Number_m10"]].copy()

# Coerce numeric columns
for c in ["CWS","Part-Number_sep","Part-Number_xm30","Part-Number_m10"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Normalize Make/Buy
df["Make/Buy"] = df["Make/Buy"].astype(str).str.upper().str.strip()
df["Make/Buy"] = df["Make/Buy"].replace({"M":"MAKE","B":"BUY"})  # just in case

# -----------------------------------------------------
# 3) QUICK HEALTH CHECK & BASIC LONG-FORM REORGANIZATION
# -----------------------------------------------------
summary = df.describe(include="all").transpose()

# Long format (useful for exploration; we’ll keep wide for modeling)
long_df = (
    df.melt(id_vars=["Usr Org","Make/Buy","CWS"],
            value_vars=["Part-Number_sep","Part-Number_xm30","Part-Number_m10"],
            var_name="ProgramCol", value_name="ProgramHours")
    .assign(Program=lambda d: d["ProgramCol"].str.replace("Part-Number_","",regex=False).str.upper())
    .drop(columns=["ProgramCol"])
)

# ---------------------------------------------
# 4) RATIO MODEL by (Usr Org, Make/Buy) GROUPS
# ---------------------------------------------
# We estimate: ProgramHours ≈ CWS * Ratio, where Ratio is avg(ProgramHours/CWS) within the group.
ratio_df = df.copy()
# Avoid divide-by-zero: only compute ratios where CWS>0 & target not null
def safe_ratio(num, den): 
    return num/den if (pd.notnull(num) and pd.notnull(den) and den>0) else np.nan

ratio_df["r_xm30"] = [safe_ratio(a,b) for a,b in zip(ratio_df["Part-Number_xm30"], ratio_df["CWS"])]
ratio_df["r_m10"]  = [safe_ratio(a,b) for a,b in zip(ratio_df["Part-Number_m10"],  ratio_df["CWS"])]

ratio_lookup = (
    ratio_df.groupby(["Usr Org","Make/Buy"], dropna=False)[["r_xm30","r_m10"]]
    .mean()
    .reset_index()
    .rename(columns={"r_xm30":"grp_ratio_xm30","r_m10":"grp_ratio_m10"})
)

df = df.merge(ratio_lookup, on=["Usr Org","Make/Buy"], how="left")

# Predictions via Ratio model
df["Pred_ratio_xm30"] = df["CWS"] * df["grp_ratio_xm30"]
df["Pred_ratio_m10"]  = df["CWS"] * df["grp_ratio_m10"]

# ------------------------------------------------
# 5) LINEAR REGRESSION MODEL with Org dummies, etc.
# ------------------------------------------------
# Features: CWS + Make/Buy + Usr Org (one-hot)
# Targets: Part-Number_xm30, Part-Number_m10
features = ["CWS","Make/Buy","Usr Org"]

# Build a pipeline that one-hot encodes categoricals and fits LinearRegression
categorical = ["Make/Buy","Usr Org"]
numeric = ["CWS"]

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), categorical),
        ("num", "passthrough", numeric),
    ],
    remainder="drop"
)

lin_xm30 = Pipeline(steps=[("pre", pre), ("lr", LinearRegression())])
lin_m10  = Pipeline(steps=[("pre", pre), ("lr", LinearRegression())])

# Train on rows that have targets present (avoid NaNs)
mask_x = df["Part-Number_xm30"].notna()
mask_m = df["Part-Number_m10"].notna()

lin_xm30.fit(df.loc[mask_x, features], df.loc[mask_x, "Part-Number_xm30"])
lin_m10.fit(df.loc[mask_m, features], df.loc[mask_m, "Part-Number_m10"])

df["Pred_lr_xm30"] = lin_xm30.predict(df[features])
df["Pred_lr_m10"]  = lin_m10.predict(df[features])

# Clip negative preds to 0 (hours cannot be negative)
for c in ["Pred_ratio_xm30","Pred_ratio_m10","Pred_lr_xm30","Pred_lr_m10"]:
    df[c] = df[c].clip(lower=0)

# --------------------------------------
# 6) EVALUATION (R2 / MAE) & COMPARISON
# --------------------------------------
def eval_metrics(y_true, y_pred, label):
    m = y_true.notna()
    if m.sum() == 0:
        return pd.Series({"Model": label, "R2": np.nan, "MAE": np.nan, "N": 0})
    return pd.Series({
        "Model": label,
        "R2": r2_score(y_true[m], y_pred[m]),
        "MAE": mean_absolute_error(y_true[m], y_pred[m]),
        "N": int(m.sum())
    })

eval_rows = []
eval_rows.append(eval_metrics(df["Part-Number_xm30"], df["Pred_ratio_xm30"], "XM30  • Ratio by (Org, Make/Buy)"))
eval_rows.append(eval_metrics(df["Part-Number_xm30"], df["Pred_lr_xm30"],    "XM30  • Linear Regression"))
eval_rows.append(eval_metrics(df["Part-Number_m10"],  df["Pred_ratio_m10"],  "M10   • Ratio by (Org, Make/Buy)"))
eval_rows.append(eval_metrics(df["Part-Number_m10"],  df["Pred_lr_m10"],     "M10   • Linear Regression"))
metrics_df = pd.DataFrame(eval_rows)

# Final comparison table (rounded for readability)
show_cols = ["Usr Org","Make/Buy","CWS",
             "Part-Number_xm30","Pred_ratio_xm30","Pred_lr_xm30",
             "Part-Number_m10","Pred_ratio_m10","Pred_lr_m10"]
pretty = df[show_cols].copy()
pretty = pretty.rename(columns={
    "Part-Number_xm30":"Actual_xm30",
    "Part-Number_m10":"Actual_m10"
})
pretty = pretty.round(2)

# ----------------
# 7) QUICK VISUALS
# ----------------
plt.figure(figsize=(7,4))
plt.title("Group Ratios: XM30/SEP and M10/SEP by (Usr Org, Make/Buy)")
x_ticks = np.arange(len(ratio_lookup))
plt.plot(x_ticks, ratio_lookup["grp_ratio_xm30"], marker="o", label="XM30 ratio")
plt.plot(x_ticks, ratio_lookup["grp_ratio_m10"], marker="s", label="M10 ratio")
plt.xticks(x_ticks, ratio_lookup["Usr Org"] + " | " + ratio_lookup["Make/Buy"], rotation=45, ha="right")
plt.ylabel("Avg Program Hours / SEP Hours")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter: Actual vs Pred (XM30)
plt.figure(figsize=(6,5))
m = df["Part-Number_xm30"].notna()
plt.scatter(df.loc[m,"Part-Number_xm30"], df.loc[m,"Pred_ratio_xm30"], label="Ratio model")
plt.scatter(df.loc[m,"Part-Number_xm30"], df.loc[m,"Pred_lr_xm30"], marker="x", label="Linear reg.")
max_x = float(np.nanmax(df.loc[m,"Part-Number_xm30"])) if m.any() else 1.0
plt.plot([0, max_x],[0, max_x])
plt.xlabel("Actual XM30 Hours")
plt.ylabel("Predicted XM30 Hours")
plt.title("XM30: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter: Actual vs Pred (M10)
plt.figure(figsize=(6,5))
m = df["Part-Number_m10"].notna()
plt.scatter(df.loc[m,"Part-Number_m10"], df.loc[m,"Pred_ratio_m10"], label="Ratio model")
plt.scatter(df.loc[m,"Part-Number_m10"], df.loc[m,"Pred_lr_m10"], marker="x", label="Linear reg.")
max_x = float(np.nanmax(df.loc[m,"Part-Number_m10"])) if m.any() else 1.0
plt.plot([0, max_x],[0, max_x])
plt.xlabel("Actual M10 Hours")
plt.ylabel("Predicted M10 Hours")
plt.title("M10: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# Residual distributions
fig, axs = plt.subplots(1,2, figsize=(10,4))
m = df["Part-Number_xm30"].notna()
axs[0].hist(df.loc[m,"Part-Number_xm30"] - df.loc[m,"Pred_lr_xm30"], bins=15)
axs[0].set_title("Residuals (XM30, Linear Reg.)")
axs[0].set_xlabel("Actual - Predicted")

m = df["Part-Number_m10"].notna()
axs[1].hist(df.loc[m,"Part-Number_m10"] - df.loc[m,"Pred_lr_m10"], bins=15)
axs[1].set_title("Residuals (M10, Linear Reg.)")
axs[1].set_xlabel("Actual - Predicted")
plt.tight_layout()
plt.show()

# ---------------------------
# 8) PRINT KEY TABLES/RESULTS
# ---------------------------
print("\n=== DATA HEALTH CHECK ===")
print(summary)

print("\n=== MODEL METRICS (higher R2, lower MAE are better) ===")
print(metrics_df.to_string(index=False))

print("\n=== PER-ROW COMPARISON (rounded) ===")
print(pretty.to_string(index=False))

# The df now contains:
#  - grp_ratio_xm30 / grp_ratio_m10 (learned per-group ratios)
#  - Pred_ratio_xm30 / Pred_ratio_m10  (ratio-model predictions)
#  - Pred_lr_xm30 / Pred_lr_m10        (linear-regression predictions)
# You can write df to a file if desired:
# df.to_excel("hours_with_predictions.xlsx", index=False)