# ===== ONE CELL: Load -> Clean -> Predict (ratio model) -> Evaluate -> Plot =====
# Requires: pandas, numpy, matplotlib

import re, numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------- 1) LOAD ----------
FILE_PATH = "your_file.xlsx"   # <-- change me (CSV or Excel both OK)

def _read_any(path):
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

df = _read_any(FILE_PATH).copy()

# ---------- 2) CLEAN / STANDARDIZE ----------
def norm(s): return re.sub(r"\s+", " ", str(s)).strip()
df.columns = [norm(c) for c in df.columns]

# Ensure the 6 expected columns exist (case-insensitive match)
want = ["Usr Org","Make/Buy","CWS","Part-Number_sep","Part-Number_xm30","Part-Number_m10"]
lower_map = {c.lower(): c for c in df.columns}
rename = {}
for w in want:
    if w in df.columns: 
        continue
    # fallback: find case-insensitive
    key = w.lower()
    if key in lower_map: 
        rename[lower_map[key]] = w
df = df.rename(columns=rename)

missing = [c for c in want if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nFound: {list(df.columns)}")

df = df[want].copy()

# Numeric coercion (hours & counts)
for c in ["CWS","Part-Number_sep","Part-Number_xm30","Part-Number_m10"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Normalize Make/Buy just in case
df["Make/Buy"] = df["Make/Buy"].astype(str).str.upper().str.strip().replace({"M":"MAKE","B":"BUY"})

# ---------- 3) RATIO MODEL (robust with fallbacks) ----------
# Ratios: target / CWS (only where CWS>0)
def _safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or den <= 0: 
        return np.nan
    return num / den

df["r_xm30"] = [ _safe_ratio(a,b) for a,b in zip(df["Part-Number_xm30"], df["CWS"]) ]
df["r_m10"]  = [ _safe_ratio(a,b) for a,b in zip(df["Part-Number_m10"],  df["CWS"]) ]

# Group medians by (Usr Org, Make/Buy)
grp = df.groupby(["Usr Org","Make/Buy"], dropna=False)[["r_xm30","r_m10"]].median().reset_index()
# Fallback medians by Make/Buy only
mk = df.groupby(["Make/Buy"], dropna=False)[["r_xm30","r_m10"]].median().reset_index().rename(
    columns={"r_xm30":"r_xm30_mk","r_m10":"r_m10_mk"}
)
# Global fallbacks
global_x = np.nanmedian(df["r_xm30"])
global_m = np.nanmedian(df["r_m10"])

# Attach group ratios
df = df.merge(grp, on=["Usr Org","Make/Buy"], how="left", suffixes=("","_grp"))

# Attach make/buy fallback
df = df.merge(mk, on="Make/Buy", how="left")

# Resolve final ratio to use (group -> make/buy -> global)
df["ratio_xm30"] = df["r_xm30_grp"].where(df["r_xm30_grp"].notna(),
                         df["r_xm30_mk"].where(df["r_xm30_mk"].notna(), global_x))
df["ratio_m10"]  = df["r_m10_grp"].where(df["r_m10_grp"].notna(),
                         df["r_m10_mk"].where(df["r_m10_mk"].notna(),  global_m))

# Predictions (0 if CWS<=0 or missing)
def _pred(cws, r):
    if pd.isna(cws) or cws <= 0 or pd.isna(r): 
        return 0.0
    return max(0.0, cws * r)

df["Pred_xm30"] = [ _pred(c, r) for c, r in zip(df["CWS"], df["ratio_xm30"]) ]
df["Pred_m10"]  = [ _pred(c, r) for c, r in zip(df["CWS"], df["ratio_m10"])  ]

# ---------- 4) EVALUATION ----------
def _mae(y, p):
    m = (~pd.isna(y)) & (~pd.isna(p))
    return float(np.mean(np.abs(y[m]-p[m]))) if m.any() else np.nan

def _mape(y, p):
    m = (~pd.isna(y)) & (~pd.isna(p)) & (y != 0)
    return float(100*np.mean(np.abs((y[m]-p[m])/y[m]))) if m.any() else np.nan

metrics = pd.DataFrame({
    "Target": ["XM30","M10"],
    "MAE":    [_mae(df["Part-Number_xm30"], df["Pred_xm30"]),
               _mae(df["Part-Number_m10"],  df["Pred_m10"])],
    "MAPE%":  [_mape(df["Part-Number_xm30"], df["Pred_xm30"]),
               _mape(df["Part-Number_m10"],  df["Pred_m10"])]
}).round(2)

# Pretty comparison
out = df[["Usr Org","Make/Buy","CWS",
          "Part-Number_xm30","Pred_xm30",
          "Part-Number_m10","Pred_m10"]].copy() \
       .rename(columns={"Part-Number_xm30":"Actual_xm30",
                        "Part-Number_m10":"Actual_m10"}) \
       .round(2)

print("\n=== MODEL METRICS (Ratio model) ===")
print(metrics.to_string(index=False))

print("\n=== PER-ROW COMPARISON (first 25) ===")
print(out.head(25).to_string(index=False))

# ---------- 5) PLOTS ----------
# Scatter: Actual vs Pred for XM30
xm_m = df["Part-Number_xm30"].notna()
plt.figure(figsize=(6,5))
plt.scatter(df.loc[xm_m,"Part-Number_xm30"], df.loc[xm_m,"Pred_xm30"])
mx = float(np.nanmax(df.loc[xm_m,"Part-Number_xm30"])) if xm_m.any() else 1.0
plt.plot([0,mx],[0,mx])
plt.title("XM30: Actual vs Predicted (Ratio model)")
plt.xlabel("Actual XM30 Hours"); plt.ylabel("Predicted XM30 Hours")
plt.tight_layout(); plt.show()

# Scatter: Actual vs Pred for M10
m_m = df["Part-Number_m10"].notna()
plt.figure(figsize=(6,5))
plt.scatter(df.loc[m_m,"Part-Number_m10"], df.loc[m_m,"Pred_m10"])
mx = float(np.nanmax(df.loc[m_m,"Part-Number_m10"])) if m_m.any() else 1.0
plt.plot([0,mx],[0,mx])
plt.title("M10: Actual vs Predicted (Ratio model)")
plt.xlabel("Actual M10 Hours"); plt.ylabel("Predicted M10 Hours")
plt.tight_layout(); plt.show()

# Optional: bar of learned ratios per (Org, Make/Buy)
ratios_view = grp.copy()
ratios_view["grp"] = ratios_view["Usr Org"] + " | " + ratios_view["Make/Buy"]
plt.figure(figsize=(8,4))
plt.bar(ratios_view["grp"], ratios_view["r_xm30"], label="XM30/SEP")
plt.bar(ratios_view["grp"], ratios_view["r_m10"],  bottom=0, alpha=0.5, label="M10/SEP")
plt.xticks(rotation=45, ha="right"); plt.ylabel("Median Ratio")
plt.title("Median Ratios by (Usr Org, Make/Buy)")
plt.legend(); plt.tight_layout(); plt.show()

# If you want to save the augmented data:
# df.to_excel("hours_with_predictions_ratio_only.xlsx", index=False)