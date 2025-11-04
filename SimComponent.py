# --- ONE-CELL PIPELINE: SEP -> model -> predict XM30 hours by org (with Plotly) ---
import pandas as pd, numpy as np, re, pathlib, textwrap
from typing import Dict, List
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import PoissonRegressor
import plotly.express as px

# =========================
# CONFIG: set your inputs
# =========================
CONFIG = {
    # File paths (edit these)
    "sep_teamcenter_path": "data/sep_v3_mbom_tc_10-13-2025.xlsx",
    "sep_oracle_path":     "data/sep_v3_mbom_oracle_10-13-2025.xlsx",
    "xm30_teamcenter_path":"data/xm30_mbom_tc_10-13-2025.xlsx",
    "xm30_oracle_path":    "data/xm30_mbom_oracle_10-13-2025.xlsx",
    # Labor standards (hours per part). Must contain a part key + hours column.
    "standards_path":      "data/standards.csv",

    # Column name mapping (left = your file’s column; right = canonical name we’ll use)
    # Adjust ONLY the right-hand side targets if your inputs differ.
    # Required canonicals:
    #   PART_NUMBER, Qty, Make or Buy, Usr Org, Src Org, PlanNo  (PlanNo is used to join standards if standards are keyed by PlanNo)
    "colmap": {
        # Examples from your screenshots—edit as needed:
        "Part Number": "PART_NUMBER",
        "Part-Number": "PART_NUMBER",
        "PART_NUMBER": "PART_NUMBER",
        "Qty": "Qty",
        "Quantity": "Qty",
        "Make/Buy": "Make or Buy",
        "Make or Buy": "Make or Buy",
        "Usr Org": "Usr Org",
        "User Org": "Usr Org",
        "Src Org": "Src Org",
        "PlanNo": "PlanNo",
        # Standards file
        "CWS": "CWS",
    },

    # Optional: skip rows for Excel inputs (per file pattern)
    "skiprows_excel": 0,

    # Oracle is authoritative for org fields; TeamCenter is master for part lists
    "org_fill": "Unknown",
}

# =========================
# Helpers
# =========================
def _clean_cols(df: pd.DataFrame, colmap: Dict[str,str]) -> pd.DataFrame:
    # Normalize incoming column names and map to canonicals
    raw_to_clean = {c: re.sub(r"\s+", " ", str(c)).strip() for c in df.columns}
    df = df.rename(columns=raw_to_clean)
    # Build map for columns present
    usemap = {k: v for k, v in colmap.items() if k in df.columns}
    df = df.rename(columns=usemap)
    return df

def _load_any(path: str, skiprows: int=0) -> pd.DataFrame:
    p = pathlib.Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p, skiprows=skiprows)
    elif p.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

def load_bom(path: str, colmap: Dict[str,str], skiprows_excel: int=0) -> pd.DataFrame:
    df = _load_any(path, skiprows_excel)
    df = _clean_cols(df, colmap).copy()
    # lightweight cleaning
    if "PART_NUMBER" in df.columns:
        df["PART_NUMBER"] = df["PART_NUMBER"].astype(str).str.strip().str.lower()
    if "Make or Buy" in df.columns:
        df["Make or Buy"] = df["Make or Buy"].astype(str).str.strip().str.title()
        df["Make or Buy"] = df["Make or Buy"].replace({"M":"Make","B":"Buy"})
    if "Qty" in df.columns:
        # coerce quantities
        df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    # Org columns (may be absent in TC)
    for c in ["Usr Org","Src Org"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
    return df

def summarize_sizes(name: str, df: pd.DataFrame):
    print(f"\n=== {name} ===")
    print(df.shape)
    if "PART_NUMBER" in df.columns:
        print("unique PART_NUMBER:", df["PART_NUMBER"].nunique())
    if "Make or Buy" in df.columns:
        print(df["Make or Buy"].value_counts(dropna=False))
    if "Usr Org" in df.columns:
        print("orgs:", df["Usr Org"].nunique())

def aggregate_org_features(oracle_df: pd.DataFrame) -> pd.DataFrame:
    """Create org-level feature pack."""
    d = oracle_df.copy()
    d["Usr Org"] = d.get("Usr Org", pd.Series(index=d.index)).fillna(CONFIG["org_fill"])
    d["Make or Buy"] = d.get("Make or Buy", pd.Series(index=d.index)).fillna("Unknown")
    d["Qty"] = pd.to_numeric(d.get("Qty", 0), errors="coerce").fillna(0)

    # per part indicators
    d["is_make"] = (d["Make or Buy"].str.title() == "Make").astype(int)
    d["is_buy"]  = (d["Make or Buy"].str.title() == "Buy").astype(int)

    # build features
    g = d.groupby("Usr Org")
    feat = pd.DataFrame({
        "OrgUniqueParts": g["PART_NUMBER"].nunique(),
        "MakePartCount": g["is_make"].sum(),
        "BuyPartCount": g["is_buy"].sum(),
        "MakeQty": g.apply(lambda x: x.loc[x["is_make"]==1, "Qty"].sum()),
        "BuyQty":  g.apply(lambda x: x.loc[x["is_buy"]==1,  "Qty"].sum()),
        "TotalQty": g["Qty"].sum(),
    }).reset_index()

    # Rates / ratios (safe divisions)
    feat["MakeShareParts"] = np.divide(feat["MakePartCount"], feat["OrgUniqueParts"], out=np.zeros_like(feat["MakePartCount"], dtype=float), where=feat["OrgUniqueParts"]>0)
    feat["BuyShareParts"]  = 1 - feat["MakeShareParts"]
    feat["MakeShareQty"]   = np.divide(feat["MakeQty"], feat["TotalQty"], out=np.zeros_like(feat["MakeQty"], dtype=float), where=feat["TotalQty"]>0)
    feat["BuyShareQty"]    = 1 - feat["MakeShareQty"]
    return feat

def attach_labor_hours(oracle_df: pd.DataFrame, standards_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join labor standards (hours per part) onto Oracle rows, then produce
    org-level total hours = sum(hours_per_part * Qty).
    We try to join on PART_NUMBER; if standards only has PlanNo, we first propagate PlanNo from Oracle if present.
    """
    d = oracle_df.copy()

    # Normalize standards
    s = standards_df.rename(columns={c:c for c in standards_df.columns})
    s = _clean_cols(s, CONFIG["colmap"])
    # Key selection: prefer PART_NUMBER; else PlanNo
    join_key = "PART_NUMBER" if "PART_NUMBER" in s.columns else ("PlanNo" if "PlanNo" in s.columns else None)
    if join_key is None:
        raise ValueError("standards must have PART_NUMBER or PlanNo plus a CWS column")

    # If standards keyed by PlanNo, we need PlanNo on Oracle rows
    if join_key == "PlanNo" and "PlanNo" not in d.columns:
        # If not present, we cannot map—user may need to merge PlanNo beforehand.
        # We’ll proceed but warn.
        print("WARNING: PlanNo not present on Oracle; labor join may be empty.")
    # Perform join
    d = _clean_cols(d, CONFIG["colmap"])
    d["Qty"] = pd.to_numeric(d.get("Qty", 0), errors="coerce").fillna(0)
    s["CWS"] = pd.to_numeric(s["CWS"], errors="coerce").fillna(0)

    d = d.merge(s[[join_key,"CWS"]].drop_duplicates(), on=join_key, how="left")
    d["Usr Org"] = d.get("Usr Org", pd.Series(index=d.index)).fillna(CONFIG["org_fill"])
    d["Hours"] = d["CWS"] * d["Qty"]  # total hours contribution per row
    org_hours = d.groupby("Usr Org")["Hours"].sum().reset_index().rename(columns={"Hours":"OrgTotalHours"})
    return org_hours, d

def make_model(X: pd.DataFrame, y: pd.Series):
    """
    Try Ridge (with CV) vs Poisson; pick best on CV MAE.
    Return fitted best pipeline and a small report.
    """
    cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)

    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", RidgeCV(alphas=np.logspace(-3,3,21), cv=cv))
    ])
    # Neg MAE (higher is better). Convert to MAE for readability.
    ridge_scores = -cross_val_score(ridge, X, y, scoring="neg_mean_absolute_error", cv=cv)
    ridge.fit(X, y)

    # Poisson (nonnegative target; model expects y>=0)
    y_nonneg = y.clip(lower=0)
    poisson = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", PoissonRegressor(alpha=1.0, max_iter=500))
    ])
    pois_scores = -cross_val_score(poisson, X, y_nonneg, scoring="neg_mean_absolute_error", cv=cv)
    poisson.fit(X, y_nonneg)

    # pick best
    mean_ridge, mean_pois = ridge_scores.mean(), pois_scores.mean()
    if mean_pois <= mean_ridge:
        best, which, mae_cv = poisson, "PoissonRegressor", mean_pois
    else:
        best, which, mae_cv = ridge, "RidgeCV", mean_ridge

    report = {
        "chosen_model": which,
        "cv_mae": float(mae_cv),
        "ridge_cv_mae": float(mean_ridge),
        "poisson_cv_mae": float(mean_pois)
    }
    return best, report

# =========================
# PIPELINE
# =========================
# Load
sep_tc   = load_bom(CONFIG["sep_teamcenter_path"], CONFIG["colmap"], CONFIG["skiprows_excel"])
sep_orcl = load_bom(CONFIG["sep_oracle_path"],     CONFIG["colmap"], CONFIG["skiprows_excel"])
xm_tc    = load_bom(CONFIG["xm30_teamcenter_path"],CONFIG["colmap"], CONFIG["skiprows_excel"])
xm_orcl  = load_bom(CONFIG["xm30_oracle_path"],    CONFIG["colmap"], CONFIG["skiprows_excel"])
standards= load_bom(CONFIG["standards_path"],      CONFIG["colmap"], CONFIG["skiprows_excel"])

# Shapes / sanity
summarize_sizes("SEP TeamCenter", sep_tc)
summarize_sizes("SEP Oracle",     sep_orcl)
summarize_sizes("XM30 TeamCenter",xm_tc)
summarize_sizes("XM30 Oracle",    xm_orcl)

# TeamCenter as master parts list (optional diagnostics)
def group_parts(df, label):
    g = (df.assign(Qty=pd.to_numeric(df.get("Qty",0), errors="coerce").fillna(0))
           .groupby("PART_NUMBER", as_index=False)["Qty"].sum())
    g["Source"] = label
    return g

sep_tc_parts = group_parts(sep_tc, "SEP_TC")
xm_tc_parts  = group_parts(xm_tc, "XM30_TC")

# Oracle provides org info—aggregate org features
sep_org_feat = aggregate_org_features(sep_orcl)
xm_org_feat  = aggregate_org_features(xm_orcl)

# Attach labor standards to SEP rows -> org-level target hours
sep_org_hours, sep_orcl_with_hours = attach_labor_hours(sep_orcl, standards)

# Join features+target for training
train = sep_org_feat.merge(sep_org_hours, on="Usr Org", how="inner").fillna(0)
feature_cols = [
    "OrgUniqueParts","MakePartCount","BuyPartCount",
    "MakeQty","BuyQty","TotalQty","MakeShareParts","BuyShareParts","MakeShareQty","BuyShareQty"
]
X_train = train[feature_cols]
y_train = train["OrgTotalHours"]

# Train model
model, model_report = make_model(X_train, y_train)
print("\nModel selection:", model_report)

# Predict for XM30 orgs
X_xm = xm_org_feat[feature_cols].fillna(0)
xm30_pred = xm_org_feat[["Usr Org"]].copy()
xm30_pred["PredictedHours"] = model.predict(X_xm)
xm30_pred = xm30_pred.sort_values("PredictedHours", ascending=False).reset_index(drop=True)

# ---------- Diagnostics / Outputs ----------
print("\n=== TRAIN set diagnostics (SEP) ===")
if len(train) >= 2:
    yhat_train = model.predict(X_train)
    print("MAE:", mean_absolute_error(y_train, yhat_train))
    print("R^2:", r2_score(y_train, yhat_train))
else:
    print("Not enough orgs for diagnostics (need >=2).")

print("\n=== Top XM30 predicted hours by org ===")
print(xm30_pred.head(20))

# ---------- Tables to inspect sizes ----------
sep_sizes = sep_orcl.groupby(["Usr Org","Make or Buy"]).agg(
    Parts=("PART_NUMBER","nunique"), Qty=("Qty","sum")
).reset_index()
xm_sizes = xm_orcl.groupby(["Usr Org","Make or Buy"]).agg(
    Parts=("PART_NUMBER","nunique"), Qty=("Qty","sum")
).reset_index()

# ---------- Plotly visuals ----------
# 1) SEP: Actual hours by org (bar)
if not sep_org_hours.empty:
    fig1 = px.bar(sep_org_hours.sort_values("OrgTotalHours", ascending=False),
                  x="Usr Org", y="OrgTotalHours",
                  title="SEP – Actual Total Labor Hours by Org")
    fig1.show()

# 2) XM30: Predicted hours by org (bar)
fig2 = px.bar(xm30_pred, x="Usr Org", y="PredictedHours",
              title="XM30 – Predicted Total Labor Hours by Org")
fig2.show()

# 3) SEP vs XM30 make/buy footprint (stacked by Qty share)
def make_share_table(sizes):
    # convert to share within org
    t = sizes.pivot_table(index="Usr Org", columns="Make or Buy", values="Qty", aggfunc="sum", fill_value=0).reset_index()
    for c in ["Buy","Make"]:
        if c not in t.columns: t[c] = 0.0
    t["TotalQty"] = t["Buy"] + t["Make"]
    t["MakeShareQty"] = np.divide(t["Make"], t["TotalQty"], out=np.zeros_like(t["Make"]), where=t["TotalQty"]>0)
    t["BuyShareQty"]  = 1 - t["MakeShareQty"]
    return t[["Usr Org","MakeShareQty","BuyShareQty","TotalQty"]]

sep_shares = make_share_table(sep_sizes)
xm_shares  = make_share_table(xm_sizes)

fig3 = px.bar(sep_shares.sort_values("MakeShareQty", ascending=False),
              x="Usr Org", y=["MakeShareQty","BuyShareQty"],
              title="SEP – Make/Buy Share of Qty by Org", barmode="stack")
fig3.show()

fig4 = px.bar(xm_shares.sort_values("MakeShareQty", ascending=False),
              x="Usr Org", y=["MakeShareQty","BuyShareQty"],
              title="XM30 – Make/Buy Share of Qty by Org", barmode="stack")
fig4.show()

# 4) Feature importance proxy (Ridge coefficients or Poisson coefficients after scaling)
try:
    # grab final linear step
    coefs = getattr(model.named_steps["model"], "coef_", None)
    if coefs is not None:
        coef_df = (pd.DataFrame({"feature": feature_cols, "coef": coefs.ravel()})
                   .sort_values("coef", ascending=False))
        fig5 = px.bar(coef_df, x="feature", y="coef", title=f"Model Coefficients – {model_report['chosen_model']}")
        fig5.update_layout(xaxis_tickangle=-45)
        fig5.show()
except Exception as e:
    print("Skipped coefficient plot:", e)

# ---------- Handy answers ----------
def estimate_total_hours_for(org_name: str) -> float:
    """Quick Q&A API: get XM30 predicted hours for a single org string."""
    row = xm30_pred.loc[xm30_pred["Usr Org"].str.lower()==org_name.lower()]
    return float(row["PredictedHours"].iloc[0]) if len(row) else np.nan

def explain_org(org_name: str) -> pd.DataFrame:
    """Return the feature row used for prediction for the given org (XM30)."""
    row = xm_org_feat.loc[xm_org_feat["Usr Org"].str.lower()==org_name.lower(), ["Usr Org"] + feature_cols]
    return row.reset_index(drop=True)

print("\nTry: estimate_total_hours_for('LIM')  |  explain_org('LIM')")