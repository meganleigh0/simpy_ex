# --- XM30 Cobra → tidy dataset (and a couple quick visuals) ---
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ---------- Config ----------
COBRA_PATH = Path("data/Cobra-XM30.xlsx")               # input 1
DASHBOARD_PATH = Path("data/Dashboard-XM30_10_15_25.xlsx")  # (optional) input 2
OUT_SUMMARY_CSV = Path("data/weekly_summary_latest.csv")    # output for Streamlit
OUT_DETAIL_PARQUET = Path("data/weekly_detail.parquet")     # optional faster source

# ---------- Helpers ----------
def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip + unify a few expected names."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("#", "_num", regex=False)
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )
    # common renames we’ve seen in your sheets/screenshots
    rename_map = {
        "chg_num": "chg",
        "resp_dept": "resp_dept",
        "be_dept": "be_dept",
        "control_acct": "control_acct",
        "date": "date",
        "acwp": "acwp",
        "bcwp": "bcwp",
        "bcws": "bcws",
        "etc": "etc",
        "project": "project",
        "ce_project": "project",  # sometimes appears as CE PROJECT
        "family": "family",
        "team": "team",
        "sub_team": "sub_team"
    }
    for k, v in list(rename_map.items()):
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    return df

def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _parse_xl(xl_path: Path, sheet_name: str) -> pd.DataFrame:
    if not xl_path.exists():
        return pd.DataFrame()
    x = pd.ExcelFile(xl_path)
    if sheet_name not in x.sheet_names:
        return pd.DataFrame()
    df = x.parse(sheet_name)
    return _std_cols(df)

# ---------- Load ----------
tbl_we_1 = _parse_xl(COBRA_PATH, "tbl_Weekly Extract")
tbl_we_2 = _parse_xl(COBRA_PATH, "tbl_Weekly Extract (2)")
sheet1   = _parse_xl(COBRA_PATH, "Sheet1")
valid_w  = _parse_xl(COBRA_PATH, "Validation-W")
dash_tab = _parse_xl(DASHBOARD_PATH, "DASHBOARD")
dash_piv = _parse_xl(DASHBOARD_PATH, "PIVOT")

# ---------- Combine weekly extracts ----------
weekly_raw = pd.concat([tbl_we_1, tbl_we_2], ignore_index=True).pipe(_std_cols)

# best-effort type cleanup
weekly_raw["date"] = pd.to_datetime(weekly_raw.get("date"), errors="coerce")
weekly_raw = _numeric(weekly_raw, ["acwp", "bcwp", "bcws", "etc"])

# ---------- Attach metadata (project / family / team) ----------
# In some files these live on Sheet1
meta_cols = [c for c in ["control_acct", "project", "family", "team"] if c in sheet1.columns]
if "control_acct" in weekly_raw.columns and meta_cols:
    weekly = weekly_raw.merge(
        sheet1[meta_cols].drop_duplicates(),
        on="control_acct",
        how="left"
    )
else:
    weekly = weekly_raw.copy()
    for c in ["project", "family", "team"]:
        if c not in weekly.columns: weekly[c] = np.nan

# ---------- Clean dims and fill ----------
for c in ["project", "family", "team", "sub_team"]:
    if c in weekly.columns:
        weekly[c] = weekly[c].astype("string").fillna("Unassigned")

# ---------- Build summary (by date + project; extendable) ----------
summary_dims = ["date", "project"]  # change or add "family", "team" if you want wider controls
num_cols = [c for c in ["acwp", "bcwp", "bcws", "etc"] if c in weekly.columns]

weekly_summary = (
    weekly
    .dropna(subset=["date"])
    .groupby(summary_dims, dropna=False)[num_cols]
    .sum()
    .reset_index()
    .sort_values("date")
)

# ---------- Persist for the Streamlit app ----------
OUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
weekly_summary.to_csv(OUT_SUMMARY_CSV, index=False)
try:
    weekly.to_parquet(OUT_DETAIL_PARQUET, index=False)
except Exception:
    # parquet may not be available in some envs; it's optional
    pass

# ---------- Quick validation prints/figures ----------
print("Rows in weekly detail:", len(weekly))
print("Rows in weekly summary:", len(weekly_summary))
print("Date range:", weekly_summary["date"].min(), "→", weekly_summary["date"].max())
print(weekly_summary.tail(3))

# Trend line (earned value)
if not weekly_summary.empty:
    fig_trend = px.line(
        weekly_summary,
        x="date",
        y=[c for c in ["acwp","bcwp","bcws"] if c in weekly_summary.columns],
        color="project",
        markers=True,
        title="Earned Value Trends by Project"
    )
    fig_trend.show()

# Current week ETC by project
if not weekly_summary.empty and "etc" in weekly_summary.columns:
    latest_date = weekly_summary["date"].max()
    latest = weekly_summary[weekly_summary["date"] == latest_date]
    fig_etc = px.bar(
        latest,
        x="project",
        y="etc",
        title=f"ETC by Project — {latest_date.date()}",
        text_auto=".2s"
    )
    fig_etc.update_layout(xaxis_title="", yaxis_title="ETC")
    fig_etc.show()