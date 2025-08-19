# -*- coding: utf-8 -*-
"""
Unified BOM pipeline: load -> clean -> compare -> visualize
- Programs can be switched by changing PROGRAM_CONFIG at the top or via widgets at the bottom.
- Sources:
    Oracle MBOM: data/bronze_boms/{program}/{program}_mbom_oracle_{DATE}.xlsx
    TC MBOM    : data/bronze_boms_{program}/{program}_mbom_tc_{DATE}.xlsm
    TC EBOM    : data/bronze_boms_{program}/{program}_ebom_tc_{DATE}.xlsm
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
BASE = Path("data/bronze_boms")

# Fill this with your programs + dates (keep your existing lists):
PROGRAM_CONFIG = {
    # "xm30": {
    #     "dates": ["02-12-2025", "02-20-2025", ...],
    #     "no_header_dates": set(["02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"])
    # },
    "m10": {
        "dates": [
            "03-05-2025","03-17-2025","03-26-2025","03-27-2025",
            "04-02-2025","04-09-2025","04-22-2025","05-07-2025",
            "06-04-2025","06-09-2025","06-23-2025","06-30-2025",
            "07-07-2025","07-21-2025","07-28-2025","08-04-2025","08-11-2025"
        ],
        "no_header_dates": set(["02-12-2025","02-20-2025","02-26-2025","03-05-2025","03-17-2025"])  # keep your exception list here
    },
    # "cuas": {
    #     "dates": [...],
    #     "no_header_dates": set([...])
    # }
}

# Column name harmonization across files
COLMAP = {
    "Part Number": "PART_NUMBER",
    "PART_NUMBER": "PART_NUMBER",
    "Part Number*": "PART_NUMBER",

    "Item Name": "Description",
    "ITEM_NAME": "Description",

    "Make or Buy": "Make/Buy",
    "Make/Buy": "Make/Buy",
    "Make / Buy": "Make/Buy",
    "MAKE_OR_BUY": "Make/Buy",
    "Make /Buy": "Make/Buy",
    "Make/ Buy": "Make/Buy",

    "Level": "Levels",
    "# Level": "Levels",
    "# Levels": "Levels",
    "LEVEL": "Levels",
}

KEEP_COLS = ["PART_NUMBER", "Description", "Make/Buy", "Levels", "Date"]


# -----------------------------
# HELPERS
# -----------------------------
def _read_oracle(program: str, date: str, no_header_dates: set) -> pd.DataFrame:
    """Oracle MBOM for a given snapshot."""
    p = BASE / program / f"{program}_mbom_oracle_{date}.xlsx"
    if not p.exists():
        return pd.DataFrame(columns=KEEP_COLS)

    # some dates come without header row
    if date in no_header_dates:
        df = pd.read_excel(p, engine="openpyxl")
    else:
        df = pd.read_excel(p, engine="openpyxl", header=5)

    df = df.rename(columns=COLMAP, errors="ignore")
    # standardize set of columns we care about
    wanted = [c for c in ["PART_NUMBER","Part-Number","PART_NUMBER.","PART_NUMBER*","Item Name","Description","Make/Buy","Make or Buy","Level","Levels"] if c in df.columns]
    df = df[wanted].copy()

    df.rename(columns=COLMAP, inplace=True)
    df["Date"] = pd.to_datetime(date)
    return df[KEEP_COLS].drop_duplicates()


def _read_tc_mbom(program: str, date: str) -> pd.DataFrame:
    """Teamcenter MBOM."""
    p = Path(f"data/bronze_boms_{program}") / f"{program}_mbom_tc_{date}.xlsm"
    if not p.exists():
        return pd.DataFrame(columns=KEEP_COLS)

    df = pd.read_excel(p, engine="openpyxl")
    df = df.rename(columns=COLMAP, errors="ignore")
    wanted = [c for c in ["PART_NUMBER","Part Number","Item Name","Description","Make/Buy","Make or Buy","Level","Levels"] if c in df.columns]
    df = df[wanted].copy()
    df.rename(columns=COLMAP, inplace=True)
    df["Date"] = pd.to_datetime(date)
    return df[KEEP_COLS].drop_duplicates()


def _read_tc_ebom(program: str, date: str) -> pd.DataFrame:
    """Teamcenter EBOM."""
    p = Path(f"data/bronze_boms_{program}") / f"{program}_ebom_tc_{date}.xlsm"
    if not p.exists():
        return pd.DataFrame(columns=KEEP_COLS)

    df = pd.read_excel(p, engine="openpyxl")
    df = df.rename(columns=COLMAP, errors="ignore")
    wanted = [c for c in ["PART_NUMBER","Part Number","Item Name","Description","Make/Buy","Make or Buy","Level","Levels"] if c in df.columns]
    df = df[wanted].copy()
    df.rename(columns=COLMAP, inplace=True)
    df["Date"] = pd.to_datetime(date)
    return df[KEEP_COLS].drop_duplicates()


def _clean_make_buy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Make/Buy text and sort for time-series logic."""
    if df.empty:
        return df.assign(previous_status=pd.Series(dtype="str"))

    out = (
        df.copy()
          .assign(Date=lambda d: pd.to_datetime(d["Date"]))
          .assign(**{
              "Make/Buy": lambda d: (
                  d["Make/Buy"]
                  .astype(str).str.strip().str.lower()
                  .replace({"nan": np.nan})
                  .where(lambda s: s.isin(["make","buy"]))
              )
          })
          .sort_values(["PART_NUMBER","Date"])
          .drop_duplicates(subset=["PART_NUMBER","Date"], keep="last")
          .reset_index(drop=True)
    )
    # previous_status per part across snapshots
    out["previous_status"] = out.groupby("PART_NUMBER")["Make/Buy"].shift(1)
    return out


def detect_flips(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (flip_log, snapshot_summary) where:
      flip_log: rows where Make/Buy changed vs previous snapshot per part
      snapshot_summary: count of parts changed by Date
    """
    if df.empty:
        return (
            pd.DataFrame(columns=["PART_NUMBER","Description","Levels","Date","previous_status","new_status","Source"]),
            pd.DataFrame(columns=["Date","num_parts_changed","Source"]),
        )

    mask = df["Make/Buy"].notna() & df["previous_status"].notna() & df["Make/Buy"].ne(df["previous_status"])
    flip_log = (
        df.loc[mask, ["PART_NUMBER","Description","Levels","Date","previous_status","Make/Buy"]]
          .rename(columns={"Make/Buy":"new_status"})
          .assign(Source=source_name)
          .sort_values(["Date","PART_NUMBER"])
          .reset_index(drop=True)
    )
    snapshot_summary = (
        flip_log.groupby(["Date"], as_index=False)["PART_NUMBER"]
                .nunique()
                .rename(columns={"PART_NUMBER":"num_parts_changed"})
                .assign(Source=source_name)
                .sort_values("Date")
    )
    return flip_log, snapshot_summary


def compare_sources_per_date(oracle_df, tc_m_df, tc_e_df) -> pd.DataFrame:
    """
    For each Date:
      - in_oracle_not_tc_m
      - in_tc_m_not_oracle
      - in_tc_e_not_tc_m
      - common_all
    Returns tidy dataframe for plotting.
    """
    all_dates = sorted(set(oracle_df["Date"]) | set(tc_m_df["Date"]) | set(tc_e_df["Date"]))
    rows = []

    for dt in all_dates:
        o = set(oracle_df.loc[oracle_df["Date"].eq(dt), "PART_NUMBER"])
        m = set(tc_m_df.loc[tc_m_df["Date"].eq(dt), "PART_NUMBER"])
        e = set(tc_e_df.loc[tc_e_df["Date"].eq(dt), "PART_NUMBER"])

        rows.append({"Date": dt, "metric": "in_oracle_not_tc_m", "count": len(o - m)})
        rows.append({"Date": dt, "metric": "in_tc_m_not_oracle", "count": len(m - o)})
        rows.append({"Date": dt, "metric": "in_tc_e_not_tc_m", "count": len(e - m)})
        rows.append({"Date": dt, "metric": "common_all", "count": len(o & m & e)})

    return pd.DataFrame(rows).sort_values(["Date","metric"]).reset_index(drop=True)


def load_program(program: str, dates: list[str], no_header_dates: set) -> dict:
    """Load + clean + detect flips + compare for a single program."""
    oracle = []
    tc_m = []
    tc_e = []

    for d in dates:
        oracle.append(_read_oracle(program, d, no_header_dates))
        tc_m.append(_read_tc_mbom(program, d))
        tc_e.append(_read_tc_ebom(program, d))

    oracle = pd.concat(oracle, ignore_index=True) if oracle else pd.DataFrame(columns=KEEP_COLS)
    tc_m = pd.concat(tc_m, ignore_index=True) if tc_m else pd.DataFrame(columns=KEEP_COLS)
    tc_e = pd.concat(tc_e, ignore_index=True) if tc_e else pd.DataFrame(columns=KEEP_COLS)

    # Clean & flip detection
    oracle_c = _clean_make_buy(oracle)
    tc_m_c   = _clean_make_buy(tc_m)
    tc_e_c   = _clean_make_buy(tc_e)

    oracle_flips, oracle_flip_summary = detect_flips(oracle_c, "Oracle MBOM")
    tc_m_flips,  tc_m_flip_summary    = detect_flips(tc_m_c,  "TC MBOM")
    tc_e_flips,  tc_e_flip_summary    = detect_flips(tc_e_c,  "TC EBOM")

    compare_tidy = compare_sources_per_date(oracle, tc_m, tc_e)

    return {
        "oracle": oracle,
        "tc_mbom": tc_m,
        "tc_ebom": tc_e,
        "oracle_flips": oracle_flips,
        "tc_m_flips": tc_m_flips,
        "tc_e_flips": tc_e_flips,
        "flip_summary": pd.concat([oracle_flip_summary, tc_m_flip_summary, tc_e_flip_summary], ignore_index=True),
        "compare_tidy": compare_tidy,
    }


# -----------------------------
# DRIVER (non-interactive)
# -----------------------------
def run_pipeline(program: str) -> dict:
    cfg = PROGRAM_CONFIG[program]
    return load_program(program, cfg["dates"], cfg.get("no_header_dates", set()))


# -----------------------------
# VISUALIZATION (Plotly)
# -----------------------------
# If you prefer Matplotlib, adapt this section; keeping Plotly since your notebook already uses it.
import plotly.express as px
import ipywidgets as W
from IPython.display import display

def make_figs(result: dict, title_prefix: str = ""):
    # 1) Flip counts by date (stacked by Source)
    if not result["flip_summary"].empty:
        f1 = px.bar(
            result["flip_summary"],
            x="Date", y="num_parts_changed", color="Source",
            barmode="group",
            title=f"{title_prefix}Make/Buy flips by snapshot"
        )
    else:
        f1 = None

    # 2) Coverage comparison per snapshot
    if not result["compare_tidy"].empty:
        f2 = px.line(
            result["compare_tidy"],
            x="Date", y="count", color="metric",
            markers=True,
            title=f"{title_prefix}Cross-source comparison (per snapshot)"
        )
    else:
        f2 = None

    # 3) Make/Buy composition by snapshot for each source (optional)
    comp_rows = []
    for label, df in [("Oracle MBOM", result["oracle"]), ("TC MBOM", result["tc_mbom"]), ("TC EBOM", result["tc_ebom"])]:
        if df.empty: 
            continue
        tmp = (df.assign(n=1)
                 .groupby(["Date","Make/Buy"], as_index=False)["n"].sum()
                 .assign(Source=label))
        comp_rows.append(tmp)
    comp = pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame(columns=["Date","Make/Buy","n","Source"])

    if not comp.empty:
        f3 = px.bar(comp, x="Date", y="n", color="Make/Buy", facet_row="Source",
                    title=f"{title_prefix}Make/Buy composition per snapshot")
    else:
        f3 = None

    return [f for f in [f1, f2, f3] if f is not None]


# -----------------------------
# Simple program selector UI
# -----------------------------
def interactive_dashboard():
    prog_dd = W.Dropdown(options=list(PROGRAM_CONFIG.keys()), description="Program:", value=list(PROGRAM_CONFIG.keys())[0])
    out = W.Output()

    def _render(change=None):
        with out:
            out.clear_output()
            res = run_pipeline(prog_dd.value)
            figs = make_figs(res, title_prefix=f"{prog_dd.value}: ")
            if figs:
                for fig in figs:
                    fig.show()
            else:
                display("No data to plot for this selection.")

            # also show the flip log table for traceability
            flip_log_all = pd.concat([res["oracle_flips"], res["tc_m_flips"], res["tc_e_flips"]], ignore_index=True)
            if not flip_log_all.empty:
                display(flip_log_all.sort_values(["Date","Source","PART_NUMBER"]).reset_index(drop=True))

    prog_dd.observe(_render, names="value")
    display(prog_dd)
    _render()

# Call this in a notebook cell to use:
# interactive_dashboard()