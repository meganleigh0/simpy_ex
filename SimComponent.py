# app.py
# Streamlit dashboard for precomputed DataFrames:
# grouped_cum, grouped_4wk, grouped_status, summary, rounded

import streamlit as st
import pandas as pd

st.set_page_config(page_title="EVMS Tables", layout="wide")

def split_rounded(df: pd.DataFrame):
    """Split rounded into multi-period rows and cumulative-only rows."""
    r = df.copy()

    # If "Missing value" strings exist, treat them as NaN
    r = r.replace("Missing value", pd.NA)

    # Coerce to numeric for robust missingness checks
    for c in r.columns:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    # Identify the cumulative column by name (fallback = first col)
    cum_col = next((c for c in r.columns if "cum" in c.lower()), r.columns[0])
    other_cols = [c for c in r.columns if c != cum_col]

    cumulative_only = r[r[other_cols].isna().all(axis=1) & r[cum_col].notna()]
    multi_period   = r.drop(index=cumulative_only.index)

    # Keep original order of rows
    cumulative_only = df.loc[cumulative_only.index, [cum_col]]
    multi_period    = df.loc[multi_period.index, df.columns.tolist()]

    return multi_period, cumulative_only, cum_col

def add_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=True).encode("utf-8")
    st.download_button(f"Download {label} CSV", csv, file_name=filename, mime="text/csv")

st.title("EVMS Status Dashboard")

# ---------- Top: Summary & Rounded ----------
left, right = st.columns([1, 1.25])

with left:
    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)
    add_download(summary, "Summary", "summary.csv")

with right:
    st.subheader("Key Metrics (Rounded)")
    multi_period, cumulative_only, cum_col = split_rounded(rounded)

    if not multi_period.empty:
        st.caption("Across Periods")
        st.dataframe(multi_period, use_container_width=True)
        add_download(multi_period, "Rounded (across periods)", "rounded_across_periods.csv")

    if not cumulative_only.empty:
        st.caption(f"Cumulative-Only Metrics (column: {cum_col})")
        st.dataframe(cumulative_only, use_container_width=True)
        add_download(cumulative_only, "Rounded (cumulative-only)", "rounded_cumulative_only.csv")

st.markdown("---")

# ---------- Detailed Tables ----------
st.header("Detailed Tables")
tab1, tab2, tab3, tab4 = st.tabs(["Cumulative", "Last 4 Weeks", "Status Period", "Rounded (raw)"])

with tab1:
    st.dataframe(grouped_cum, use_container_width=True)
    add_download(grouped_cum, "Cumulative", "grouped_cumulative.csv")

with tab2:
    st.dataframe(grouped_4wk, use_container_width=True)
    add_download(grouped_4wk, "Last 4 Weeks", "grouped_last_4_weeks.csv")

with tab3:
    st.dataframe(grouped_status, use_container_width=True)
    add_download(grouped_status, "Status Period", "grouped_status_period.csv")

with tab4:
    st.dataframe(rounded, use_container_width=True)
    add_download(rounded, "Rounded (raw)", "rounded_raw.csv")