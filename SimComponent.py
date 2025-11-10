# evms_dashboard.py
# Assumes these DataFrames already exist in memory:
#   grouped_cum, grouped_4wk, grouped_status, summary, rounded

import pandas as pd
import streamlit as st

def split_rounded(df: pd.DataFrame):
    r = df.copy().replace("Missing value", pd.NA)
    # find the cumulative column by name; fall back to first column
    cum_candidates = [c for c in r.columns if "cum" in c.lower()]
    cum_col = cum_candidates[0] if cum_candidates else r.columns[0]
    other_cols = [c for c in r.columns if c != cum_col]

    # coerce to numeric to detect real missingness
    for c in other_cols:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    mask_cum_only = r[other_cols].isna().all(axis=1) & r[cum_col].notna()
    cumulative_only = df.loc[mask_cum_only, [cum_col]]
    multi_period = df.loc[~mask_cum_only, df.columns]
    return multi_period, cumulative_only, cum_col

def app():
    st.set_page_config(page_title="EVMS Tables", layout="wide")
    st.title("EVMS Status Dashboard")

    col1, col2 = st.columns([1, 1.25])
    with col1:
        st.subheader("Summary")
        st.dataframe(summary, width="stretch")

    with col2:
        st.subheader("Key Metrics (Rounded)")
        multi, cum_only, cum_col = split_rounded(rounded)
        if not multi.empty:
            st.caption("Across Periods")
            st.dataframe(multi, width="stretch")
        if not cum_only.empty:
            st.caption(f"Cumulative-Only ({cum_col})")
            st.dataframe(cum_only, width="stretch")

    st.markdown("---")
    st.header("Detailed Tables")
    tab1, tab2, tab3, tab4 = st.tabs(["Cumulative", "Last 4 Weeks", "Status Period", "Rounded (raw)"])
    with tab1:
        st.dataframe(grouped_cum, width="stretch")
    with tab2:
        st.dataframe(grouped_4wk, width="stretch")
    with tab3:
        st.dataframe(grouped_status, width="stretch")
    with tab4:
        st.dataframe(rounded, width="stretch")

if __name__ == "__main__":
    # If not under Streamlit, render a notebook-friendly preview without warnings.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if in_streamlit:
        app()
    else:
        from IPython.display import display, HTML
        display(HTML("<h2>EVMS Status Dashboard (Notebook Preview)</h2>"))
        display(HTML("<h3>Summary</h3>")); display(summary)
        multi, cum_only, cum_col = split_rounded(rounded)
        display(HTML("<h3>Rounded — Across Periods</h3>")); display(multi)
        display(HTML(f"<h3>Rounded — Cumulative-Only ({cum_col})</h3>")); display(cum_only)
        display(HTML("<hr/><h3>Cumulative</h3>")); display(grouped_cum)
        display(HTML("<h3>Last 4 Weeks</h3>")); display(grouped_4wk)
        display(HTML("<h3>Status Period</h3>")); display(grouped_status)
        display(HTML("<h3>Rounded (raw)</h3>")); display(rounded)