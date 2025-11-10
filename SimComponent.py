# evms_dashboard.py
import pandas as pd
import streamlit as st

# ---------------- helpers ----------------
def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().replace("Missing value", pd.NA)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def split_rounded(df: pd.DataFrame):
    r = coerce_numeric(df)
    # detect the cumulative column (fallback to first col)
    cum_col = next((c for c in r.columns if "cum" in c.lower()), r.columns[0])
    other_cols = [c for c in r.columns if c != cum_col]
    mask_cum_only = r[other_cols].isna().all(axis=1) & r[cum_col].notna()
    cumulative_only = df.loc[mask_cum_only, [cum_col]]
    multi_period = df.loc[~mask_cum_only, df.columns]
    return multi_period, cumulative_only, cum_col

def metric_value(rounded_df, name, period_col):
    # robust fetch: works even if not present
    try:
        return float(coerce_numeric(rounded_df).loc[name, period_col])
    except Exception:
        return None

def add_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=True).encode("utf-8")
    st.download_button(f"Download {label} CSV", csv, file_name=filename, mime="text/csv")

def bar_chart(data_dict, title, y_title="Value"):
    """data_dict: {metric_name: {'Cumulative': x, 'Last 4 Weeks': y, 'Status Period': z}, ...}"""
    df = (
        pd.DataFrame(data_dict)
        .rename_axis("Period")
        .reset_index()
        .melt(id_vars="Period", var_name="Metric", value_name="Value")
        .dropna()
    )
    if df.empty:
        st.info("No data available for this chart.")
        return
    # Use built-in Streamlit chart (works without extra deps)
    st.caption(title)
    # pivot into wide form for multi-series bars by period
    wide = df.pivot(index="Period", columns="Metric", values="Value")
    st.bar_chart(wide, width="stretch", height=260)

# ---------------- app ----------------
def app():
    st.set_page_config(page_title="EVMS Tables", layout="wide")
    st.title("EVMS Status Dashboard")

    # ---- TOP: Summary & Rounded ----
    col1, col2 = st.columns([1, 1.25])

    with col1:
        st.subheader("Summary")
        st.dataframe(summary, width="stretch")
        add_download(summary, "Summary", "summary.csv")

    with col2:
        st.subheader("Key Metrics (Rounded)")
        multi_period, cumulative_only, cum_col = split_rounded(rounded)

        # KPI cards (Cumulative)
        kpi_cols = st.columns(4)
        spi_c = metric_value(rounded, "SPI", cum_col)
        cpi_c = metric_value(rounded, "CPI", cum_col)
        sv_c  = metric_value(rounded, "SV",  cum_col)
        cv_c  = metric_value(rounded, "CV",  cum_col)

        kpi_cols[0].metric("SPI (Cumulative)", f"{spi_c:.2f}" if spi_c is not None else "—")
        kpi_cols[1].metric("CPI (Cumulative)", f"{cpi_c:.2f}" if cpi_c is not None else "—")
        kpi_cols[2].metric("SV (Cumulative)",  f"{sv_c:,.0f}"  if sv_c  is not None else "—")
        kpi_cols[3].metric("CV (Cumulative)",  f"{cv_c:,.0f}"  if cv_c  is not None else "—")

        # Tables
        if not multi_period.empty:
            st.caption("Across Periods")
            st.dataframe(multi_period, width="stretch")
        if not cumulative_only.empty:
            st.caption(f"Cumulative-Only ({cum_col})")
            st.dataframe(cumulative_only, width="stretch")

        # Small visuals from Rounded
        bar_chart(
            {
                "SPI": {
                    c: metric_value(rounded, "SPI", c)
                    for c in [cum_col, "Last 4 Weeks", "Status Period"]
                    if c in rounded.columns
                },
                "CPI": {
                    c: metric_value(rounded, "CPI", c)
                    for c in [cum_col, "Last 4 Weeks", "Status Period"]
                    if c in rounded.columns
                },
            },
            title="SPI & CPI by Period"
        )

        bar_chart(
            {
                "SV": {
                    c: metric_value(rounded, "SV", c)
                    for c in [cum_col, "Last 4 Weeks", "Status Period"]
                    if c in rounded.columns
                },
                "CV": {
                    c: metric_value(rounded, "CV", c)
                    for c in [cum_col, "Last 4 Weeks", "Status Period"]
                    if c in rounded.columns
                },
            },
            title="SV & CV by Period"
        )

    st.markdown("---")

    # ---- Detailed Tables (no 'Rounded raw' tab) ----
    st.header("Detailed Tables")
    tab1, tab2, tab3 = st.tabs(["Cumulative", "Last 4 Weeks", "Status Period"])

    with tab1:
        st.dataframe(grouped_cum, width="stretch")
        add_download(grouped_cum, "Cumulative", "grouped_cumulative.csv")

    with tab2:
        st.dataframe(grouped_4wk, width="stretch")
        add_download(grouped_4wk, "Last 4 Weeks", "grouped_last_4_weeks.csv")

    with tab3:
        st.dataframe(grouped_status, width="stretch")
        add_download(grouped_status, "Status Period", "grouped_status_period.csv")

# Run app (works in both streamlit run and notebook preview)
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if in_streamlit:
        app()
    else:
        # Notebook preview (no warnings)
        from IPython.display import display, HTML
        display(HTML("<h2>EVMS Status Dashboard (Notebook Preview)</h2>"))
        display(HTML("<h3>Summary</h3>")); display(summary)
        mp, co, cc = split_rounded(rounded)
        display(HTML("<h3>Rounded — Across Periods</h3>")); display(mp)
        display(HTML(f"<h3>Rounded — Cumulative-Only ({cc})</h3>")); display(co)
        display(HTML("<hr/><h3>Cumulative</h3>")); display(grouped_cum)
        display(HTML("<h3>Last 4 Weeks</h3>")); display(grouped_4wk)
        display(HTML("<h3>Status Period</h3>")); display(grouped_status)