# evms_dashboard.py (or a Jupyter cell)
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------- helpers --------------------------
def _fmt_money(x):
    if pd.isna(x): return ""
    return f"({abs(x):,.2f})" if float(x) < 0 else f"{x:,.2f}"

def _fmt_pct(x):
    if pd.isna(x): return ""
    return f"({abs(x):,.2f})%" if float(x) < 0 else f"{x:,.2f}%"

def _fmt_idx(x):
    if pd.isna(x): return ""
    return f"{float(x):.2f}"

def _coerce(df):
    out = df.copy()
    out = out.replace("Missing value", np.nan)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def detect_period_cols(df):
    """Return tuple (cum, fourwk, status) for columns in 'rounded'."""
    cols = {c.lower(): c for c in df.columns}
    get = lambda key: next((v for k,v in cols.items() if key in k), None)
    cum = get("cum") or next(iter(df.columns))
    fourwk = get("4wk") or get("4 week") or get("last 4") or None
    status = get("status") or get("period") or None
    return cum, fourwk, status

def detect_summary_rows(df):
    """Return tuple (cum, fourwk, status) for index labels in 'summary'."""
    idx = {str(i).lower(): i for i in df.index}
    get = lambda key: next((v for k,v in idx.items() if key in k), None)
    cum = get("cum") or get("cumulative")
    fourwk = get("4wk") or get("4 week") or get("last 4")
    status = get("status") or get("period")
    return cum, fourwk, status

MONEY_METRICS = {"ACWP","BCWP","BCWS","ETC","SV","CV","BAC","EAC","VAC","BCWR"}
PCT_METRICS   = {"SV%","CV%","VAC%"}
INDEX_METRICS = {"SPI","CPI","TCPI","BEI"}

def format_metrics_table(df):
    """Format a metrics table where metric names are the index."""
    dfn = _coerce(df)
    # return a display-only DataFrame with strings for each cell
    out = dfn.copy().astype(object)
    for r in out.index:
        for c in out.columns:
            val = dfn.loc[r, c]
            name = str(r).strip().upper()
            if name in MONEY_METRICS:
                out.loc[r, c] = _fmt_money(val)
            elif name in PCT_METRICS or name.endswith("%"):
                out.loc[r, c] = _fmt_pct(val)
            elif name in INDEX_METRICS:
                out.loc[r, c] = _fmt_idx(val)
            else:
                # default: number with 2 decimals
                out.loc[r, c] = "" if pd.isna(val) else f"{float(val):,.2f}"
    return out

def split_rounded(df):
    """Split rounded into multi-period rows and cumulative-only rows."""
    cum_col, fourwk_col, status_col = detect_period_cols(df)
    cols = [x for x in [cum_col, fourwk_col, status_col] if x in df.columns]
    dfn = _coerce(df[cols])
    # rows that have values only in cumulative column
    other_cols = [c for c in cols if c != cum_col]
    mask_cum_only = dfn[other_cols].isna().all(axis=1) & dfn[cum_col].notna()
    cum_only = df.loc[mask_cum_only, [cum_col]]
    multi    = df.loc[~mask_cum_only, cols]
    return multi, cum_only, cum_col, fourwk_col, status_col

def tidy_period_colnames(df, cum_col, fourwk_col, status_col):
    rename_map = {}
    if cum_col and cum_col in df.columns:     rename_map[cum_col] = "CUM"
    if fourwk_col and fourwk_col in df.columns: rename_map[fourwk_col] = "4 Wk"
    if status_col and status_col in df.columns: rename_map[status_col] = "Current Status"
    return df.rename(columns=rename_map)

def add_download(df, label, fname):
    st.download_button(f"Download {label} CSV", df.to_csv(index=True).encode("utf-8"),
                       file_name=fname, mime="text/csv", use_container_width=True)

# -------------------------- page --------------------------
st.set_page_config(page_title="EVMS Tables", layout="wide")

# CSS polish: section bars + soft cards
st.markdown("""
<style>
.section { margin: 0.25rem 0 0.75rem 0;}
.section > .bar {
    background: #FFD84D; color: #111; font-weight: 700;
    padding: 6px 10px; border-radius: 6px; display: inline-block;
    letter-spacing: 0.2px;
}
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px;
    padding: 8px 10px;
    margin-top: 0.35rem;
}
.small { font-size: 0.92rem; }
.stDataFrame { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

st.title("EVMS Status Dashboard")

# ---------- TOP ROW ----------
left, right = st.columns([1, 1.25])

# Left: summary in three stacked sections (CUM / 4 Wk / Current Status)
with left:
    cum_row, wk4_row, stat_row = detect_summary_rows(summary)

    def show_summary_block(title, row_key):
        if row_key is None: return
        st.markdown(f'<div class="section"><span class="bar">{title}</span></div>', unsafe_allow_html=True)
        row = _coerce(summary.loc[[row_key], ["ACWP","BCWP","BCWS","ETC"]])
        # display horizontally (original column order), but nicer numbers
        row_fmt = row.copy()
        for c in row_fmt.columns:
            row_fmt[c] = row_fmt[c].map(_fmt_money)
        st.container().markdown('<div class="card small">', unsafe_allow_html=True)
        st.dataframe(row_fmt, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    show_summary_block("CUM", cum_row)
    show_summary_block("4 Week", wk4_row)
    show_summary_block("Current Status Period", stat_row)

# Right: Key Metrics (Rounded) split + formatted
with right:
    st.subheader("Key Metrics (Rounded)")
    multi, cum_only, cum_col, fourwk_col, status_col = split_rounded(rounded)

    if not multi.empty:
        st.caption("Across Periods")
        multi_disp = tidy_period_colnames(multi, cum_col, fourwk_col, status_col)
        st.container().markdown('<div class="card small">', unsafe_allow_html=True)
        st.dataframe(format_metrics_table(multi_disp), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    if not cum_only.empty:
        label = f"Cumulative-Only ({'CUM' if cum_col else 'Cumulative'})"
        st.caption(label)
        cum_only_disp = tidy_period_colnames(cum_only, cum_col, fourwk_col, status_col)
        st.container().markdown('<div class="card small">', unsafe_allow_html=True)
        st.dataframe(format_metrics_table(cum_only_disp), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- DETAILED TABLES ----------
st.header("Detailed Tables")

tab1, tab2, tab3 = st.tabs(["Cumulative", "Last 4 Weeks", "Status Period"])

with tab1:
    st.caption(f"{len(grouped_cum):,} rows")
    st.dataframe(grouped_cum, width="stretch")
    add_download(grouped_cum, "Cumulative", "grouped_cumulative.csv")

with tab2:
    st.caption(f"{len(grouped_4wk):,} rows")
    st.dataframe(grouped_4wk, width="stretch")
    add_download(grouped_4wk, "Last 4 Weeks", "grouped_last_4_weeks.csv")

with tab3:
    st.caption(f"{len(grouped_status):,} rows")
    st.dataframe(grouped_status, width="stretch")
    add_download(grouped_status, "Status Period", "grouped_status_period.csv")