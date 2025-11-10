# --- Totals summary from grouped_status, grouped_4wk, grouped_cum (one cell) ---
import pandas as pd

# Pick the metric columns in a fixed order if they exist
_metric_order = ["ACWP", "BCWP", "BCWS", "ETC"]
def _available_metrics(*dfs):
    present = []
    for c in _metric_order:
        if any(c in df.columns for df in dfs):
            present.append(c)
    return present

def _total_series(df: pd.DataFrame, cols):
    """Return TOTAL row for df over `cols`.
       If the last row is the TOTAL row, use it; else compute the sum."""
    if len(df) == 0:
        return pd.Series({c: 0.0 for c in cols})
    last_idx = str(df.index[-1]).upper()
    if last_idx == "TOTAL" and set(cols).issubset(df.columns):
        s = df.iloc[-1][cols]
    else:
        s = df[cols].sum(numeric_only=True)
    # ensure numeric
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s.index = cols
    return s

cols = _available_metrics(grouped_status, grouped_4wk, grouped_cum)

summary = pd.DataFrame(
    [
        _total_series(grouped_status, cols),
        _total_series(grouped_4wk, cols),
        _total_series(grouped_cum, cols),
    ],
    index=["Status Period", "Last 4 Weeks", "Cumulative"],
)[cols]

# Optional: nicer display (round if you like)
summary = summary.astype(float)

summary