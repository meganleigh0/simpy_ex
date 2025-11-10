limport pandas as pd

# ---- CONFIG: your fixed monthly accounting close dates (MM-DD)
ACCOUNTING_CLOSE_MD = [(1,26),(2,23),(3,30),(4,27),(5,25),(6,29),
                       (7,27),(8,24),(9,28),(10,26),(11,23),(12,31)]

# ---- OPTIONAL: to reproduce a past run, set OVERRIDE_TODAY = 'YYYY-MM-DD' (else None for actual today)
OVERRIDE_TODAY = None  # e.g., '2025-10-15'  -> most recent close = 2025-09-28

# ---- Pick the source DF and date column
df = xm30_cobra_export_weekly_extract.copy()
date_col = 'DATE' if 'DATE' in df.columns else next((c for c in df.columns if c.lower() == 'date'), None)
if date_col is None:
    raise ValueError("No DATE column found.")

# ---- Ensure datetime (naive) for comparisons
df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.tz_localize(None)

# ---- Determine "today" and build candidate close dates around it
today = (pd.to_datetime(OVERRIDE_TODAY) if OVERRIDE_TODAY else pd.Timestamp.today()).normalize()
years = [today.year - 1, today.year, today.year + 1]  # handles January edge cases cleanly

candidates = pd.to_datetime([f"{y:04d}-{m:02d}-{d:02d}" for y in years for (m, d) in ACCOUNTING_CLOSE_MD])
last_close = candidates[candidates <= today].max()  # most recent accounting period close (<= today)

# ---- Filter: dates strictly greater than the most recent accounting close
filtered_data = df[df[date_col] > last_close].copy()

# (Optional) expose the anchor date you filtered from
print("Most recent accounting close:", last_close.date())
filtered_data