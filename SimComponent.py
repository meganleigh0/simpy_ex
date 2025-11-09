import pandas as pd

# --- 1) SET YOUR DATES HERE ---
# If you only care about "up to this date", leave START_DATE=None and set END_DATE.
START_DATE = None                     # e.g., "2024-06-01" or None
END_DATE   = "2024-10-15"             # <-- cutoff/status date (inclusive)

# --- 2) LOAD THE WEEKLY EXTRACT SHEET ---
# If you've already loaded this elsewhere, comment these two lines and reuse your df.
xf = pd.ExcelFile("data/Cobra-XM30.xlsx")
df = xf.parse("tbl_Weekly Extract")

# --- 3) CLEAN/CAST ---
df = df.copy()
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df["COST-SET"] = df["COST-SET"].astype(str).str.strip().str.upper()
df["# HOURS"] = pd.to_numeric(df["# HOURS"], errors="coerce").fillna(0)

# Optional: keep only the "HOURS" plug if that matters in your workbook
# df = df[df["PLUG"].astype(str).str.upper().eq("HOURS")]

# --- 4) DATE FILTER (inclusive) ---
end_ts = pd.to_datetime(END_DATE) if END_DATE is not None else df["DATE"].max()
if START_DATE is not None:
    start_ts = pd.to_datetime(START_DATE)
    mask = (df["DATE"] >= start_ts) & (df["DATE"] <= end_ts)
else:
    mask = (df["DATE"] <= end_ts)

df_filt = df.loc[mask]

# --- 5) GROUP/PIVOT (CHG# x COST-SET) ---
grouped = (
    df_filt.pivot_table(
        index="CHG#",
        columns="COST-SET",
        values="# HOURS",
        aggfunc="sum",
        fill_value=0,
    )
    .sort_index()
)

# Ensure standard column order and presence
desired_cols = ["ACWP", "BCWP", "BCWS", "ETC"]
for c in desired_cols:
    if c not in grouped.columns:
        grouped[c] = 0
grouped = grouped.reindex(columns=desired_cols)

# --- 6) GRAND TOTAL ROW (optional) ---
grouped.loc["TOTAL"] = grouped.sum(numeric_only=True)

# Display result
grouped