import glob, os
import pandas as pd
import ace_tools as tools

# ────────────────────────────────────────────────────────────────
# 0)  CONFIG – adjust if your folders / patterns differ
# ────────────────────────────────────────────────────────────────
# Example patterns (adjust to match your reality)
PATTERNS = [
    "data/bronze_boms/mbom_oracle_*.xlsx",
    "data/bronze_boms/mbom_tc_*.xlsm"
]

# Columns we need to keep
KEEP_COLS = [
    "PART_NUMBER",          # part id
    "Make/Buy",             # status
    "Date",                 # snapshot date (from filename)
    "Description",          # <- optional
    "Levels"                # <- optional
]

# ────────────────────────────────────────────────────────────────
# 1)  LOAD & CONCATENATE
# ────────────────────────────────────────────────────────────────
dfs = []
for pattern in PATTERNS:
    for path in glob.glob(pattern):
        snap_date = pd.to_datetime(os.path.basename(path).split("_")[-1].split(".")[0])
        tmp = pd.read_excel(path, dtype=str, engine="openpyxl", usecols=lambda c: c in KEEP_COLS)
        tmp["Date"] = snap_date
        dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)

# ────────────────────────────────────────────────────────────────
# 2)  CLEAN & NORMALISE
# ────────────────────────────────────────────────────────────────
df['Date'] = pd.to_datetime(df['Date'])
df['Make/Buy'] = (df['Make/Buy']
                    .astype(str).str.strip().str.lower()
                    .where(lambda s: s.isin(['make', 'buy'])))

df = (df.sort_values(['PART_NUMBER', 'Date'])
        .drop_duplicates(subset=['PART_NUMBER', 'Date'], keep='last')
        .reset_index(drop=True))

# ────────────────────────────────────────────────────────────────
# 3)  DETECT FLIPS
# ────────────────────────────────────────────────────────────────
df['previous_status'] = df.groupby('PART_NUMBER')['Make/Buy'].shift()
mask_flip = (df['Make/Buy'].notna() & df['previous_status'].notna()
             & df['Make/Buy'].ne(df['previous_status']))

flip_log = (df[mask_flip]
            .loc[:, ['PART_NUMBER', 'Description', 'Levels', 'Date',
                     'previous_status', 'Make/Buy']]
            .rename(columns={'Make/Buy': 'new_status'})
            .sort_values(['PART_NUMBER', 'Date'])
            .reset_index(drop=True))

# ────────────────────────────────────────────────────────────────
# 4)  WEEKLY SUMMARY
# ────────────────────────────────────────────────────────────────
flip_log['week_start'] = flip_log['Date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_summary = (flip_log.groupby('week_start')['PART_NUMBER']
                          .nunique()
                          .rename('num_parts_changed')
                          .reset_index()
                          .sort_values('week_start'))

# ────────────────────────────────────────────────────────────────
# 5)  SAVE + SHOW
# ────────────────────────────────────────────────────────────────
csv_path = "make_buy_flip_log.csv"
flip_log.to_csv(csv_path, index=False)

tools.display_dataframe_to_user("Make-Buy Flip Log (detailed)", flip_log)
tools.display_dataframe_to_user("Parts changed per week", weekly_summary)

print(f"\nDetailed log saved to: {csv_path}")