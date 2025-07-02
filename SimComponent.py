import pandas as pd

# ────────────────────────────────────────────────────────────────
# 0️⃣  ASSUMPTIONS
# ────────────────────────────────────────────────────────────────
# • `df` already exists in memory and contains at least
#     PART_NUMBER | Make/Buy | Date | Description | Levels
# • You do NOT want to treat a missing Make/Buy value as a flip
#   (i.e. we ignore transitions from NaN → make/buy or vice-versa)
# • Dates are the actual snapshot dates you care about.

# ────────────────────────────────────────────────────────────────
# 1️⃣  CLEAN & NORMALISE
# ────────────────────────────────────────────────────────────────
df_clean = (
    df.copy()
      .assign(Date=pd.to_datetime(df['Date']))                       # guarantee datetime
      .assign(Make_Buy=(df['Make/Buy']                               # standardise text
                        .astype(str).str.strip().str.lower()
                        .where(lambda s: s.isin(['make', 'buy']))))  # keep only make/buy
      .drop(columns=['Make/Buy'])
      .rename(columns={'Make_Buy': 'Make/Buy'})
      .sort_values(['PART_NUMBER', 'Date'])
      .drop_duplicates(subset=['PART_NUMBER', 'Date'], keep='last')  # one snapshot per day
      .reset_index(drop=True)
)

# ────────────────────────────────────────────────────────────────
# 2️⃣  DETECT TRUE FLIPS
# ────────────────────────────────────────────────────────────────
df_clean['previous_status'] = df_clean.groupby('PART_NUMBER')['Make/Buy'].shift()

mask_flip = (
    df_clean['Make/Buy'].notna() &
    df_clean['previous_status'].notna() &
    df_clean['Make/Buy'].ne(df_clean['previous_status'])
)

flip_log = (df_clean[mask_flip]
            .loc[:, ['PART_NUMBER', 'Description', 'Levels',
                     'Date', 'previous_status', 'Make/Buy']]
            .rename(columns={'Make/Buy': 'new_status'})
            .sort_values(['Date', 'PART_NUMBER'])
            .reset_index(drop=True))

# ────────────────────────────────────────────────────────────────
# 3️⃣  SNAPSHOT-BY-SNAPSHOT SUMMARY
# ────────────────────────────────────────────────────────────────
snapshot_summary = (flip_log.groupby('Date')['PART_NUMBER']
                             .nunique()
                             .rename('num_parts_changed')
                             .reset_index()
                             .sort_values('Date'))

# ────────────────────────────────────────────────────────────────
# 4️⃣  QUICK LOOK
# ────────────────────────────────────────────────────────────────
print("\n--- First 15 flip events ---")
display(flip_log.head(15))

print("\n--- Parts that changed on each snapshot date ---")
display(snapshot_summary)

# Optional: save for dashboards / further analysis
flip_log.to_csv('make_buy_flip_log.csv', index=False)
snapshot_summary.to_csv('make_buy_flip_snapshot_summary.csv', index=False)