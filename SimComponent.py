import pandas as pd

# ------------------------------------------------------------------
# 0)  ⇨  make sure df exists in memory before you run this cell
# ------------------------------------------------------------------
# df  should hold columns:  PART_NUMBER | Make/Buy | Date
# If you still have your dictionary of DataFrames, concatenate those
# into `df` first, then run this cell.

# ------------------------------------------------------------------
# 1)  Clean & dedupe
# ------------------------------------------------------------------
df_clean = (
    df.copy()
      .assign(Date=pd.to_datetime(df['Date']))                # ensure datetime
      .assign(Make_Buy=(df['Make/Buy']                        # normalise text
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({'make': 'make', 'buy': 'buy'})))
      .drop(columns=['Make/Buy'])
      .rename(columns={'Make_Buy': 'Make/Buy'})
      .sort_values(['PART_NUMBER', 'Date'])
      .drop_duplicates(subset=['PART_NUMBER', 'Date'], keep='last')
      .reset_index(drop=True)
)

# ------------------------------------------------------------------
# 2)  Detect make ↔ buy flips
# ------------------------------------------------------------------
df_clean['previous_status'] = df_clean.groupby('PART_NUMBER')['Make/Buy'].shift()
flip_log = df_clean[df_clean['Make/Buy'] != df_clean['previous_status']].copy()

flip_log['previous_status'] = flip_log['previous_status'].fillna('-')  # first record
flip_log = (flip_log[['PART_NUMBER', 'Date', 'previous_status', 'Make/Buy']]
            .rename(columns={'Make/Buy': 'new_status'})
            .sort_values(['PART_NUMBER', 'Date'])
            .reset_index(drop=True))

# ------------------------------------------------------------------
# 3)  Summarise
# ------------------------------------------------------------------
switch_counts = (
    flip_log.groupby('PART_NUMBER', as_index=False)
            .size()
            .rename(columns={'size': 'flip_count'})
            .sort_values('flip_count', ascending=False)
)

# ------------------------------------------------------------------
# 4)  Persist + quick peek
# ------------------------------------------------------------------
flip_log.to_csv('make_buy_flip_log.csv', index=False)

print("\n=== First 20 flip events ===")
print(flip_log.head(20).to_string(index=False))

print("\n=== Parts with the most flips ===")
print(switch_counts.head(20).to_string(index=False))

# ------------------------------------------------------------------
# 5)  (Optional) – status over time heat-map style
# ------------------------------------------------------------------
# status_matrix = df_clean.pivot(index='PART_NUMBER', columns='Date', values='Make/Buy')
# status_matrix.head()  # gives you a wide view of make/buy status by snapshot