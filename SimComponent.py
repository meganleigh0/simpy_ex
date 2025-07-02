import pandas as pd

# ------------------------------------------------------------------
# 0)  House-keeping
# ------------------------------------------------------------------
# df already has PART_NUMBER | Make/Buy | Date
df['Date'] = pd.to_datetime(df['Date'])

# normalise the status text you actually use in your files
df['Make/Buy'] = (df['Make/Buy']
                    .str.strip().str.lower()
                    .map({'make': 'make', 'buy': 'buy'}))

# one record per part per day (keep the last snapshot if duplicates)
df = (df.sort_values(['PART_NUMBER', 'Date'])
        .drop_duplicates(subset=['PART_NUMBER', 'Date'], keep='last')
        .reset_index(drop=True))

# ------------------------------------------------------------------
# 1)  Detect make ↔ buy flips
# ------------------------------------------------------------------
prev_status = df.groupby('PART_NUMBER')['Make/Buy'].shift()     # previous status per part
df['changed'] = df['Make/Buy'].ne(prev_status)                  # True where it flipped
df['changed'] = df['changed'].fillna(False)                     # first record per part → False

# ------------------------------------------------------------------
# 2)  Summarise
# ------------------------------------------------------------------
switch_counts = (
    df[df['changed']]
      .groupby('PART_NUMBER', as_index=False)
      .size()
      .rename(columns={'size': 'switch_count'})
      .sort_values('switch_count', ascending=False)
)

print("Top parts by # of Make↔Buy flips")
print(switch_counts.head(10))

# If you need the full flip-event log for auditing:
# flip_log = df[df['changed']].sort_values(['PART_NUMBER', 'Date'])