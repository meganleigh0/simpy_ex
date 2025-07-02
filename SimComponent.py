import pandas as pd
import plotly.express as px

# ────────────────────────────────────────────────────────────────
# 1)  CLEAN & NORMALISE  (works even if you already ran it earlier)
# ────────────────────────────────────────────────────────────────
df_clean = (
    df.copy()
      .assign(Date=pd.to_datetime(df['Date']))                       # make sure Date is datetime
      .assign(Make_Buy=(df['Make/Buy']                               # standardise status text
                        .astype(str).str.strip().str.lower()
                        .where(lambda s: s.isin(['make', 'buy']))))  # keep only make/buy
      .drop(columns=['Make/Buy'])
      .rename(columns={'Make_Buy': 'Make/Buy'})
      .sort_values(['PART_NUMBER', 'Date'])
      .drop_duplicates(subset=['PART_NUMBER', 'Date'], keep='last')  # one snapshot per day
      .reset_index(drop=True)
)

# ────────────────────────────────────────────────────────────────
# 2)  DETECT TRUE FLIPS
#     (ignore NaN → make/buy or make/buy → NaN transitions)
# ────────────────────────────────────────────────────────────────
df_clean['previous_status'] = df_clean.groupby('PART_NUMBER')['Make/Buy'].shift()

flip_mask = (
    df_clean['Make/Buy'].notna() &
    df_clean['previous_status'].notna() &
    df_clean['Make/Buy'].ne(df_clean['previous_status'])
)

flip_log = (df_clean[flip_mask]
            .loc[:, ['PART_NUMBER', 'Description', 'Levels',
                     'Date', 'previous_status', 'Make/Buy']]
            .rename(columns={'Make/Buy': 'new_status'})
            .sort_values(['Date', 'PART_NUMBER'])
            .reset_index(drop=True))

# ────────────────────────────────────────────────────────────────
# 3)  SNAPSHOT-LEVEL SUMMARY  (parts that flipped on each date)
# ────────────────────────────────────────────────────────────────
snapshot_summary = (flip_log.groupby('Date')['PART_NUMBER']
                             .nunique()
                             .rename('num_parts_changed')
                             .reset_index()
                             .sort_values('Date'))

# ────────────────────────────────────────────────────────────────
# 4)  PLOTLY VISUALISATION
# ────────────────────────────────────────────────────────────────
fig = px.line(snapshot_summary,
              x='Date',
              y='num_parts_changed',
              markers=True,
              title='Distinct Parts Switching Make/Buy Status per Snapshot')

fig.update_layout(
    xaxis_title='Snapshot Date',
    yaxis_title='Number of Parts Switched',
    hovermode='x unified'
)

fig.show()

# (Optional) save the detail & summary for dashboards
# flip_log.to_csv('make_buy_flip_log.csv', index=False)
# snapshot_summary.to_csv('make_buy_flip_snapshot_summary.csv', index=False)