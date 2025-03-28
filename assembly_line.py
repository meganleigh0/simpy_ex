import pandas as pd
import os
from datetime import datetime

def calculate_and_save_mbom_completion_snapshot(ebom_df, mbom_teamcenter_df, mbom_oracle_df, snapshot_log_path):
    snapshot_date = pd.Timestamp.now().normalize()

    def compute_percentages(mbom_df, source_label):
        mbom_makes = mbom_df[mbom_df['Make or Buy'] == 'MAKE']
        mbom_buys = mbom_df[mbom_df['Make or Buy'] == 'BUY']
        mbom_make_count = mbom_makes.shape[0]
        mbom_buy_count = mbom_buys.shape[0]

        ebom_makes = ebom_df[ebom_df['Make or Buy'] == 'Make']
        ebom_buys = ebom_df[ebom_df['Make or Buy'] == 'Buy']
        ebom_make_count = ebom_makes.shape[0]
        ebom_buy_count = ebom_buys.shape[0]

        make_percent = (mbom_make_count / ebom_make_count) * 100 if ebom_make_count else 0
        buy_percent = (mbom_buy_count / ebom_buy_count) * 100 if ebom_buy_count else 0

        return [
            {"snapshot_date": snapshot_date, "source": source_label, "make_or_buy": "Make", "percent_matched": make_percent},
            {"snapshot_date": snapshot_date, "source": source_label, "make_or_buy": "Buy", "percent_matched": buy_percent}
        ]

    # Calculate for both sources
    snapshot_records = []
    snapshot_records.extend(compute_percentages(mbom_oracle_df, "Oracle"))
    snapshot_records.extend(compute_percentages(mbom_teamcenter_df, "TeamCenter"))

    # Convert to DataFrame
    snapshot_df = pd.DataFrame(snapshot_records)

    # Load or create log
    if os.path.exists(snapshot_log_path):
        log_df = pd.read_csv(snapshot_log_path, parse_dates=["snapshot_date"])
        combined_df = pd.concat([log_df, snapshot_df], ignore_index=True)
    else:
        combined_df = snapshot_df

    # Save back to CSV
    combined_df.to_csv(snapshot_log_path, index=False)
    return combined_df