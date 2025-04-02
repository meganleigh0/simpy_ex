import pandas as pd
import os
from pyspark.sql import SparkSession

# -------------------------------
# 1. Configuration for London BOMs
# -------------------------------
london_config = {
    "lt01": ["03-25-2025", "04-01-2025"],
    "lt02": ["03-28-2025"]
}

# -------------------------------
# 2. Comparison + metric functions
# -------------------------------
def compare_single_mbom(df_ebom, df_mbom):
    group_df_ebom = pd.DataFrame(df_ebom.groupby("PART_NUMBER").agg({"Quantity": "sum", "Make or Buy": "first"})).reset_index()
    group_df_mbom = pd.DataFrame(df_mbom.groupby("PART_NUMBER").agg({"Quantity": "sum", "Make or Buy": "first"}))

    combined_df = group_df_ebom.merge(group_df_mbom, on="PART_NUMBER", how="left", suffixes=("_ebom", "_mbom"))
    combined_df["Match_EBOM_MBOM"] = combined_df["Quantity_ebom"] == combined_df["Quantity"]
    return combined_df

def calculate_london_metrics(combined_df, snapshot_date, variant_id):
    records = []
    for mob in ['Make', 'Buy']:
        ebom_parts = combined_df[combined_df['Make or Buy_ebom'] == mob]
        total_parts = ebom_parts['PART_NUMBER'].nunique()
        matched_parts = ebom_parts[ebom_parts['Match_EBOM_MBOM'] == True]['PART_NUMBER'].nunique()
        qty_mismatch_parts = ebom_parts[
            (ebom_parts['Quantity'].notnull()) & (ebom_parts['Match_EBOM_MBOM'] == False)
        ]['PART_NUMBER'].nunique()
        missing_parts = ebom_parts[ebom_parts['Quantity'].isnull()]['PART_NUMBER'].nunique()
        percent_matched = (matched_parts / total_parts * 100) if total_parts > 0 else 0

        records.append({
            'snapshot_date': snapshot_date,
            'variant_id': variant_id,
            'source': 'London_MBOM',
            'make_or_buy': mob,
            'total_parts': total_parts,
            'matched_parts': matched_parts,
            'quantity_mismatches': qty_mismatch_parts,
            'missing_parts': missing_parts,
            'percent_matched': percent_matched
        })

    return pd.DataFrame(records)

# -------------------------------
# 3. Process all London programs
# -------------------------------
for program, dates in london_config.items():
    all_snapshots = []

    for snapshot_date in dates:
        variant_id = f"{program}_{snapshot_date}"
        try:
            # File paths
            base_path = f"/Volumes/poc/default/london_boms/{program}"
            ebom_path = f"{base_path}/ebom_{snapshot_date}.csv"
            mbom_path = f"{base_path}/mbom_{snapshot_date}.csv"

            # Load CSVs
            ebom_df = pd.read_csv(ebom_path)
            ebom_df.rename(columns={"Make/Buy": "Make or Buy", "Qty": "Quantity"}, inplace=True)
            mbom_df = pd.read_csv(mbom_path)

            # Compare and calculate
            combined_df = compare_single_mbom(ebom_df, mbom_df)
            snapshot_df = calculate_london_metrics(combined_df, snapshot_date, variant_id)
            all_snapshots.append(snapshot_df)

        except Exception as e:
            print(f"Error processing {variant_id}: {e}")
            continue

    if all_snapshots:
        full_df = pd.concat(all_snapshots, ignore_index=True)
        spark_df = spark.createDataFrame(full_df)

        # Save to gold volume
        gold_path = f"/Volumes/poc/default/gold_bom_snapshot_london/{program}"
        spark_df.write.format("delta").mode("append").save(gold_path)

        # Register SQL table
        spark.sql(f"""
            CREATE OR REPLACE TABLE london_{program}_bom_completion_snapshot
            AS SELECT * FROM delta.`{gold_path}`
        """)

        print(f"Saved to gold + SQL table: london_{program}_bom_completion_snapshot")

print("All London BOMs processed.")