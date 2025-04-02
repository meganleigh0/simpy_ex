import pandas as pd
from pyspark.sql import SparkSession

# -------------------------------
# 1. Define your London BOM config
# -------------------------------
london_config = {
    "lt01": ["03-25-2025", "04-01-2025"],
    "lt02": ["03-28-2025"]
}

# -------------------------------
# 2. Functions
# -------------------------------

def clean_and_standardize(df):
    df.columns = [col.strip() for col in df.columns]
    if "Make/Buy" in df.columns:
        df.rename(columns={"Make/Buy": "Make or Buy"}, inplace=True)
    return df

def compare_ebom_mbom(df_ebom, df_mbom):
    ebom_grouped = df_ebom.groupby("PART_NUMBER").agg({
        "Quantity": "sum", 
        "Make or Buy": "first"
    }).reset_index()
    
    mbom_grouped = df_mbom.groupby("PART_NUMBER").agg({
        "Quantity": "sum"
    }).reset_index()

    combined = ebom_grouped.merge(mbom_grouped, on="PART_NUMBER", how="left", suffixes=("_ebom", "_mbom"))
    combined["Match_EBOM_MBOM"] = combined["Quantity_ebom"] == combined["Quantity_mbom"]
    return combined

def calculate_snapshot_metrics(combined_df, snapshot_date, variant_id):
    records = []
    for mob in ["Make", "Buy"]:
        subset = combined_df[combined_df["Make or Buy"] == mob]
        total = subset["PART_NUMBER"].nunique()
        matched = subset[subset["Match_EBOM_MBOM"] == True]["PART_NUMBER"].nunique()
        qty_mismatches = subset[
            (subset["Quantity_mbom"].notnull()) & (subset["Match_EBOM_MBOM"] == False)
        ]["PART_NUMBER"].nunique()
        missing = subset[subset["Quantity_mbom"].isnull()]["PART_NUMBER"].nunique()
        percent = (matched / total * 100) if total > 0 else 0

        records.append({
            "snapshot_date": snapshot_date,
            "variant_id": variant_id,
            "source": "London_MBOM",
            "make_or_buy": mob,
            "total_parts": total,
            "matched_parts": matched,
            "quantity_mismatches": qty_mismatches,
            "missing_parts": missing,
            "percent_matched": percent
        })
    return pd.DataFrame(records)

# -------------------------------
# 3. Process and Save
# -------------------------------

for program, dates in london_config.items():
    all_snapshots = []

    for snapshot_date in dates:
        variant_id = f"{program}_{snapshot_date}"
        print(f"Processing {variant_id}...")

        try:
            base_path = f"/Volumes/poc/default/london_boms/{program}"
            ebom_path = f"{base_path}/ebom_{snapshot_date}.csv"
            mbom_path = f"{base_path}/mbom_{snapshot_date}.csv"

            # Load + clean
            ebom_df = clean_and_standardize(pd.read_csv(ebom_path))
            mbom_df = clean_and_standardize(pd.read_csv(mbom_path))

            # Validate columns
            required = {"PART_NUMBER", "Quantity", "Make or Buy"}
            if not required.issubset(set(ebom_df.columns)):
                raise ValueError(f"EBOM columns invalid: {ebom_df.columns}")
            if "PART_NUMBER" not in mbom_df.columns or "Quantity" not in mbom_df.columns:
                raise ValueError(f"MBOM columns invalid: {mbom_df.columns}")

            # Compare and calculate
            combined_df = compare_ebom_mbom(ebom_df, mbom_df)
            snapshot_df = calculate_snapshot_metrics(combined_df, snapshot_date, variant_id)
            all_snapshots.append(snapshot_df)

        except Exception as e:
            print(f"Error processing {variant_id}: {e}")
            continue

    # Save to gold + register SQL table
    if all_snapshots:
        full_df = pd.concat(all_snapshots, ignore_index=True)
        spark_df = spark.createDataFrame(full_df)

        gold_path = f"/Volumes/poc/default/gold_bom_snapshot_london/{program}"
        spark_df.write.format("delta").mode("append").save(gold_path)

        spark.sql(f"""
            CREATE OR REPLACE TABLE london_{program}_bom_completion_snapshot
            AS SELECT * FROM delta.`{gold_path}`
        """)

        print(f"Saved to: london_{program}_bom_completion_snapshot")

print("✅ All London BOMs processed successfully.")