import pandas as pd

# === Load Data ===
file_path = "data/Cobra-XM30.xlsx"
sheet = "tbl_Weekly Extract"

df = pd.read_excel(file_path, sheet_name=sheet)
df.columns = df.columns.str.strip()

# === Normalize column names for consistency ===
df.rename(columns=lambda x: x.upper().replace(" ", "_"), inplace=True)

# === Filter relevant columns only ===
cols_needed = ["CHG#", "COST-SET", "HOURS"]
df = df[[c for c in cols_needed if c in df.columns]]

# === Coerce HOURS to numeric ===
df["HOURS"] = pd.to_numeric(df["HOURS"], errors="coerce").fillna(0)

# === Group and pivot exactly like Excel ===
grouped = (
    df.groupby(["CHG#", "COST-SET"])["HOURS"]
      .sum()
      .unstack(fill_value=0)
      .reindex(columns=["ACWP", "BCWP", "BCWS", "ETC"], fill_value=0)
)

# === Add Grand Total Row (matching Excel) ===
grouped.loc["Grand Total"] = grouped.sum(numeric_only=True)

# === Optional: Round to match Excel’s precision ===
grouped = grouped.round(4)

# === Show result ===
print(grouped.tail(10))