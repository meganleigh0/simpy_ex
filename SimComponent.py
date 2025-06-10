import pandas as pd

# Load the Excel file but keep the top rows so we can access row 0
raw = pd.read_excel("your_file.xlsx", header=None)

# Extract the real header row (row 5 in 0-based indexing)
column_headers = raw.iloc[5].tolist()

# Replace the first column's header with the title from row 0, column 0
column_headers[0] = raw.iloc[0, 0]

# Now read the actual data starting from row 6 (index 6), and apply the fixed headers
df = pd.read_excel("your_file.xlsx", skiprows=6, header=None)
df.columns = column_headers

# Optional: clean up any unnamed or NaN columns
df = df.loc[:, ~df.columns.isna()]
df.columns = [str(col).strip() for col in df.columns]

print(df.head())