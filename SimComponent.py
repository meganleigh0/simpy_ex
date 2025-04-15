import pandas as pd

# Load the IBOM data (replace 'ibom.csv' with your file path)
df = pd.read_csv('ibom.csv')

# Preview the IBOM
print("Raw IBOM:")
print(df.head())

# Basic summary
print("\n--- IBOM Summary ---")
print("Total parts listed:", len(df))
print("Unique part numbers:", df['part number'].nunique())

# Group by level
print("\n--- Parts by Hierarchical Level ---")
level_counts = df['level'].value_counts().sort_index()
print(level_counts)

# Total quantity by part number (ignores hierarchy)
print("\n--- Total Quantity by Part Number ---")
total_quantity = df.groupby('part number')['quantity'].sum().reset_index()
total_quantity = total_quantity.sort_values(by='quantity', ascending=False)
print(total_quantity)

# Optional: export flattened BOM summary
total_quantity.to_csv('flattened_bom_summary.csv', index=False)
print("\nFlattened BOM summary exported to 'flattened_bom_summary.csv'")