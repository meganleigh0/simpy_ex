import pandas as pd

# Load both Excel sheets
other_rates = pd.read_excel("RatesFile.xlsx", sheet_name="OTHER RATES", header=None)
burden_df = pd.read_excel("BurdenFile.xlsx", sheet_name="20220511FRP Burden")

# Step 1: Extract G&A rates from Other Rates sheet
gna_cssc_years = ['CY2022', 'CY2023', 'CY2024', 'CY2025']
gna_cssc_values = other_rates.loc[6, 8:12].values  # Row 7, columns I-L (0-indexed)
gna_gdls_values = other_rates.loc[7, 8:12].values  # Row 8, columns I-L

# Step 2: Create mapping for each year
years = [2022, 2023, 2024, 2025]
gna_cssc_map = dict(zip(years, gna_cssc_values))
gna_gdls_map = dict(zip(years, gna_gdls_values))

# Step 3: Apply mapping to burden_df based on Effective Date (column 'C')
burden_df['G&A CSSC'] = burden_df.apply(
    lambda row: gna_cssc_map.get(row['Effective Date'], row.get('G&A CSSC')),
    axis=1
)
burden_df['G&A GDLS'] = burden_df.apply(
    lambda row: gna_gdls_map.get(row['Effective Date'], row.get('G&A GDLS')),
    axis=1
)

# Step 4: Forward-fill any remaining missing G&A values by group
burden_df['G&A CSSC'] = burden_df['G&A CSSC'].ffill()
burden_df['G&A GDLS'] = burden_df['G&A GDLS'].ffill()

# Step 5: Save updated sheet
burden_df.to_excel("Updated_Burden_Rate_Import.xlsx", index=False)