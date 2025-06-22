import numpy as np

# STEP 1: Map relevant cost elements to descriptions (Other_Rates row labels)
material_to_rate_key = {
    "PSGA": "CSSC G & A ALLOWABLE G&A",
    "DVGA": "DIVISION GENERAL & ADMIN",
    "ALLOWABLE SUPPORT RATE": "ALLOWABLE SUPPORT RATE",
    "ALLOWABLE CONTROL TEST RATE": "ALLOWABLE CONTROL TEST RATE",
    "DeptNA ALLOWABLE REORDER POINT": "DeptNA ALLOWABLE REORDER POINT",
    "PRLS": "GDLS PROCUREMENT ALLOW",
    "PRFT": "FREIGHT – GDLS & CSSC AL",
    "DeptNA ALLOWABLE MAJOR END-IT": "DeptNA ALLOWABLE MAJOR END-IT",
}

# STEP 2: Clean column names to just the year from 'CY2022', 'CY2023.1', etc.
clean_columns = {col: str(col).replace("CY", "").split(".")[0] for col in other_rates.columns if "CY" in str(col)}
other_rates_renamed = other_rates.rename(columns=clean_columns)

# STEP 3: Iterate and update BURDEN_RATE
for material, rate_key in material_to_rate_key.items():
    if rate_key not in other_rates_renamed.index:
        continue  # Skip if not in the Other Rates

    for year in range(2022, 2032):  # For CY2022 to CY2031
        value = other_rates_renamed.loc[rate_key].get(str(year), np.nan)
        
        if not pd.isna(value):
            # Find the row in BURDEN_RATE that matches burden pool and year
            mask = (BURDEN_RATE["Burden Pool"] == "CSSC") & (BURDEN_RATE["Date"] == year)
            BURDEN_RATE.loc[mask, material] = value