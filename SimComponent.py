# Create mapping for each target column
mapping_instructions = [
    {
        'dept': 'PSGA - CSSC G & A',
        'rate_type': 'ALLOWABLE G & A RATE',
        'target_column': 'G&A CSSC'
    },
    {
        'dept': 'DVGA - DIVISION GENERAL & ADM',
        'rate_type': 'ALLOWABLE G & A RATE',
        'target_column': 'G&A GDLS'
    }
]

# Iterate over mapping instructions
for instruction in mapping_instructions:
    dept = instruction['dept']
    rate_type = instruction['rate_type']
    target_col = instruction['target_column']
    
    # Filter the row from other_rates that matches dept + rate_type
    rate_row = other_rates[
        (other_rates['Unnamed: 0'] == dept) & 
        (other_rates['Unnamed: 2'] == rate_type)
    ]
    
    if rate_row.empty:
        print(f"Warning: No matching rate found for {dept} - {rate_type}")
        continue

    # Extract year-rate pairs from CY2022 to CY2025
    for year in range(2022, 2026):
        column_name = f'CY{year}'
        rate_value = float(rate_row[column_name].values[0])

        # Update matching rows in BURDEN_RATE where Description and Effective Date match
        mask = (
            BURDEN_RATE['Description'].str.contains(target_col.split()[-1], na=False) & 
            (BURDEN_RATE['Effective Date'].astype(str) == str(year))
        )
        BURDEN_RATE.loc[mask, target_col] = round(rate_value, 5)

# Forward-fill to handle years beyond 2025
for col in ['G&A CSSC', 'G&A GDLS']:
    BURDEN_RATE[col] = BURDEN_RATE.groupby('Description')[col].ffill()

# Show updated BURDEN_RATE
BURDEN_RATE[['Description', 'Effective Date', 'G&A CSSC', 'G&A GDLS']].dropna(how='all', subset=['G&A CSSC', 'G&A GDLS'])