import pandas as pd
import re

# Sample DataFrame
df = pd.DataFrame({
    'PartNumber': ['ABC123-20230409', 'XYZ-01012023-PQ', '456DEF', '789GHI-12-31-2022']
})

# Function to remove date-like patterns
def remove_dates(part):
    # Common date patterns: YYYYMMDD, MMDDYYYY, DDMMYYYY, YYYY-MM-DD, MM-DD-YYYY, etc.
    date_patterns = [
        r'\b\d{8}\b',                   # 20230409
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', # 12/31/2022 or 12-31-2022
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b', # 2022/12/31 or 2022-12-31
        r'\b\d{2}\d{2}\d{4}\b',         # 12312022 or 31122022
    ]
    for pattern in date_patterns:
        part = re.sub(pattern, '', part)
    return part.strip('-_')

# Apply the function
df['CleanPartNumber'] = df['PartNumber'].apply(remove_dates)

print(df)