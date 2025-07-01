import pandas as pd
import re

def load_all_mboms_with_metadata(mbom_dict):
    all_mboms = []

    for path, df in mbom_dict.items():
        # Match MM-DD-YYYY and extension
        match = re.search(r'(\d{2}-\d{2}-\d{4})\.(xlsx|xlsm)$', path)
        if match:
            date_str, ext = match.groups()
            df = df.copy()
            df['Date'] = pd.to_datetime(date_str, format='%m-%d-%Y')
            df['Source'] = 'Oracle' if ext == 'xlsx' else 'TeamCenter'
            all_mboms.append(df)
        else:
            print(f"Warning: No valid date found in path {path}")

    return pd.concat(all_mboms, ignore_index=True)

# Example usage:
# mbom_dict = { 'path/to/file_06-01-2025.xlsx': df1, 'path/to/file_06-01-2025.xlsm': df2, ... }
all_mboms_df = load_all_mboms_with_metadata(mbom_dict)