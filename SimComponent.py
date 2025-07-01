for path, df in df_dict.items():
    # Extract date from the file name using regex
    match = re.search(r'(\d{2}-\d{2}-\d{4})\.xlsx$', path)
    if match:
        date_str = match.group(1)
        df = df.copy()
        df['Date'] = pd.to_datetime(date_str, format='%m-%d-%Y')
        all_dfs.append(df)
    else:
        print(f"Warning: No valid date found in path {path}")

# Concatenate all dataframes into one
combined_df = pd.concat(all_dfs, ignore_index=True)
