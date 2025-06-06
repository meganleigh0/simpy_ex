# Convert both to Series first (already are, from .loc)
years = years_test_rate.squeeze()   # CY2022, CY2023, etc.
rates = allowable_control_test_rate.squeeze()  # numeric values

# Combine into a single DataFrame
combined_df = pd.DataFrame({
    'Year': years.values,
    'Rate': rates.values
})

# Optional: set Year as index
combined_df.set_index('Year', inplace=True)

print(combined_df)