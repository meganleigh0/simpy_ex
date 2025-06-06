# ── robust version: works even if some column labels are floats or NaN ────────────
def strip_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any column whose *string* representation contains 'Unnamed'
    and make sure all remaining column names are strings.
    """
    mask = pd.Series(df.columns).apply(lambda c: "Unnamed" not in str(c))
    cleaned = df.loc[:, mask.values]
    cleaned.columns = cleaned.columns.map(str)         # force str type for consistency
    return cleaned.rename_axis(None, axis=1)