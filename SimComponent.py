import pandas as pd
import plotly.express as px

def analyze_bom_data(df_ebom, df_mbom_tc, df_mbom_oracle):
    """
    Analyze EBOM vs. MBOM data from different sources (EBOM, TeamCenter, Oracle).
    
    Parameters
    ----------
    df_ebom : pd.DataFrame
        DataFrame containing EBOM data with at least a 'PART_NUMBER' column.
    df_mbom_tc : pd.DataFrame
        DataFrame containing TeamCenter MBOM data with at least a 'PART_NUMBER' column.
    df_mbom_oracle : pd.DataFrame
        DataFrame containing Oracle MBOM data with at least a 'PART_NUMBER' column.

    Returns
    -------
    combined_df : pd.DataFrame
        A DataFrame with all unique part numbers (from all sources) and boolean columns
        indicating presence in EBOM, TeamCenter MBOM, and Oracle MBOM. Optionally also
        has columns for BASE_PART and SUFFIX (if underscore format is used).
    summary : dict
        Dictionary of key metrics (unique counts in each source, intersection sizes, etc.).
    fig : plotly.graph_objects.Figure or plotly.express.Figure
        A Plotly bar chart illustrating selected summary metrics.
    """

    # -----------------------------------------------------------------------------
    # 1. Gather Unique Part Numbers from Each Source
    # -----------------------------------------------------------------------------
    ebom_parts = set(df_ebom["PART_NUMBER"].unique())
    tc_parts = set(df_mbom_tc["PART_NUMBER"].unique())
    oracle_parts = set(df_mbom_oracle["PART_NUMBER"].unique())

    # -----------------------------------------------------------------------------
    # 2. Create Combined DataFrame of All Unique Parts
    # -----------------------------------------------------------------------------
    all_parts = ebom_parts.union(tc_parts).union(oracle_parts)
    combined_df = pd.DataFrame({"PART_NUMBER": list(all_parts)})

    # Add columns for presence in each source
    combined_df["IN_EBOM"] = combined_df["PART_NUMBER"].isin(ebom_parts)
    combined_df["IN_MBom_TC"] = combined_df["PART_NUMBER"].isin(tc_parts)
    combined_df["IN_MBom_Oracle"] = combined_df["PART_NUMBER"].isin(oracle_parts)

    # -----------------------------------------------------------------------------
    # 3. (Optional) Parse Base & Suffix
    # -----------------------------------------------------------------------------
    def split_part_suffix(part):
        if "_" in part:
            base, suffix = part.rsplit("_", 1)
            return base, suffix
        return part, None

    combined_df["BASE_PART"], combined_df["SUFFIX"] = zip(
        *combined_df["PART_NUMBER"].apply(split_part_suffix)
    )

    # -----------------------------------------------------------------------------
    # 4. Compute Summary Metrics
    # -----------------------------------------------------------------------------
    # Individual counts
    num_ebom = len(ebom_parts)
    num_mbom_tc = len(tc_parts)
    num_mbom_oracle = len(oracle_parts)
    num_all_unique = len(all_parts)

    # Intersections
    intersect_ebom_tc = ebom_parts.intersection(tc_parts)
    intersect_ebom_oracle = ebom_parts.intersection(oracle_parts)
    intersect_tc_oracle = tc_parts.intersection(oracle_parts)
    intersect_all_three = ebom_parts.intersection(tc_parts, oracle_parts)

    summary = {
        "num_ebom": num_ebom,
        "num_mbom_tc": num_mbom_tc,
        "num_mbom_oracle": num_mbom_oracle,
        "num_all_unique": num_all_unique,
        "intersect_ebom_tc": len(intersect_ebom_tc),
        "intersect_ebom_oracle": len(intersect_ebom_oracle),
        "intersect_tc_oracle": len(intersect_tc_oracle),
        "intersect_all_three": len(intersect_all_three),
    }

    # -----------------------------------------------------------------------------
    # 5. Create a Visualization (Bar Chart of Unique Part Counts)
    # -----------------------------------------------------------------------------
    plot_data = pd.DataFrame({
        "Source": ["EBOM", "MBOM_TC", "MBOM_Oracle", "All_Combined"],
        "Count": [
            summary["num_ebom"],
            summary["num_mbom_tc"],
            summary["num_mbom_oracle"],
            summary["num_all_unique"]
        ]
    })

    fig = px.bar(
        plot_data,
        x="Source", 
        y="Count", 
        title="Unique Part Counts by Source",
        text="Count",
        color="Source"
    )
    fig.update_layout(xaxis_title="Source", yaxis_title="Count of Unique Parts")
    fig.update_traces(textposition='outside')

    return combined_df, summary, fig


# -----------------------------------------------------------------------------
# Example usage:
# -----------------------------------------------------------------------------

# combined_df, metrics_summary, figure = analyze_bom_data(
#     LATEST_RELEASE_TC,        # EBOM dataframe
#     LATEST_RELEASE_TC_MBOM,   # TeamCenter MBOM dataframe
#     oracle_df                 # Oracle MBOM dataframe
# )

# print(metrics_summary)
# figure.show()
# combined_df.head()
