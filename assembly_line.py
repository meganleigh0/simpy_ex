import pandas as pd
import plotly.express as px

def analyze_bom_by_base(df_ebom, df_mbom_tc, df_mbom_oracle):
    """
    Analyze EBOM vs. MBOM by base part (ignoring suffix differences).

    Parameters
    ----------
    df_ebom : pd.DataFrame
        EBOM dataframe with at least "PART_NUMBER".
    df_mbom_tc : pd.DataFrame
        TeamCenter MBOM dataframe with at least "PART_NUMBER".
    df_mbom_oracle : pd.DataFrame
        Oracle MBOM dataframe with at least "PART_NUMBER".
    
    Returns
    -------
    combined_df : pd.DataFrame
        DataFrame of all unique base parts, with flags for presence
        in each source. Also has a list of suffixes per base part, if needed.
    summary : dict
        Dictionary of count metrics (unique base parts in each BOM, intersections, etc.).
    fig : plotly.Figure
        A bar chart showing the counts of unique base parts found in each BOM.
    """

    # -----------------------------
    # 1. Parse base parts
    # -----------------------------
    def get_base_part(part):
        # If there's an underscore, split on the last underscore
        if "_" in part:
            return part.rsplit("_", 1)[0]
        return part  # No underscore, entire string is the base
    
    # Create new columns for base parts in each BOM
    df_ebom["BASE_PART"] = df_ebom["PART_NUMBER"].apply(get_base_part)
    df_mbom_tc["BASE_PART"] = df_mbom_tc["PART_NUMBER"].apply(get_base_part)
    df_mbom_oracle["BASE_PART"] = df_mbom_oracle["PART_NUMBER"].apply(get_base_part)
    
    # We might also collect suffixes, but for membership we only care about the base:
    # If you need to see all suffixes per base, you can group by base part and collect them.

    # -----------------------------
    # 2. Get sets of base parts
    # -----------------------------
    ebom_base_parts = set(df_ebom["BASE_PART"].unique())
    tc_base_parts = set(df_mbom_tc["BASE_PART"].unique())
    oracle_base_parts = set(df_mbom_oracle["BASE_PART"].unique())

    # Combined set of all base parts
    all_base_parts = ebom_base_parts.union(tc_base_parts).union(oracle_base_parts)

    # -----------------------------
    # 3. Create combined DataFrame
    # -----------------------------
    combined_df = pd.DataFrame({"BASE_PART": list(all_base_parts)})

    # Flags for presence
    combined_df["IN_EBOM_BASE"] = combined_df["BASE_PART"].isin(ebom_base_parts)
    combined_df["IN_TC_BASE"] = combined_df["BASE_PART"].isin(tc_base_parts)
    combined_df["IN_ORACLE_BASE"] = combined_df["BASE_PART"].isin(oracle_base_parts)

    # Optionally collect all suffixes from each BOM. For example:
    def collect_suffixes(df, base_col="BASE_PART", part_col="PART_NUMBER"):
        # Group by base part, collect unique suffixes
        # If no underscore, suffix is set to None
        grouped = df.groupby(base_col)[part_col].apply(
            lambda s: sorted(set(p.rsplit("_", 1)[1] if "_" in p else None for p in s))
        )
        return grouped

    ebom_suffixes = collect_suffixes(df_ebom)
    tc_suffixes = collect_suffixes(df_mbom_tc)
    oracle_suffixes = collect_suffixes(df_mbom_oracle)

    # Merge these suffix sets into the combined_df
    combined_df = combined_df.merge(
        ebom_suffixes.rename("EBOM_SUFFIXES"), how="left", left_on="BASE_PART", right_index=True
    ).merge(
        tc_suffixes.rename("TC_SUFFIXES"), how="left", left_on="BASE_PART", right_index=True
    ).merge(
        oracle_suffixes.rename("ORACLE_SUFFIXES"), how="left", left_on="BASE_PART", right_index=True
    )

    # -----------------------------
    # 4. Compute summary metrics
    # -----------------------------
    num_ebom_base = len(ebom_base_parts)
    num_tc_base = len(tc_base_parts)
    num_oracle_base = len(oracle_base_parts)
    num_all_base = len(all_base_parts)

    intersect_ebom_tc = ebom_base_parts.intersection(tc_base_parts)
    intersect_ebom_oracle = ebom_base_parts.intersection(oracle_base_parts)
    intersect_tc_oracle = tc_base_parts.intersection(oracle_base_parts)
    intersect_all_three = ebom_base_parts.intersection(tc_base_parts, oracle_base_parts)

    summary = {
        "ebom_base_count": num_ebom_base,
        "tc_base_count": num_tc_base,
        "oracle_base_count": num_oracle_base,
        "all_base_count": num_all_base,
        "ebom_tc_intersect_base": len(intersect_ebom_tc),
        "ebom_oracle_intersect_base": len(intersect_ebom_oracle),
        "tc_oracle_intersect_base": len(intersect_tc_oracle),
        "all_three_intersect_base": len(intersect_all_three)
    }

    # -----------------------------
    # 5. Create a bar chart
    # -----------------------------
    df_plot = pd.DataFrame({
        "Source": ["EBOM Base", "TC Base", "Oracle Base", "All Base"],
        "Count": [
            num_ebom_base,
            num_tc_base,
            num_oracle_base,
            num_all_base
        ]
    })

    fig = px.bar(
        df_plot,
        x="Source",
        y="Count",
        color="Source",
        text="Count",
        title="Unique Base Part Counts by Source"
    )
    fig.update_layout(xaxis_title="Source", yaxis_title="Count of Unique Base Parts")
    fig.update_traces(textposition="outside")

    return combined_df, summary, fig


# ------------------------
# Example usage:
# ------------------------
# combined_df, base_summary, base_fig = analyze_bom_by_base(
#     LATEST_RELEASE_TC,        # EBOM dataframe
#     LATEST_RELEASE_TC_MBOM,   # TeamCenter MBOM dataframe
#     oracle_df
# )
# display(combined_df.head(10))        # or print(combined_df.head(10)) in a script
# print(base_summary)
# base_fig.show()
import dash
from dash import dcc, html, dash_table
import pandas as pd

def create_dashboard(df_combined, summary, fig):
    """
    Create a Dash dashboard that shows:
      1. A bar chart of base part counts by source.
      2. A quick data table of the combined DataFrame (with base part presence).
    
    Returns a Dash app (not yet run).
    """
    app = dash.Dash(__name__)

    # Prepare columns for data table (only show a few if your DF is huge)
    table_columns = [
        {"name": "BASE_PART", "id": "BASE_PART"},
        {"name": "IN_EBOM_BASE", "id": "IN_EBOM_BASE"},
        {"name": "IN_TC_BASE", "id": "IN_TC_BASE"},
        {"name": "IN_ORACLE_BASE", "id": "IN_ORACLE_BASE"},
        # Add suffix columns if desired:
        {"name": "EBOM_SUFFIXES", "id": "EBOM_SUFFIXES"},
        {"name": "TC_SUFFIXES", "id": "TC_SUFFIXES"},
        {"name": "ORACLE_SUFFIXES", "id": "ORACLE_SUFFIXES"},
    ]

    # Convert your df to dictionary for Dash datatable
    data_table = dash_table.DataTable(
        columns=table_columns,
        data=df_combined.to_dict("records"),
        page_size=10,  # display 10 rows per page
    )

    app.layout = html.Div([
        html.H1("BOM Comparison Dashboard"),
        dcc.Graph(figure=fig),
        html.H2("Combined BOM Table (Base-level)"),
        data_table,
        html.Br(),
        html.H2("Summary Metrics"),
        html.Ul([
            html.Li(f"EBOM base count: {summary['ebom_base_count']}"),
            html.Li(f"TeamCenter base count: {summary['tc_base_count']}"),
            html.Li(f"Oracle base count: {summary['oracle_base_count']}"),
            html.Li(f"All unique base parts: {summary['all_base_count']}"),
            html.Li(f"EBOM-TC intersection: {summary['ebom_tc_intersect_base']}"),
            html.Li(f"EBOM-Oracle intersection: {summary['ebom_oracle_intersect_base']}"),
            html.Li(f"TC-Oracle intersection: {summary['tc_oracle_intersect_base']}"),
            html.Li(f"All three intersection: {summary['all_three_intersect_base']}"),
        ])
    ])

    return app

# ------------------------
# Example usage:
# ------------------------
# combined_df, summary, fig = analyze_bom_by_base(LATEST_RELEASE_TC, LATEST_RELEASE_TC_MBOM, oracle_df)
# dash_app = create_dashboard(combined_df, summary, fig)
# dash_app.run_server(debug=True, host='0.0.0.0', port=8050)
