# --- PLOTLY CELL: visualize the summary metrics ---

import plotly.express as px
import plotly.graph_objects as go

# Expect `summary` DataFrame from previous cell.
sum_long = summary.melt(
    id_vars=["method", "space", "K", "n_clusters"],
    value_vars=["silhouette_cosine(sampled)", "mean_cluster_size", "median_cluster_size", "pct_terms_in_tiny_clusters(<=2)"],
    var_name="metric",
    value_name="value"
)

# 1) Silhouette bar chart
fig_sil = px.bar(
    summary,
    x="method",
    y="silhouette_cosine(sampled)",
    text="silhouette_cosine(sampled)",
    title=f"Cosine Silhouette (sampled) — K={int(summary['K'].iloc[0])}",
)
fig_sil.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_sil.update_layout(xaxis_title="", yaxis_title="Silhouette (cosine)")
fig_sil.show()

# 2) Cluster size bars (mean vs median) grouped
fig_sizes = go.Figure()
fig_sizes.add_bar(name="Mean cluster size", x=summary["method"], y=summary["mean_cluster_size"])
fig_sizes.add_bar(name="Median cluster size", x=summary["method"], y=summary["median_cluster_size"])
fig_sizes.update_layout(
    barmode="group",
    title=f"Cluster Size (Mean vs Median) — K={int(summary['K'].iloc[0])}",
    xaxis_title="", yaxis_title="Size (# terms)"
)
fig_sizes.show()

# 3) Percent tiny clusters (<=2)
fig_tiny = px.bar(
    summary,
    x="method",
    y="pct_terms_in_tiny_clusters(<=2)",
    text="pct_terms_in_tiny_clusters(<=2)",
    title="Percent of Tiny Clusters (≤ 2 terms)",
)
fig_tiny.update_traces(texttemplate="%{text:.2%}", textposition="outside")
fig_tiny.update_layout(xaxis_title="", yaxis_title="Percent")
fig_tiny.show()

# 4) Summary table (nice to have)
fig_table = go.Figure(
    data=[
        go.Table(
            header=dict(values=list(summary.columns)),
            cells=dict(values=[summary[c] for c in summary.columns])
        )
    ]
)
fig_table.update_layout(title="Model Comparison Summary")
fig_table.show()