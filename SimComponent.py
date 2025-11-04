import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go

tuning_summary = RESULTS["tuning_summary"]
best_K = RESULTS["best_K"]
topic_df = RESULTS["topic_df"]
docs_df = RESULTS["docs_df"]
labels = RESULTS["labels"]
H = RESULTS["H"]
terms = RESULTS["terms"]

# 1) Silhouette vs K (tuning curve)
fig1 = px.line(
    tuning_summary, x="K", y="silhouette_cosine(sampled)",
    markers=True, title="NMF Tuning: Silhouette (cosine) vs K"
)
fig1.update_layout(yaxis_title="Silhouette (cosine)", xaxis=dict(dtick=10))
fig1.show()

# 2) Topic sizes (doc counts per topic)
counts = pd.Series(labels).value_counts().sort_index()
size_df = pd.DataFrame({"topic": [f"Topic {i:02d}" for i in counts.index], "docs": counts.values})
fig2 = px.bar(size_df, x="topic", y="docs", title=f"Topic Sizes (K={best_K})")
fig2.update_layout(xaxis_title="", yaxis_title="# of Documents")
fig2.show()

# 3) Top terms for the top-N biggest topics (horizontal bars)
TOP_N = 5
big_topics = counts.sort_values(ascending=False).index[:TOP_N].tolist()

# Build a long DF: (topic, term, weight) for the top terms of each selected topic
rows = []
TOP_TERMS_PER_TOPIC = 12
for t in big_topics:
    top_idx = np.argsort(H[t])[::-1][:TOP_TERMS_PER_TOPIC]
    for i in top_idx:
        rows.append({"topic": f"Topic {t:02d}", "term": terms[i], "weight": H[t, i]})
long_terms = pd.DataFrame(rows)

fig3 = px.bar(
    long_terms, x="weight", y="term", color="topic",
    orientation="h", facet_row="topic",
    title=f"Top Terms per Topic (Top {TOP_N} by size)"
)
fig3.update_layout(showlegend=False)
fig3.show()

# 4) Summary table (nice for screenshots)
fig4 = go.Figure(
    data=[go.Table(
        header=dict(values=list(tuning_summary.columns)),
        cells=dict(values=[tuning_summary[c] for c in tuning_summary.columns])
    )]
)
fig4.update_layout(title="NMF Tuning Summary")
fig4.show()