# Plotly bars of top words per topic (interactive + optional faceted view)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Pull from the RESULTS produced by the NMF cell
H = RESULTS["H"]          # topics x terms matrix
terms = RESULTS["terms"]  # array of term strings
labels = RESULTS["labels"]  # dominant topic per doc
best_K = RESULTS["best_K"]

TOP_TERMS_PER_TOPIC = 15   # change to taste
TITLE_PREFIX = f"NMF (K={best_K}) – Top {TOP_TERMS_PER_TOPIC} Terms"

# ---------- Helper: build a long dataframe for a given list of topic ids ----------
def long_terms_for_topics(topic_ids, top_n=15):
    rows = []
    for t in topic_ids:
        top_idx = np.argsort(H[t])[::-1][:top_n]
        for i in top_idx:
            rows.append({
                "topic_id": t,
                "topic": f"Topic {t:02d}",
                "term": terms[i],
                "weight": float(H[t, i]),
                "rank": int(np.where(top_idx == i)[0][0] + 1),
            })
    return pd.DataFrame(rows)

# ---------- 1) Interactive single-topic view with a dropdown ----------
topic_ids = list(range(H.shape[0]))
frames = []
buttons = []

for t in topic_ids:
    df_t = long_terms_for_topics([t], top_n=TOP_TERMS_PER_TOPIC)
    frames.append(go.Frame(
        name=str(t),
        data=[go.Bar(x=df_t["weight"], y=df_t["term"], orientation="h")]
    ))
    buttons.append(dict(label=f"Topic {t:02d}",
                        method="animate",
                        args=[[str(t)], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}]))

# initial (Topic 0)
df0 = long_terms_for_topics([0], top_n=TOP_TERMS_PER_TOPIC)
fig_single = go.Figure(
    data=[go.Bar(x=df0["weight"], y=df0["term"], orientation="h")],
    layout=go.Layout(
        title=f"{TITLE_PREFIX}: Topic 00",
        xaxis_title="Term weight (H)",
        yaxis_title="Term",
        updatemenus=[dict(
            type="dropdown", x=1.02, xanchor="left", y=1, yanchor="top",
            buttons=buttons, showactive=True
        )]
    ),
    frames=frames
)
fig_single.update_layout(yaxis=dict(autorange="reversed"))  # biggest at top
fig_single.show()

# ---------- 2) (Optional) Faceted view for Top-N largest topics by doc count ----------
TOP_N = 6  # change to taste; None to skip
if TOP_N:
    size_by_topic = pd.Series(labels).value_counts().sort_values(ascending=False)
    top_topic_ids = size_by_topic.index[:TOP_N].tolist()
    long_df = long_terms_for_topics(top_topic_ids, top_n=TOP_TERMS_PER_TOPIC)

    fig_facets = px.bar(
        long_df, x="weight", y="term",
        facet_row="topic", orientation="h",
        title=f"{TITLE_PREFIX} per Topic – Top {TOP_N} Topics by Size",
        color="topic"  # just to separate facets in legend (legend hidden)
    )
    fig_facets.update_layout(showlegend=False, xaxis_title="Term weight (H)")
    # Make each facet order terms top→bottom
    for a in fig_facets.select_yaxes():
        a.update(autorange="reversed")
    fig_facets.show()