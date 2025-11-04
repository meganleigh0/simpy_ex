# === Plotly: Bar charts of top words per topic ===

import pandas as pd
import numpy as np
import plotly.express as px

TOP_WORDS = 12  # modify if you want more or fewer terms

rows = []
for t in range(best_K):
    top_idx = np.argsort(H[t])[::-1][:TOP_WORDS]
    for i in top_idx:
        rows.append({
            "topic": f"Topic {t:02d}",
            "term": terms[i],
            "weight": float(H[t, i])
        })

top_terms_df = pd.DataFrame(rows)

fig = px.bar(
    top_terms_df,
    x="weight",
    y="term",
    color="topic",
    orientation="h",
    facet_row="topic",
    title=f"Top {TOP_WORDS} Terms per Topic (K={best_K})",
    height=300 * min(best_K, 8)  # auto scales (#topics capped visually at ~8)
)

fig.update_layout(
    showlegend=False,
    xaxis_title="Topic Weight",
    yaxis_title="Term",
    title_font_size=20
)

fig.update_traces(hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}")
fig.show()