import plotly.express as px
import numpy as np
import pandas as pd

# show top words for each topic
TOP_N = 6
rows = []
for t in range(min(TOP_N, len(topic_df))):
    for term in topic_df["top_terms"][t].split(", "):
        rows.append({"topic": f"Topic {t:02d}", "term": term.strip()})
df_long = pd.DataFrame(rows)

fig = px.bar(
    df_long,
    y="term", x="topic",
    orientation="h",
    color="topic",
    title="Filtered NMF Topics — Top Words per Topic",
)
fig.update_layout(
    showlegend=False,
    xaxis_title="Topic ID",
    yaxis_title="Important Terms",
)
fig.show()