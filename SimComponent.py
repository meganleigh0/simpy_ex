# --- Plotly: dropdown to browse top important words per topic (with weights) ---

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go

TOP_TERMS_PER_TOPIC = 15  # how many words to show per topic

def is_important(term: str) -> bool:
    # treat a term as important if ANY token (unigram/bigram) is in the list
    toks = re.split(r"\s+", term.lower().strip())
    return any(tok in important_words for tok in toks)

def topic_terms_with_weights(topic_id: int, top_n: int = TOP_TERMS_PER_TOPIC):
    weights = H[topic_id]                      # (n_terms,)
    idx_sorted = np.argsort(weights)[::-1]     # high → low
    idx_keep = [i for i in idx_sorted if is_important(terms[i])]
    idx_top = idx_keep[:top_n]
    y_terms = [str(terms[i]) for i in idx_top]
    x_wts   = [float(weights[i]) for i in idx_top]
    return y_terms, x_wts

# Build frames & dropdown buttons (one per topic)
frames = []
buttons = []
for t in range(K):
    y_terms, x_wts = topic_terms_with_weights(t, TOP_TERMS_PER_TOPIC)
    frames.append(go.Frame(
        name=str(t),
        data=[go.Bar(x=x_wts, y=y_terms, orientation="h")]
    ))
    buttons.append(dict(
        label=f"Topic {t:02d}",
        method="animate",
        args=[[str(t)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
    ))

# Initial view (Topic 0)
y0, x0 = topic_terms_with_weights(0, TOP_TERMS_PER_TOPIC)
fig = go.Figure(
    data=[go.Bar(x=x0, y=y0, orientation="h")],
    layout=go.Layout(
        title=f"NMF (K={K}) – Top {TOP_TERMS_PER_TOPIC} Important Terms • Topic 00",
        xaxis_title="Term weight (H)",
        yaxis_title="Term",
        updatemenus=[dict(
            type="dropdown", x=1.02, xanchor="left", y=1, yanchor="top",
            buttons=buttons, showactive=True
        )]
    ),
    frames=frames
)

# Put the largest bar at the top for readability
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.show()