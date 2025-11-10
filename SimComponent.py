# --- Imports
import re
from collections import Counter

import numpy as np
import pandas as pd

from IPython.display import display

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# ==============================
# Config
# ==============================
INPUT_PATH         = "assets/clustering_data_source.csv"
TEXT_COL           = "sentence"
LOWERCASE          = True
STOP_WORDS         = "english"
NGRAM_RANGE        = (1, 2)
MIN_DF             = 5
MAX_FEATURES       = 10_000
RANDOM_STATE       = 42

# Add Important words to filter topics
IMPORTANT_WORDS_PATH = "assets/important_words.txt"

# Load important words list (comma/whitespace/semicolon/pipe separated; 1 per line also OK)
with open(IMPORTANT_WORDS_PATH, "r", encoding="utf-8") as f:
    important_words = {
        w.strip().lower()
        for w in re.split(r"[,\s;|]+", f.read())
        if w.strip()
    }

# Test multiple K values
K_GRID          = [10, 20, 40, 60, 80, 100]
MAX_ITER        = 400
SAMPLE_FOR_SIL  = 5000   # sample if too large

TOP_TERMS_PER_TOPIC = 20
SHOW_DOC_PREVIEW    = 10


# ==============================
# Build documents
# ==============================
RAW = pd.read_csv(INPUT_PATH)
RAW["token"] = RAW["token"].fillna("").astype(str)

sentences = (
    RAW.sort_values(["did", "sentence_beg", "place"])
       .groupby(["did", "sentence_beg", "sentence_end"])
       .agg({"token": lambda s: " ".join(s)})
       .reset_index()
)

docs_df = (
    sentences.groupby("did")["token"]
             .apply(lambda s: " ".join(s))
             .reset_index()
             .rename(columns={"token": TEXT_COL})
)

docs = docs_df[TEXT_COL].astype(str).fillna("")


# ==============================
# TF-IDF
# ==============================
tfidf = TfidfVectorizer(
    lowercase=LOWERCASE,
    stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
    norm="l2",
)
X = tfidf.fit_transform(docs)
terms = np.array(tfidf.get_feature_names_out())


# ==============================
# Score NMF model using cosine silhouette on normalized doc-topic (W)
# ==============================
def nmf_fit_score(K: int):
    nmf = NMF(
        n_components=K,
        init="nndsvd",
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    W = nmf.fit_transform(X)  # (n_docs, K)
    H = nmf.components_       # (K, n_terms)

    # normalize W for cosine distances; label each doc by strongest topic
    Wn = normalize(W, norm="l2", axis=1)
    labels = Wn.argmax(axis=1)

    # silhouette sample if needed
    n = Wn.shape[0]
    if n > SAMPLE_FOR_SIL:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=SAMPLE_FOR_SIL, replace=False)
        Ws, ys = Wn[idx], labels[idx]
    else:
        Ws, ys = Wn, labels

    sil = float("nan")
    if len(np.unique(ys)) >= 2:
        sil = float(silhouette_score(Ws, ys, metric="cosine"))

    # cluster size stats
    counts = list(Counter(labels).values())
    mean_sz = float(np.mean(counts))
    med_sz  = float(np.median(counts))
    pct_tiny = float(sum(c <= (0.05 * len(labels)) for c in counts) / max(1, len(counts)))

    return nmf, W, H, sil, mean_sz, med_sz, pct_tiny, labels


# ==============================
# Grid search over K
# ==============================
rows   = []
models = {}

for K in K_GRID:
    nmf, W, H, sil, mean_sz, med_sz, pct_tiny, labels = nmf_fit_score(K)
    rows.append({
        "K": K,
        "silhouette_cosine(sampled)": round(sil, 4),
        "mean_cluster_size": round(mean_sz, 1),
        "median_cluster_size": round(med_sz, 1),
        "pct_tiny_topics(<=5%docs)": round(pct_tiny, 4),
    })
    models[K] = (nmf, W, H, labels)

tuning_summary = pd.DataFrame(rows).sort_values("silhouette_cosine(sampled)", ascending=False).reset_index(drop=True)

print("\nNMF tuning summary:")
display(tuning_summary)

best_K = int(tuning_summary.loc[0, "K"])
nmf, W, H, labels = models[best_K]
Wn = normalize(W, norm="l2", axis=1)


# ==============================
# Select top terms per topic
# (NEW) Filter to important_words for display
# ==============================
# vectorized mask of allowed vocab
allowed_mask = np.array([t.lower() in important_words for t in terms])

top_terms = []
for k in range(best_K):
    order = np.argsort(H[k])[::-1]             # strongest -> weakest
    # keep only important words
    filtered = [i for i in order if allowed_mask[i]]
    # if none match, keep empty list (only important words should be shown)
    chosen_idx = filtered[:TOP_TERMS_PER_TOPIC]
    chosen_terms = terms[chosen_idx] if len(chosen_idx) else []
    top_terms.append(list(chosen_terms))

topic_df = pd.DataFrame({
    "topic":     [f"Topic {i:02d}" for i in range(best_K)],
    "top_terms": [", ".join(tt) if len(tt) else "(no important terms)" for tt in top_terms],
})


# ==============================
# Document assignments
# ==============================
docs_df["topic"] = labels
docs_df["topic_prob"] = W.max(axis=1)
docs_preview = docs_df[["did", TEXT_COL, "topic", "topic_prob"]].head(SHOW_DOC_PREVIEW)

print("\nTop terms per topic:")
display(topic_df.head(10))

print("\nSample of document assignments:")
display(docs_preview)

RESULTS = {
    "tuning_summary": tuning_summary,
    "best_K": best_K,
    "topic_df": topic_df,
    "docs_df": docs_df,
    "W": W, "Wn": Wn, "H": H, "terms": terms, "labels": labels,
}