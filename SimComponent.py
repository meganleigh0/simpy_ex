# === ONE CELL: build documents ➜ TF-IDF ➜ NMF (scan K) ➜ pick best ➜ outputs ===
import math, re, numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# --------------------
# Config (safe, strong defaults)
# --------------------
INPUT_PATH         = "assets/clustering_data_source.csv"   # did, sentence_beg, place, token, sentence_end
TEXT_COL           = "sentence"
LOWERCASE          = True
STOP_WORDS         = "english"
NGRAM_RANGE        = (1, 2)
MIN_DF             = 5
MAX_FEATURES       = 10_000
RANDOM_STATE       = 42

# Try a few K values and pick by cosine silhouette on doc-topic mixtures (W)
K_GRID             = [20, 40, 60, 80] 
MAX_ITER           = 400
SAMPLE_FOR_SIL     = 5000   # sample docs for silhouette if very large

TOP_TERMS_PER_TOPIC = 20
SHOW_DOC_PREVIEW     = 10

# --------------------
# Build documents (same logic as your screenshots: join tokens into sentences, then per did)
# --------------------
RAW = pd.read_csv(INPUT_PATH)
RAW["token"] = RAW["token"].fillna("").astype(str)

sentences = (
    RAW.sort_values(["did", "sentence_beg", "place"])
       .groupby(["did", "sentence_beg", "sentence_end"])
       .agg({"token": lambda s: " ".join(s)})
       .reset_index()
)

docs_df = (
    sentences.groupby("did")["token"].apply(lambda s: " ".join(s))
             .reset_index().rename(columns={"token": TEXT_COL})
)

docs = docs_df[TEXT_COL].astype(str).fillna("")

# --------------------
# TF-IDF doc-term matrix
# --------------------
tfidf = TfidfVectorizer(
    lowercase=LOWERCASE, stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE, min_df=MIN_DF, max_features=MAX_FEATURES, norm="l2"
)
X = tfidf.fit_transform(docs)                # (n_docs, n_terms)
terms = np.array(tfidf.get_feature_names_out())

# --------------------
# Helper: score an NMF model using cosine silhouette on normalized doc-topic (W)
# --------------------
def nmf_fit_score(K):
    nmf = NMF(
        n_components=int(K), init="nndsvd", random_state=RANDOM_STATE,
        max_iter=MAX_ITER, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0
    )
    W = nmf.fit_transform(X)         # (n_docs, K)
    H = nmf.components_              # (K, n_terms)
    # normalize W for cosine distances; label each doc by strongest topic
    Wn = normalize(W, norm="l2", axis=1)
    labels = Wn.argmax(axis=1)

    # sample for silhouette if needed
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
    counts = Counter(labels).values()
    mean_sz = float(np.mean(list(counts)))
    med_sz  = float(np.median(list(counts)))
    pct_tiny = float(sum(c <= 5 for c in counts)) / max(1, len(counts))
    return nmf, W, H, sil, mean_sz, med_sz, pct_tiny, labels

# --------------------
# Scan K, pick the winner
# --------------------
rows = []
models = {}
for K in K_GRID:
    nmf, W, H, sil, mean_sz, med_sz, pct_tiny, labels = nmf_fit_score(K)
    rows.append({
        "K": K,
        "silhouette_cosine(sampled)": round(sil, 4),
        "mean_cluster_size": round(mean_sz, 1),
        "median_cluster_size": round(med_sz, 1),
        "pct_tiny_topics(<=5docs)": round(pct_tiny, 4)
    })
    models[K] = (nmf, W, H, labels)

tuning_summary = pd.DataFrame(rows).sort_values("silhouette_cosine(sampled)", ascending=False).reset_index(drop=True)
best_K = int(tuning_summary.loc[0, "K"])
nmf, W, H, labels = models[best_K]
Wn = normalize(W, norm="l2", axis=1)

print("NMF tuning summary (higher silhouette is better):")
display(tuning_summary)
print(f"\nSelected K = {best_K}")

# --------------------
# Topic descriptors: top terms per topic
# --------------------
top_terms = []
for k in range(best_K):
    top_idx = np.argsort(H[k])[::-1][:TOP_TERMS_PER_TOPIC]
    top_terms.append([terms[i] for i in top_idx])

topic_df = pd.DataFrame(
    {"topic": [f"Topic {i:02d}" for i in range(best_K)],
     "top_terms": [", ".join(tt) for tt in top_terms]}
)

# --------------------
# Document assignments (preview)
# --------------------
docs_df["topic"] = labels
docs_df["topic_prob"] = W.max(axis=1)  # unnormalized max weight (proxy)
docs_preview = docs_df[["did", TEXT_COL, "topic", "topic_prob"]].head(SHOW_DOC_PREVIEW)

print("\nTop terms per topic:")
display(topic_df.head(10))
print("\nSample of document assignments:")
display(docs_preview)

# Also keep objects in memory for the Plotly cell
RESULTS = {
    "tuning_summary": tuning_summary,
    "best_K": best_K,
    "topic_df": topic_df,
    "docs_df": docs_df,
    "W": W, "Wn": Wn, "H": H, "terms": terms, "labels": labels
}