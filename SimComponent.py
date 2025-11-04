# --- ALL-IN-ONE CELL: data prep ➜ vectorize ➜ LSA/NMF ➜ MiniBatchKMeans/Birch/Agglomerative/NMF ➜ scores ➜ summary ---

import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =========================
# CONFIG
# =========================
INPUT_PATH        = "assets/clustering_data_source.csv"   # <- update if needed
TEXT_COL          = "sentence"

MIN_DF            = 5
MAX_FEATURES      = 10_000
NGRAM_RANGE       = (1, 2)
LOWERCASE         = True
STOP_WORDS        = "english"

SVD_COMPONENTS    = 200
K_MIN, K_MAX      = 8, 60
SAMPLE_FOR_SCORES = 2000   # for silhouette sampling
SHOW_PREVIEW      = 25
RANDOM_STATE      = 42

# =========================
# LOAD + BUILD SENTENCES (keeps your current approach)
# =========================
DATA = pd.read_csv(INPUT_PATH)

# Expecting columns like: did, sentence_beg, place, token, sentence_end
# (This reproduces the flow in your screenshots.)
DATA["token"] = DATA["token"].fillna("").astype(str)

sentences = (
    DATA.sort_values(["did", "sentence_beg", "place"])
        .groupby(["did", "sentence_beg", "sentence_end"])
        .agg({"token": lambda x: " ".join(x)})
        .reset_index()
)

full_sentences = (
    sentences.groupby("did")["token"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"token": TEXT_COL})
)

df = full_sentences.copy()
docs = df[TEXT_COL].astype(str).fillna("")

# =========================
# TF-IDF (Docs x Terms)
# =========================
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
n_terms = len(terms)

# =========================
# LSA on X.T  (Terms x Docs ➜ Term embeddings)
# =========================
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
term_vecs = svd.fit_transform(X.T)                      # (n_terms, SVD_COMPONENTS)
term_vecs = normalize(term_vecs, norm="l2", axis=1)     # cosine-friendly

print(f"LSA term vectors: {term_vecs.shape}")

# =========================
# Heuristic for K (keeps your style)
# =========================
def pick_k(n):
    k = int(round(math.sqrt(max(2, n)/2)))
    return max(K_MIN, min(K_MAX, k))

K = pick_k(n_terms)
print(f"Chosen K = {K} (heuristic)")

# =========================
# Helpers
# =========================
def build_mapping(terms_arr, labels, doc_freq):
    """Return {head_term: [synonyms]} where head is most frequent term in each cluster."""
    groups = defaultdict(list)
    for t, l in zip(terms_arr, labels):
        groups[int(l)].append(t)

    mapping = {}
    idx = {t: i for i, t in enumerate(terms_arr)}
    for l, ts in groups.items():
        head = max(ts, key=lambda t: (doc_freq[idx[t]], -len(t)))
        syns = [t for t in ts if t != head]
        mapping[head] = sorted(syns, key=lambda s: (-doc_freq[idx[s]], len(s)))
    return mapping, groups

def fast_scores(embeddings, labels, sample_cap=2000):
    """Silhouette on random sample + simple cluster-size stats (cosine)."""
    n = embeddings.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan, 0.0

    rng = np.random.default_rng(RANDOM_STATE)
    if sample_cap:
        sel = rng.choice(n, size=min(sample_cap, n), replace=False)
        E = embeddings[sel]
        y = np.asarray(labels)[sel]
    else:
        E = embeddings
        y = np.asarray(labels)

    if len(np.unique(y)) < 2:
        sil = np.nan
    else:
        sil = float(silhouette_score(E, y, metric="cosine"))

    # size stats
    counts = Counter(labels).values()
    mean_sz = float(np.mean(list(counts)))
    med_sz  = float(np.median(list(counts)))
    pct_small = float(sum(c <= 2 for c in counts)) / max(1, len(labels))
    return sil, mean_sz, med_sz, pct_small

# presence to estimate term (document) frequency for sorting heads/synonyms
df_presence = (X > 0).astype(np.int32)
term_doc_freq = np.asarray(df_presence.sum(axis=0)).ravel()

# =========================
# --- Method A: MiniBatchKMeans on term_vecs (LSA) ---
# =========================
kmeans = MiniBatchKMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    batch_size=4096,
    n_init=5
)
labels_A = kmeans.fit_predict(term_vecs)
map_A, groups_A = build_mapping(terms, labels_A, term_doc_freq)
sil_A, mean_A, med_A, small_A = fast_scores(term_vecs, labels_A, sample_cap=SAMPLE_FOR_SCORES)

# =========================
# --- Method B: Birch on term_vecs (LSA) ---
# =========================
birch = Birch(n_clusters=K, threshold=0.5)
labels_B = birch.fit_predict(term_vecs)
map_B, groups_B = build_mapping(terms, labels_B, term_doc_freq)
sil_B, mean_B, med_B, small_B = fast_scores(term_vecs, labels_B, sample_cap=SAMPLE_FOR_SCORES)

# =========================
# --- Method C: Agglomerative on term_vecs (LSA) ---
#      Use cosine metric with average linkage (scikit >= 1.2 uses `metric`, not `affinity`)
# =========================
agg = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average")
labels_C = agg.fit_predict(term_vecs)
map_C, groups_C = build_mapping(terms, labels_C, term_doc_freq)
sil_C, mean_C, med_C, small_C = fast_scores(term_vecs, labels_C, sample_cap=SAMPLE_FOR_SCORES)

# =========================
# --- Method D: NMF topics in topic-space (H^T) ---
# =========================
nmf = NMF(n_components=K, init="nndsvd", random_state=RANDOM_STATE, max_iter=300)
W = nmf.fit_transform(X)      # docs x topics
H = nmf.components_           # topics x terms
T_topic = normalize(H.T, norm="l2", axis=1)   # term loadings per topic vectorized as terms x topics
labels_D = H.argmax(axis=0)   # assign each term to its strongest topic
map_D, groups_D = build_mapping(terms, labels_D, term_doc_freq)
sil_D, mean_D, med_D, small_D = fast_scores(T_topic, labels_D, sample_cap=SAMPLE_FOR_SCORES)

# =========================
# SUMMARY TABLE
# =========================
summary = pd.DataFrame(
    [
        {
            "method": "MiniBatchKMeans",
            "space":  f"LSA (SVD={SVD_COMPONENTS})",
            "K": K,
            "silhouette_cosine(sampled)": round(sil_A, 4),
            "mean_cluster_size": round(mean_A, 1),
            "median_cluster_size": round(med_A, 1),
            "pct_terms_in_tiny_clusters(<=2)": round(small_A, 4),
            "n_clusters": int(len(np.unique(labels_A))),
        },
        {
            "method": "Birch",
            "space":  f"LSA (SVD={SVD_COMPONENTS})",
            "K": K,
            "silhouette_cosine(sampled)": round(sil_B, 4),
            "mean_cluster_size": round(mean_B, 1),
            "median_cluster_size": round(med_B, 1),
            "pct_terms_in_tiny_clusters(<=2)": round(small_B, 4),
            "n_clusters": int(len(np.unique(labels_B))),
        },
        {
            "method": "Agglomerative (avg-link, cosine)",
            "space":  f"LSA (SVD={SVD_COMPONENTS})",
            "K": K,
            "silhouette_cosine(sampled)": round(sil_C, 4),
            "mean_cluster_size": round(mean_C, 1),
            "median_cluster_size": round(med_C, 1),
            "pct_terms_in_tiny_clusters(<=2)": round(small_C, 4),
            "n_clusters": int(len(np.unique(labels_C))),
        },
        {
            "method": "NMF topics",
            "space":  "Topic loadings (Hᵀ)",
            "K": K,
            "silhouette_cosine(sampled)": round(sil_D, 4),
            "mean_cluster_size": round(mean_D, 1),
            "median_cluster_size": round(med_D, 1),
            "pct_terms_in_tiny_clusters(<=2)": round(small_D, 4),
            "n_clusters": int(len(np.unique(labels_D))),
        },
    ],
    columns=[
        "method","space","K","silhouette_cosine(sampled)",
        "mean_cluster_size","median_cluster_size",
        "pct_terms_in_tiny_clusters(<=2)","n_clusters"
    ]
)

# Optional: quick peek at cluster heads/synonyms for each method
def preview_mapping(name, mapping, n=SHOW_PREVIEW):
    print(f"\n[{name}] preview of {min(n, len(mapping))} heads")
    for i, (head, syns) in enumerate(sorted(mapping.items(), key=lambda kv: -len(kv[1]))[:n], start=1):
        tail = ", ".join(syns[:12]) + (" ..." if len(syns) > 12 else "")
        print(f"{i:>3}. {head}: {tail}")

preview_mapping("MiniBatchKMeans", map_A)
preview_mapping("Birch", map_B)
preview_mapping("Agglomerative", map_C)
preview_mapping("NMF topics", map_D)

print("\nSUMMARY:")
display(summary)