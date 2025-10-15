# ============================================================
# Fast synonym-ish term grouping (scikit-learn only, <5 min)
# Methods: MiniBatchKMeans, Birch, NMF topics
# Uses TF-IDF -> TruncatedSVD (LSA) for speedy, dense term vectors
# ============================================================

# -----------------------------
# CONFIG — tweak as needed
# -----------------------------
DATA_PATH     = "assets/reliability_data_test.json"
TEXT_COL      = "trimmed"

MIN_DF        = 5             # drop very-rare terms
MAX_FEATURES  = 10000         # limit vocab for speed/memory
NGRAM_RANGE   = (1, 2)        # unigrams + bigrams
LOWERCASE     = True
STOP_WORDS    = "english"

SVD_COMPONENTS = 200          # LSA dimensionality (100–300 works well)
K_MIN, K_MAX   = 8, 60        # clamp cluster/topic count
SAMPLE_FOR_SCORES = 2000      # max sampled terms for silhouette

SHOW_PREVIEW   = 25           # heads per method to preview (0 to skip)
RANDOM_STATE   = 42

# -----------------------------
# Imports
# -----------------------------
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_score

# -----------------------------
# Load data
# -----------------------------
df = pd.read_json(DATA_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not in df. Available: {list(df.columns)}")
docs = df[TEXT_COL].astype(str).fillna("")
print(f"Loaded df: {df.shape}, using '{TEXT_COL}'.")
print(df[[TEXT_COL]].head(3), "\n")

# -----------------------------
# TF-IDF (docs x terms) — sparse and fast
# -----------------------------
tfidf = TfidfVectorizer(
    lowercase=LOWERCASE,
    stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
    norm="l2",
)
X = tfidf.fit_transform(docs)             # (n_docs, n_terms), sparse
terms = np.array(tfidf.get_feature_names_out())
n_terms = len(terms)
if n_terms == 0:
    raise ValueError("No terms survived vectorization. Lower MIN_DF or adjust preprocessing.")
print(f"TF-IDF: docs x terms = {X.shape} | vocab size = {n_terms}")

# -----------------------------
# Fast term embeddings via LSA
# SVD on X.T (terms x docs) -> dense (n_terms x SVD_COMPONENTS)
# -----------------------------
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
term_vecs = svd.fit_transform(X.T)        # dense, compact
term_vecs = normalize(term_vecs, norm="l2", axis=1)  # cosine-friendly
print(f"LSA term vectors: {term_vecs.shape}\n")

# -----------------------------
# Heuristic for K (no tuning loop)
# -----------------------------
def pick_k(n):
    k = int(round(math.sqrt(max(2, n)/2)))
    return max(K_MIN, min(K_MAX, k))
K = pick_k(n_terms)
print(f"Chosen K = {K} (heuristic)\n")

# -----------------------------
# Helpers: mapping, preview, metrics (FAST)
# -----------------------------
def build_mapping(terms_arr, labels, doc_freq):
    """Head = most frequent term in cluster (tie-break: shortest)."""
    groups = defaultdict(list)
    for t, l in zip(terms_arr, labels):
        groups[int(l)].append(t)
    mapping = {}
    # quick index for freq lookup
    idx = {t: i for i, t in enumerate(terms_arr)}
    for l, ts in groups.items():
        head = max(ts, key=lambda t: (doc_freq[idx[t]], -len(t)))
        syns = [t for t in ts if t != head]
        mapping[head] = sorted(syns, key=lambda s: (-doc_freq[idx[s]], len(s)))
    return mapping, groups

def preview_mapping(name, mapping, n=25):
    if n <= 0 or not mapping:
        return
    print(f"[{name}] preview of {min(n, len(mapping))} heads (sorted by #synonyms):")
    for head, syns in sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n]:
        print(f"{head:22s} : {', '.join(syns) if syns else '(singleton)'}")
    print()

def fast_scores(embeddings, labels, sample_cap=2000):
    """Silhouette (cosine) on a random sample + cluster size stats. No O(n²) heavy sims."""
    n = embeddings.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(RANDOM_STATE)
    if n > sample_cap:
        sel = rng.choice(n, size=sample_cap, replace=False)
        E = embeddings[sel]
        y = np.asarray(labels)[sel]
    else:
        E = embeddings
        y = np.asarray(labels)
    # require at least 2 labels for silhouette
    if len(np.unique(y)) < 2:
        sil = np.nan
    else:
        sil = float(silhouette_score(E, y, metric="cosine"))
    # size stats
    counts = Counter(labels).values()
    mean_sz = float(np.mean(list(counts)))
    med_sz  = float(np.median(list(counts)))
    pct_small = float(sum(c <= 2 for c in counts) / max(1, len(labels)))  # tiny clusters share
    return sil, mean_sz, med_sz, pct_small

# For frequency-based head selection, we use doc frequency from TF-IDF presence
# (more robust than raw counts here; quick proxy)
df_presence = (X > 0).astype(np.int32)
term_doc_freq = np.asarray(df_presence.sum(axis=0)).ravel()  # per-term in original 'terms' order

# -----------------------------
# Method A: MiniBatchKMeans on LSA term vectors (FAST)
# -----------------------------
kmeans = MiniBatchKMeans(n_clusters=K, random_state=RANDOM_STATE, batch_size=4096, n_init=5)
labels_A = kmeans.fit_predict(term_vecs)
map_A, groups_A = build_mapping(terms, labels_A, term_doc_freq)
sil_A, mean_A, med_A, small_A = fast_scores(term_vecs, labels_A, sample_cap=SAMPLE_FOR_SCORES)

# -----------------------------
# Method B: Birch (FAST, incremental)
# -----------------------------
birch = Birch(n_clusters=K, threshold=0.5)  # threshold is mild; K controls final clusters
labels_B = birch.fit_predict(term_vecs)
map_B, groups_B = build_mapping(terms, labels_B, term_doc_freq)
sil_B, mean_B, med_B, small_B = fast_scores(term_vecs, labels_B, sample_cap=SAMPLE_FOR_SCORES)

# -----------------------------
# Method C: NMF topics (on sparse X), evaluate in topic space
# -----------------------------
nmf = NMF(n_components=K, init="nndsvd", random_state=RANDOM_STATE, max_iter=300)
W = nmf.fit_transform(X)          # (docs x K)
H = nmf.components_               # (K x terms)
labels_C = H.argmax(axis=0)       # topic per term
T_topic = normalize(H.T, norm="l2", axis=1)   # (terms x K)
map_C, groups_C = build_mapping(terms, labels_C, term_doc_freq)
sil_C, mean_C, med_C, small_C = fast_scores(T_topic, labels_C, sample_cap=SAMPLE_FOR_SCORES)

# -----------------------------
# Summary table
# -----------------------------
summary = pd.DataFrame([
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
        "method": "NMF topics",
        "space":  "Topic loadings (Hᵀ)",
        "K": K,
        "silhouette_cosine(sampled)": round(sil_C, 4),
        "mean_cluster_size": round(mean_C, 1),
        "median_cluster_size": round(med_C, 1),
        "pct_terms_in_tiny_clusters(<=2)": round(small_C, 4),
        "n_clusters": int(len(np.unique(labels_C))),
    },
], columns=[
    "method","space","K","silhouette_cosine(sampled)","mean_cluster_size",
    "median_cluster_size","pct_terms_in_tiny_clusters(<=2)","n_clusters"
])

print("\n==== Fast Term Grouping Summary ====\n")
display(summary)

# -----------------------------
# Quick previews (heads -> synonyms) for sanity
# -----------------------------
def preview(name, mapping, n=SHOW_PREVIEW):
    if n <= 0: return
    preview_mapping(name, mapping, n=n)

preview("MiniBatchKMeans", map_A, SHOW_PREVIEW)
preview("Birch",          map_B, SHOW_PREVIEW)
preview("NMF topics",     map_C, SHOW_PREVIEW)

# -----------------------------
# Notes:
# - This is fast because we: (1) limit vocab, (2) use SVD to 200 dims, (3) sample for silhouette.
# - Pick the winner by higher silhouette and clean previews.
# - If you need even faster: lower MAX_FEATURES (e.g., 7000) or SVD_COMPONENTS (e.g., 150).
# -----------------------------