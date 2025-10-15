# ============================================================
# Synonym-like term grouping with TF-IDF + (Agglomerative, KMeans, NMF)
# One cell • scikit-learn only • dense conversion fixed
# ============================================================

# -----------------------------
# CONFIG — tweak as needed
# -----------------------------
DATA_PATH    = "assets/reliability_data_test.json"
TEXT_COL     = "trimmed"

MIN_DF       = 5            # drop very-rare terms (min doc freq)
MAX_FEATURES = 15000        # cap vocab to keep memory OK when densifying
NGRAM_RANGE  = (1, 2)       # unigrams + bigrams (captures short phrases)
LOWERCASE    = True
STOP_WORDS   = "english"

# Cluster/topic count heuristic (no tuning step):
# k ≈ sqrt(#terms/2), clipped to [8, 80].
K_LOWER, K_UPPER = 8, 80

SHOW_PREVIEW = 30           # preview lines per method (0 to skip)

# -----------------------------
# Imports
# -----------------------------
import math
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load data
# -----------------------------
df = pd.read_json(DATA_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not in df. Available: {list(df.columns)}")
docs = df[TEXT_COL].astype(str).fillna("")
print(f"Loaded df: {df.shape}, using '{TEXT_COL}'. First 3 rows:")
print(df[[TEXT_COL]].head(3), "\n")

# -----------------------------
# TF-IDF (docs x terms) + raw counts for term frequency
# -----------------------------
tfidf = TfidfVectorizer(
    lowercase=LOWERCASE,
    stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
    norm="l2",           # cosine-friendly
)
X = tfidf.fit_transform(docs)      # SPARSE (n_docs, n_terms)
terms = np.array(tfidf.get_feature_names_out())
n_terms = len(terms)
if n_terms == 0:
    raise ValueError("No terms survived vectorization. Lower MIN_DF or change preprocessing.")
print(f"TF-IDF shape: {X.shape} | vocab size: {n_terms}")

# Raw counts for frequency-based 'head' selection
cv = CountVectorizer(
    lowercase=LOWERCASE,
    stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
)
C = cv.fit_transform(docs)         # SPARSE (n_docs, n_terms2)
cv_terms = np.array(cv.get_feature_names_out())

# Align vocabularies between TF-IDF and Count
common, tfidf_idx, cv_idx = np.intersect1d(terms, cv_terms, return_indices=True)
if len(common) < n_terms:
    X = X[:, tfidf_idx]            # keep only common columns in TF-IDF
    terms = terms[tfidf_idx]
    n_terms = len(terms)
    print(f"Aligned to common vocab of size {n_terms} for frequency lookup.")
term_freq = np.asarray(C[:, cv_idx].sum(axis=0)).ravel()  # frequency per term matching 'terms'

# -----------------------------
# Dense term vectors (terms x docs) for clustering
# NOTE: Agglomerative/KMeans require DENSE inputs.
# -----------------------------
# T_sparse: (terms x docs) but still sparse. Convert to DENSE via .toarray() safely sized by MAX_FEATURES.
T_dense = normalize(X.T.toarray(), norm="l2", axis=1)     # (n_terms, n_docs), L2-normalized

# -----------------------------
# Heuristic for number of clusters/topics
# -----------------------------
def pick_k(n_terms):
    k = int(round(math.sqrt(max(2, n_terms)/2.0)))
    return max(K_LOWER, min(K_UPPER, k))
K = pick_k(n_terms)
print(f"Chosen K = {K} (heuristic)\n")

# -----------------------------
# Metrics
# -----------------------------
def evaluate_labels(term_matrix_dense, labels):
    """
    term_matrix_dense: (n_terms, dim) dense L2-normalized vectors
    labels: cluster/topic label for each term (len = n_terms)
    Returns:
      n_terms, n_clusters, pct_singletons, mean_intra_sim, mean_inter_sim
    """
    labels = np.asarray(labels)
    n_terms = len(labels)
    uniq = np.unique(labels)
    n_clusters = len(uniq)
    counts = [np.sum(labels == l) for l in uniq]
    pct_singletons = (sum(1 for c in counts if c == 1) / max(1, n_terms))

    # Cosine similarity across terms (dense)
    # If vocab is large, this can be big; for typical sizes (<6k) it's OK.
    sims = cosine_similarity(term_matrix_dense)  # (n_terms, n_terms)

    intra_vals = []
    idx_by_lab = {l: np.where(labels == l)[0] for l in uniq}
    for l, idx in idx_by_lab.items():
        if len(idx) > 1:
            block = sims[np.ix_(idx, idx)]
            m = (block.sum() - np.trace(block)) / (len(idx) * (len(idx) - 1))
            intra_vals.append(m)

    inter_vals = []
    labs = list(idx_by_lab.keys())
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            a, b = idx_by_lab[labs[i]], idx_by_lab[labs[j]]
            inter_vals.append(sims[np.ix_(a, b)].mean())

    mean_intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
    mean_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    return n_terms, n_clusters, pct_singletons, mean_intra, mean_inter

def build_mapping(terms_arr, labels, freqs):
    """Group terms by label; 'head' = highest frequency (tie-break: shortest)."""
    groups = defaultdict(list)
    for t, l in zip(terms_arr, labels):
        groups[int(l)].append(t)

    # quick index for frequencies
    term_to_idx = {t: i for i, t in enumerate(terms_arr)}
    mapping = {}
    for l, ts in groups.items():
        head = max(ts, key=lambda t: (freqs[term_to_idx[t]], -len(t)))
        syns = [t for t in ts if t != head]
        mapping[head] = sorted(syns, key=lambda s: (-freqs[term_to_idx[s]], len(s)))
    return mapping

def preview_mapping(name, mapping, n=30):
    if n <= 0 or not mapping:
        return
    print(f"[{name}] preview of {min(n, len(mapping))} heads (sorted by #synonyms):")
    for head, syns in sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n]:
        print(f"{head:22s} : {', '.join(syns) if syns else '(singleton)'}")
    print()

# -----------------------------
# METHOD A: Agglomerative (cosine, average)
# (Handles version diff: metric vs affinity)
# -----------------------------
try:
    agg = AgglomerativeClustering(n_clusters=K, linkage="average", metric="cosine")
except TypeError:
    # older scikit-learn
    agg = AgglomerativeClustering(n_clusters=K, linkage="average", affinity="cosine")
labels_A = agg.fit_predict(T_dense)
nA, kA, sA, intraA, interA = evaluate_labels(T_dense, labels_A)
map_A = build_mapping(terms, labels_A, term_freq)

# -----------------------------
# METHOD B: KMeans (euclidean) on dense term vectors
# -----------------------------
try:
    kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
except TypeError:
    # older scikit-learn
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
labels_B = kmeans.fit_predict(T_dense)
nB, kB, sB, intraB, interB = evaluate_labels(T_dense, labels_B)
map_B = build_mapping(terms, labels_B, term_freq)

# -----------------------------
# METHOD C: NMF topics on TF-IDF (no dense needed for fit)
# Assign each term to topic with highest loading in H.
# Evaluate using topic-space term vectors (H^T normalized).
# -----------------------------
nmf = NMF(n_components=K, init="nndsvd", random_state=42, max_iter=400)
W = nmf.fit_transform(X)       # X can stay SPARSE here
H = nmf.components_            # (K, n_terms)
labels_C = H.argmax(axis=0)
T_topic = normalize(H.T, norm="l2", axis=1)  # dense (n_terms, K)
nC, kC, sC, intraC, interC = evaluate_labels(T_topic, labels_C)
map_C = build_mapping(terms, labels_C, term_freq)

# -----------------------------
# SUMMARY TABLE
# -----------------------------
summary = pd.DataFrame([
    {
        "method": "Agglomerative (cosine, average)",
        "space":  "TF-IDF term vectors (dense)",
        "params": f"K={K}",
        "n_terms": nA, "n_clusters": kA,
        "pct_singletons": round(sA, 4),
        "mean_intra_sim": round(intraA, 4),
        "mean_inter_sim": round(interA, 4),
    },
    {
        "method": "KMeans (euclidean)",
        "space":  "TF-IDF term vectors (dense)",
        "params": f"K={K}",
        "n_terms": nB, "n_clusters": kB,
        "pct_singletons": round(sB, 4),
        "mean_intra_sim": round(intraB, 4),
        "mean_inter_sim": round(interB, 4),
    },
    {
        "method": "NMF topics",
        "space":  "Topic loadings Hᵀ (dense)",
        "params": f"K={K}",
        "n_terms": nC, "n_clusters": kC,
        "pct_singletons": round(sC, 4),
        "mean_intra_sim": round(intraC, 4),
        "mean_inter_sim": round(interC, 4),
    },
], columns=["method","space","params","n_terms","n_clusters","pct_singletons","mean_intra_sim","mean_inter_sim"])

print("\n==== Term Grouping Summary (no tuning) ====\n")
display(summary)

# -----------------------------
# Quick previews for sanity-checking
# -----------------------------
preview_mapping("Agglomerative", map_A, n=SHOW_PREVIEW)
preview_mapping("KMeans",        map_B, n=SHOW_PREVIEW)
preview_mapping("NMF topics",    map_C, n=SHOW_PREVIEW)

# -----------------------------
# Tips:
# - If you hit memory limits on .toarray(), lower MAX_FEATURES (e.g., 8k–12k).
# - Prefer the method with higher mean_intra_sim, lower mean_inter_sim, and fewer singletons.
# - NMF often yields crisp, theme-based groups even when clustering looks noisy.
# -----------------------------