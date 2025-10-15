# ============================================================
# Synonym-like term grouping with TF-IDF + (Agglomerative, KMeans, NMF)
# Dependencies: pandas, numpy, scikit-learn
# ============================================================

# -----------------------------
# CONFIG — tweak as needed
# -----------------------------
DATA_PATH   = "assets/reliability_data_test.json"
TEXT_COL    = "trimmed"           # your text column name
MIN_DF      = 5                   # drop rare terms (min doc frequency)
MAX_FEATURES= 25000               # cap vocab size
NGRAM_RANGE = (1, 2)              # unigrams + bigrams capture short phrases
LOWERCASE   = True
STOP_WORDS  = "english"           # basic stopword removal

# Cluster/topic count heuristic (no tuning step):
# k ≈ sqrt(#terms/2), clipped to [8, 80].
K_LOWER, K_UPPER = 8, 80

# Preview controls
SHOW_PREVIEW = 30                 # lines per method preview (set 0 to skip)

# -----------------------------
# Imports
# -----------------------------
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

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
# Build TF-IDF (docs x terms) and raw counts to get term frequency
# -----------------------------
tfidf = TfidfVectorizer(
    lowercase=LOWERCASE,
    stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
    norm="l2",           # cosine-friendly
)
X = tfidf.fit_transform(docs)          # shape: (n_docs, n_terms)
terms = np.array(tfidf.get_feature_names_out())
n_terms = len(terms)
if n_terms == 0:
    raise ValueError("No terms survived vectorization. Try lowering MIN_DF or adjusting preprocessing.")
print(f"TF-IDF shape: {X.shape}  |  vocab size (terms): {n_terms}")

# For term frequency (to choose cluster “heads”), use raw counts:
cv = CountVectorizer(
    lowercase=LOWERCASE,
    stop_words=STOP_WORDS,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
)
C = cv.fit_transform(docs)             # (n_docs, n_terms2) — may differ if analyzer differs
# Align counts to TF-IDF terms (shared intersection)
cv_terms = np.array(cv.get_feature_names_out())
common, tfidf_idx, cv_idx = np.intersect1d(terms, cv_terms, return_indices=True)
if len(common) < n_terms:
    # reduce TF-IDF matrix to common terms to keep things aligned for frequency lookup
    X = X[:, tfidf_idx]
    terms = terms[tfidf_idx]
    n_terms = len(terms)
    print(f"Aligned to common vocab of size {n_terms} for frequency lookup.")
term_freq = np.asarray(C[:, cv_idx].sum(axis=0)).ravel()  # frequency per term in docs order
# Normalize term vectors are columns of X.T (terms x docs)
T = normalize(X.T, norm="l2", axis=1)                     # (n_terms, n_docs) cosine-friendly

# -----------------------------
# Heuristic for number of clusters/topics
# -----------------------------
def pick_k(n_terms):
    k = int(round(math.sqrt(max(2, n_terms)/2.0)))
    return max(K_LOWER, min(K_UPPER, k))

K = pick_k(n_terms)
print(f"Chosen k (clusters/topics) = {K} (heuristic)")

# -----------------------------
# Metrics: cohesion/separation and singletons
# -----------------------------
def evaluate_labels(term_matrix, labels):
    """
    term_matrix: (n_terms, dim) L2-normalized vectors (we use terms x docs TF-IDF)
    labels: cluster/topic label for each term (length n_terms), noise not expected here
    Returns:
      n_terms, n_clusters, pct_singletons, mean_intra_sim, mean_inter_sim
    """
    labels = np.asarray(labels)
    n_terms = len(labels)
    uniq = np.unique(labels)
    n_clusters = len(uniq)

    # Singletons
    counts = [np.sum(labels == l) for l in uniq]
    pct_singletons = (sum(1 for c in counts if c == 1) / max(1, n_terms))

    # Cosine similarity matrix on term vectors
    # For large vocab, this can be big; to stay safe, we compute blockwise averages instead.
    # But for clarity/compactness, compute full matrix once if n_terms is manageable.
    if n_terms <= 6000:
        sims = cosine_similarity(term_matrix)  # (n_terms, n_terms)
        # Intra-cluster: average pairwise similarity within each cluster (excluding self)
        intra_vals = []
        idx_by_lab = {l: np.where(labels == l)[0] for l in uniq}
        for l, idx in idx_by_lab.items():
            if len(idx) > 1:
                block = sims[np.ix_(idx, idx)]
                m = (block.sum() - np.trace(block)) / (len(idx) * (len(idx) - 1))
                intra_vals.append(m)
        # Inter-cluster: average similarity across different clusters
        inter_vals = []
        labs = list(idx_by_lab.keys())
        for i in range(len(labs)):
            for j in range(i+1, len(labs)):
                a, b = idx_by_lab[labs[i]], idx_by_lab[labs[j]]
                inter_vals.append(sims[np.ix_(a, b)].mean())
        mean_intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
        mean_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    else:
        # Fallback for very large term sets: sample pairs
        rng = np.random.default_rng(42)
        idx = np.arange(n_terms)
        intra_vals, inter_vals = [], []
        for l in np.unique(labels):
            members = idx[labels == l]
            if len(members) > 1:
                a = rng.choice(members, size=min(50, len(members)), replace=False)
                b = rng.choice(members, size=min(50, len(members)), replace=False)
                intra_vals.append((term_matrix[a] @ term_matrix[b].T).mean())
        # sample inter pairs from 200 random terms
        sample = rng.choice(idx, size=min(200, n_terms), replace=False)
        S = term_matrix[sample] @ term_matrix[sample].T
        mean_inter = float((S.sum() - np.trace(S)) / (S.size - len(sample))) if S.size>0 else float("nan")
        mean_intra = float(np.mean(intra_vals)) if intra_vals else float("nan")

    return n_terms, n_clusters, pct_singletons, mean_intra, mean_inter

def build_mapping(terms, labels, freqs):
    """Group terms by label; 'head' is most frequent term; others are synonyms."""
    groups = defaultdict(list)
    for t, l in zip(terms, labels):
        groups[int(l)].append(t)

    mapping = {}
    for l, ts in groups.items():
        # choose head by highest frequency, tie-break by shortest
        head = max(ts, key=lambda t: (freqs[np.where(terms == t)[0][0]], -len(t)))
        syns = [t for t in ts if t != head]
        mapping[head] = sorted(syns, key=lambda s: (-freqs[np.where(terms == s)[0][0]], len(s)))
    return mapping

def preview_mapping(name, mapping, n=30):
    if n <= 0 or not mapping:
        return
    print(f"\n[{name}] preview of {min(n, len(mapping))} heads (sorted by #synonyms):")
    for head, syns in sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n]:
        print(f"{head:22s} : {', '.join(syns) if syns else '(singleton)'}")

# -----------------------------
# METHOD A: Agglomerative (cosine) on term vectors
# -----------------------------
agg = AgglomerativeClustering(
    n_clusters=K, linkage="average", metric="cosine"
)
labels_A = agg.fit_predict(T)  # T is (n_terms, n_docs)
nA, kA, sA, intraA, interA = evaluate_labels(T, labels_A)
map_A = build_mapping(terms, labels_A, term_freq)

# -----------------------------
# METHOD B: KMeans on term vectors
# -----------------------------
kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
labels_B = kmeans.fit_predict(T)     # kmeans uses euclidean; T is L2-normalized
nB, kB, sB, intraB, interB = evaluate_labels(T, labels_B)
map_B = build_mapping(terms, labels_B, term_freq)

# -----------------------------
# METHOD C: NMF topics on TF-IDF (docs x terms)
# -----------------------------
# NMF factorizes X ≈ W (docs x K) @ H (K x terms); we assign each term to the topic with highest loading in H
nmf = NMF(n_components=K, init="nndsvd", random_state=42, max_iter=400)
W = nmf.fit_transform(X)         # (n_docs, K)
H = nmf.components_              # (K, n_terms)
labels_C = H.argmax(axis=0)      # topic id per term
# For NMF evaluation, use term vectors in topic space (columns of H^T normalized)
T_topic = normalize(H.T, norm="l2", axis=1)  # (n_terms, K)
nC, kC, sC, intraC, interC = evaluate_labels(T_topic, labels_C)
map_C = build_mapping(terms, labels_C, term_freq)

# -----------------------------
# SUMMARY TABLE
# -----------------------------
summary = pd.DataFrame([
    {
        "method": "Agglomerative (cosine, average)",
        "space":  "TF-IDF term vectors",
        "params": f"K={K}",
        "n_terms": nA, "n_clusters": kA,
        "pct_singletons": round(sA, 4),
        "mean_intra_sim": round(intraA, 4),
        "mean_inter_sim": round(interA, 4),
    },
    {
        "method": "KMeans (euclidean)",
        "space":  "TF-IDF term vectors",
        "params": f"K={K}",
        "n_terms": nB, "n_clusters": kB,
        "pct_singletons": round(sB, 4),
        "mean_intra_sim": round(intraB, 4),
        "mean_inter_sim": round(interB, 4),
    },
    {
        "method": "NMF topics",
        "space":  "Topic loadings (Hᵀ)",
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
# Quick previews so you can sanity-check
# -----------------------------
preview_mapping("Agglomerative", map_A, n=SHOW_PREVIEW)
preview_mapping("KMeans",        map_B, n=SHOW_PREVIEW)
preview_mapping("NMF topics",    map_C, n=SHOW_PREVIEW)

# -----------------------------
# Notes on interpretation (short):
# - Prefer higher mean_intra_sim, lower mean_inter_sim, and a lower pct_singletons.
# - NMF can produce crisp, theme-based groups even when clustering looks noisy.
# - Agglomerative (cosine) often yields tight semantic families in TF-IDF space.
# - KMeans is fast and stable; good baseline to compare against.
# -----------------------------