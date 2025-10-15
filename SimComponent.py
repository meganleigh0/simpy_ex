# ============================================================
# Synonym clustering baselines + summary table (one cell)
# ============================================================

# -----------------------------
# CONFIG — tweak as needed
# -----------------------------
DATA_PATH = "assets/reliability_data_test.json"
TEXT_COL  = "trimmed"

SPACY_MODEL = "en_core_web_md"      # try "en_core_web_lg" if you have it
GOOD_POS    = {"NOUN", "ADJ", "VERB"}
INCLUDE_NOUN_CHUNKS = True          # include short multi-word terms
CHUNK_MIN_LEN = 2
CHUNK_MAX_LEN = 4
MIN_FREQ = 5                         # drop very rare terms
LOWERCASE = True

BATCH_SIZE = 500
N_PROCESS  = -1

# Clustering defaults (no tuning sweep—just sensible baselines)
AGG_DISTANCE_THRESHOLD = 0.20       # Agglomerative distance threshold (cosine)
HDBSCAN_MIN_CLUSTER_SIZE = 5        # HDBSCAN baseline
HDBSCAN_MIN_SAMPLES     = 1

# Preview controls
SHOW_PREVIEW = 30                    # number of heads to preview per method (set 0 to skip)

# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import json, math, warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import spacy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Optional libs (will skip related methods if missing)
_have_sbert = False
_have_hdb   = False
try:
    from sentence_transformers import SentenceTransformer
    _have_sbert = True
except Exception:
    pass
try:
    import hdbscan
    _have_hdb = True
except Exception:
    pass

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
df = pd.read_json(DATA_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not in df. Available: {list(df.columns)}")
texts = df[TEXT_COL].astype(str).fillna("")
print(f"Loaded df: {df.shape}, using column '{TEXT_COL}'. Example rows:\n")
display(df[[TEXT_COL]].head(3))

# -------------------------------------------------------
# spaCy pipeline (parser kept if we want noun chunks)
# -------------------------------------------------------
disable = ["ner"]
if not INCLUDE_NOUN_CHUNKS:
    disable.append("parser")

try:
    nlp = spacy.load(SPACY_MODEL, disable=disable)
except Exception as e:
    raise RuntimeError(
        f"Could not load spaCy model '{SPACY_MODEL}'. "
        f"Install with: python -m spacy download {SPACY_MODEL}"
    ) from e

# -------------------------------------------------------
# Text -> candidate terms (lemmas + optional noun chunks)
# -------------------------------------------------------
def iter_terms(text_iterable):
    """Yield normalized single-token lemmas and (optionally) short noun-chunk terms."""
    for doc in nlp.pipe((str(t) for t in text_iterable),
                        batch_size=BATCH_SIZE, n_process=N_PROCESS):
        # single-token terms
        for tok in doc:
            if tok.is_alpha and not tok.is_stop and tok.pos_ in GOOD_POS:
                term = tok.lemma_.lower() if LOWERCASE else tok.lemma_
                if term:
                    yield term

        # noun-chunk terms ("fuel_pump", "engine_mount")
        if INCLUDE_NOUN_CHUNKS and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                toks = [t.lemma_.lower() if LOWERCASE else t.lemma_
                        for t in chunk if t.is_alpha and not t.is_stop]
                if CHUNK_MIN_LEN <= len(toks) <= CHUNK_MAX_LEN:
                    yield "_".join(toks)

vocab_counter = Counter(iter_terms(texts))
candidates = [w for w, f in vocab_counter.items() if f >= MIN_FREQ]
print(f"\nUnique terms: {len(vocab_counter):,} | candidates (freq ≥ {MIN_FREQ}): {len(candidates):,}")
if not candidates:
    raise ValueError("No candidates passed MIN_FREQ—lower MIN_FREQ or check the text column.")

# -------------------------------------------------------
# Embeddings
# -------------------------------------------------------
def embed_spacy(terms):
    """spaCy vectors for single tokens; for chunks, average token vectors."""
    dim = nlp.vocab.vectors_length
    out = []
    for t in terms:
        if "_" in t:  # multi-word chunk
            doc = nlp(t.replace("_", " "))
            vecs = np.array([tok.vector for tok in doc if tok.has_vector and tok.is_alpha and not tok.is_stop])
            v = vecs.mean(axis=0) if vecs.size else np.zeros(dim)
        else:
            lex = nlp.vocab[t]
            v = lex.vector if lex.has_vector else np.zeros(dim)
        out.append(v)
    arr = np.vstack(out)
    # drop OOV rows
    keep = np.linalg.norm(arr, axis=1) > 0
    arr  = arr[keep]
    kept_terms = [w for w, k in zip(terms, keep) if k]
    # L2 normalize (cosine-friendly)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return kept_terms, arr / norms

def embed_sbert(terms, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    emb = model.encode(terms, convert_to_numpy=True, normalize_embeddings=True)
    return terms, emb

# -------------------------------------------------------
# Clustering helpers
# -------------------------------------------------------
def run_agglomerative(emb, threshold=0.20):
    model = AgglomerativeClustering(metric="cosine", linkage="average",
                                    distance_threshold=float(threshold), n_clusters=None)
    labels = model.fit(emb).labels_
    return labels

def run_hdbscan(emb, min_cluster_size=5, min_samples=1):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean',   # on normalized vectors ~ cosine
                                cluster_selection_method='eom')
    labels = clusterer.fit_predict(emb)
    return labels

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------
def cluster_metrics(emb, labels):
    """
    Returns:
      n_terms, n_clusters, pct_singletons, mean_intra_sim, mean_inter_sim
    """
    emb = np.asarray(emb)
    labels = np.asarray(labels)
    n_terms = len(labels)
    unique = [l for l in np.unique(labels) if l != -1]  # ignore noise=-1 for HDBSCAN
    n_clusters = len(unique)
    counts = [np.sum(labels == l) for l in unique]
    pct_singletons = (sum(1 for c in counts if c == 1) / max(1, len(labels))) if len(labels) else 0.0

    # Similarities (sampled if very large to keep it fast)
    sims = cosine_similarity(emb)
    intra_vals, inter_vals = [], []
    idx_by_lab = {l: np.where(labels == l)[0] for l in unique}

    for l, idx in idx_by_lab.items():
        if len(idx) > 1:
            block = sims[np.ix_(idx, idx)]
            # exclude diagonal
            m = (np.sum(block) - np.trace(block)) / (len(idx) * (len(idx) - 1))
            intra_vals.append(m)

    labs = list(idx_by_lab.keys())
    for i in range(len(labs)):
        for j in range(i+1, len(labs)):
            a, b = idx_by_lab[labs[i]], idx_by_lab[labs[j]]
            inter_vals.append(sims[np.ix_(a, b)].mean())

    mean_intra = float(np.mean(intra_vals)) if intra_vals else float("nan")
    mean_inter = float(np.mean(inter_vals)) if inter_vals else float("nan")
    return n_terms, n_clusters, pct_singletons, mean_intra, mean_inter

def build_mapping(terms, labels, counter):
    """Pick head = most frequent; others are synonyms."""
    groups = defaultdict(list)
    for t, l in zip(terms, labels):
        if l == -1:  # ignore HDBSCAN noise
            continue
        groups[int(l)].append(t)

    mapping = {}
    for l, ts in groups.items():
        head = max(ts, key=lambda t: (counter[t], -len(t)))
        syns = [t for t in ts if t != head]
        mapping[head] = sorted(syns, key=lambda s: (-counter[s], len(s)))
    return mapping

def preview_mapping(name, mapping, n=30):
    if n <= 0 or not mapping:
        return
    print(f"\n[{name}] preview of {min(n, len(mapping))} heads (sorted by #synonyms):")
    for head, syns in sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:n]:
        line = ", ".join(syns) if syns else "(singleton)"
        print(f"{head:22s} : {line}")

# -------------------------------------------------------
# RUN METHODS
# -------------------------------------------------------
rows = []
method_previews = []

# --- Method A: spaCy embeddings + Agglomerative ---
terms_spaCy, emb_spaCy = embed_spacy(candidates)
labels_A = run_agglomerative(emb_spaCy, threshold=AGG_DISTANCE_THRESHOLD)
n_terms, n_clusters, pct_singletons, m_intra, m_inter = cluster_metrics(emb_spaCy, labels_A)
map_A = build_mapping(terms_spaCy, labels_A, vocab_counter)
rows.append({
    "method": "Agglomerative (cosine)",
    "embeddings": SPACY_MODEL,
    "params": f"thr={AGG_DISTANCE_THRESHOLD}",
    "n_terms": n_terms,
    "n_clusters": n_clusters,
    "pct_singletons": round(pct_singletons, 4),
    "mean_intra_sim": round(m_intra, 4),
    "mean_inter_sim": round(m_inter, 4),
})
method_previews.append(("Agg+spaCy", map_A))

# --- Method B: SBERT embeddings + Agglomerative (if available) ---
if _have_sbert:
    terms_sbert, emb_sbert = embed_sbert(candidates)  # normalized by model
    labels_B = run_agglomerative(emb_sbert, threshold=AGG_DISTANCE_THRESHOLD)
    n_terms, n_clusters, pct_singletons, m_intra, m_inter = cluster_metrics(emb_sbert, labels_B)
    map_B = build_mapping(terms_sbert, labels_B, vocab_counter)
    rows.append({
        "method": "Agglomerative (cosine)",
        "embeddings": "SBERT (all-MiniLM-L6-v2)",
        "params": f"thr={AGG_DISTANCE_THRESHOLD}",
        "n_terms": n_terms,
        "n_clusters": n_clusters,
        "pct_singletons": round(pct_singletons, 4),
        "mean_intra_sim": round(m_intra, 4),
        "mean_inter_sim": round(m_inter, 4),
    })
    method_previews.append(("Agg+SBERT", map_B))
else:
    warnings.warn("sentence-transformers not installed: skipping Agg+SBERT.")

# --- Method C: SBERT embeddings + HDBSCAN (if available) ---
if _have_sbert and _have_hdb:
    labels_C = run_hdbscan(emb_sbert, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                           min_samples=HDBSCAN_MIN_SAMPLES)
    n_terms, n_clusters, pct_singletons, m_intra, m_inter = cluster_metrics(emb_sbert, labels_C)
    map_C = build_mapping(terms_sbert, labels_C, vocab_counter)
    rows.append({
        "method": "HDBSCAN (euclidean on normalized vecs)",
        "embeddings": "SBERT (all-MiniLM-L6-v2)",
        "params": f"min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}",
        "n_terms": n_terms,
        "n_clusters": n_clusters,
        "pct_singletons": round(pct_singletons, 4),
        "mean_intra_sim": round(m_intra, 4),
        "mean_inter_sim": round(m_inter, 4),
    })
    method_previews.append(("HDBSCAN+SBERT", map_C))
elif _have_sbert and not _have_hdb:
    warnings.warn("hdbscan not installed: skipping HDBSCAN+SBERT.")
# -------------------------------------------------------
# SUMMARY TABLE
# -------------------------------------------------------
summary = pd.DataFrame(rows, columns=[
    "method","embeddings","params","n_terms","n_clusters","pct_singletons","mean_intra_sim","mean_inter_sim"
])
print("\n==== Synonym Clustering Summary (no tuning) ====\n")
display(summary)

# -------------------------------------------------------
# Optional: quick previews of synonym maps per method
# -------------------------------------------------------
for name, mapping in method_previews:
    preview_mapping(name, mapping, n=SHOW_PREVIEW)

# Tip for picking a winner quickly:
# - Prefer higher mean_intra_sim, lower mean_inter_sim, and lower pct_singletons.
# - Then eyeball the previews above for sanity.
