# ===========================================
# Synonym Clustering (one cell, simple + well-noted)
# ===========================================
# WHAT THIS DOES
# 1) Reads your dataframe with a text column (df["trimmed"]).
# 2) Tokenizes + lemmatizes with spaCy, filters by POS, and (optionally) extracts noun chunks.
# 3) Builds a vocabulary of candidate terms with min frequency.
# 4) Embeds the terms with either:
#       A) Sentence-Transformers (SBERT) if installed (preferred), or
#       B) spaCy vectors (fallback).
# 5) Clusters terms by meaning using:
#       - AgglomerativeClustering (cosine, distance threshold)  [always runs]
#       - HDBSCAN (if installed)                                [optional]
# 6) Picks a representative "head" for each cluster (highest frequency), maps the rest as synonyms.
# 7) Prints helpful summaries, returns a tidy cluster DataFrame and a synonym_map dict.
# 8) (Optional) Saves the synonym_map to JSON and shows how to normalize text using the map.

# -----------------------------
# CONFIG (tweak these easily)
# -----------------------------
TEXT_COL = "trimmed"            # your text column
DATA_PATH = "assets/reliability_data_test.json"   # your JSON path
SPACY_MODEL = "en_core_web_md"  # try "en_core_web_lg" if available
GOOD_POS = {"NOUN", "ADJ", "VERB"}   # parts of speech to include
INCLUDE_NOUN_CHUNKS = True           # capture short multi-word terms (e.g., "fuel_pump")
CHUNK_MIN_LEN = 2                    # tokens per chunk (min)
CHUNK_MAX_LEN = 4                    # tokens per chunk (max)
MIN_FREQ = 5                         # minimum frequency to keep a term
BATCH_SIZE = 500
N_PROCESS = -1                       # use all cores
LOWERCASE = True

# Agglomerative settings
AGG_DISTANCE_THRESHOLD = 0.18        # main value you’ll tune
AGG_SWEEP = True                     # set True to scan thresholds quickly
AGG_SWEEP_RANGE = (0.12, 0.30, 10)   # start, end, num_points

# HDBSCAN (optional) settings (runs only if hdbscan is installed)
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 1

# Output settings
SHOW_PREVIEW_CLUSTERS = 60           # how many cluster lines to print as a preview
SAVE_JSON = False
JSON_PATH = "synonym_map.json"

# -------------------------------------------------------
# IMPORTS
# -------------------------------------------------------
import os, json, math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import spacy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

# Try sentence-transformers (preferred for semantics); fall back to spaCy vectors
USE_SBERT = False
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer(SBERT_MODEL_NAME)
    USE_SBERT = True
except Exception:
    USE_SBERT = False  # will fall back gracefully

# Try HDBSCAN (optional second method)
HAVE_HDBSCAN = False
try:
    import hdbscan
    HAVE_HDBSCAN = True
except Exception:
    HAVE_HDBSCAN = False

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
df = pd.read_json(DATA_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Column '{TEXT_COL}' not in df. Available: {list(df.columns)}")
texts = df[TEXT_COL].astype(str).fillna("")
print(f"Loaded df: {df.shape}, using text column: '{TEXT_COL}'")
print(df[[TEXT_COL]].head(3))

# -------------------------------------------------------
# SPACY PIPELINE
# -------------------------------------------------------
# Note: we disable parser/ner for speed; if you want noun chunks, we DO need the parser.
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
# TEXT -> TERMS (lemmas + optional noun chunks)
# -------------------------------------------------------
def iter_terms(text_iterable):
    """
    Yields normalized terms per document:
      - single-token lemmas (POS in GOOD_POS, alpha, not stop)
      - optional noun chunks (joined with '_', length-controlled)
    """
    for doc in nlp.pipe((str(t) for t in text_iterable),
                        batch_size=BATCH_SIZE, n_process=N_PROCESS):
        # token-level
        for tok in doc:
            if tok.is_alpha and not tok.is_stop and tok.pos_ in GOOD_POS:
                term = tok.lemma_.lower() if LOWERCASE else tok.lemma_
                if term:
                    yield term

        # noun-chunk-level
        if INCLUDE_NOUN_CHUNKS and doc.has_annotation("DEP"):
            for chunk in doc.noun_chunks:
                toks = [t.lemma_.lower() if LOWERCASE else t.lemma_
                        for t in chunk if t.is_alpha and not t.is_stop]
                if CHUNK_MIN_LEN <= len(toks) <= CHUNK_MAX_LEN:
                    yield "_".join(toks)

# -------------------------------------------------------
# BUILD VOCAB (min freq)
# -------------------------------------------------------
vocab_counter = Counter(iter_terms(texts))
candidates = [w for w, f in vocab_counter.items() if f >= MIN_FREQ]
print(f"\nUnique terms total: {len(vocab_counter):,}")
print(f"Candidates with freq >= {MIN_FREQ}: {len(candidates):,}")
if not candidates:
    raise ValueError("No candidates passed MIN_FREQ. Lower MIN_FREQ or check data.")

# -------------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------------
def embed_terms_spacy(terms):
    """Vector for single tokens from lexeme; for multi-word chunks, average token vectors."""
    dim = nlp.vocab.vectors_length
    out = []
    for t in terms:
        if "_" in t:
            doc = nlp(t.replace("_", " "))
            vecs = np.array([tok.vector for tok in doc if tok.has_vector and tok.is_alpha and not tok.is_stop])
            v = vecs.mean(axis=0) if vecs.size else np.zeros(dim)
        else:
            lex = nlp.vocab[t]
            v = lex.vector if lex.has_vector else np.zeros(dim)
        out.append(v)
    arr = np.vstack(out)
    return arr

def l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

if USE_SBERT:
    print("\nUsing SBERT embeddings:", SBERT_MODEL_NAME)
    embeddings = _sbert.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
else:
    print("\nUsing spaCy embeddings:", SPACY_MODEL)
    E = embed_terms_spacy(candidates)
    keep = np.linalg.norm(E, axis=1) > 0
    candidates = [w for w, k in zip(candidates, keep) if k]
    embeddings = l2_normalize(E[keep])
    print(f"Dropped OOV/no-vector terms: {int((~keep).sum())}")

print("Embedding shape:", embeddings.shape)

# -------------------------------------------------------
# METHOD 1: AGGLOMERATIVE (COSINE) + THRESHOLD SWEEP
# -------------------------------------------------------
def agglomerative_cluster(emb, threshold):
    model = AgglomerativeClustering(
        metric="cosine", linkage="average",
        distance_threshold=float(threshold), n_clusters=None
    )
    labels = model.fit(emb).labels_
    return labels

if AGG_SWEEP:
    import numpy as np
    lo, hi, num = AGG_SWEEP_RANGE
    thr_list = np.linspace(lo, hi, num)
    print("\n[Threshold sweep] threshold → (#clusters, %singletons)")
    for thr in thr_list:
        labs = agglomerative_cluster(embeddings, thr)
        counts = Counter(labs).values()
        singletons = sum(1 for s in counts if s == 1) / len(labs)
        print(f"  {thr:0.3f} → ({len(set(labs)):,}, {singletons:0.1%})")

labels_agg = agglomerative_cluster(embeddings, AGG_DISTANCE_THRESHOLD)
print(f"\n[Agglomerative] threshold={AGG_DISTANCE_THRESHOLD} → clusters={len(set(labels_agg)):,}")

# -------------------------------------------------------
# METHOD 2 (optional): HDBSCAN (only if installed)
# -------------------------------------------------------
labels_hdb = None
if HAVE_HDBSCAN:
    print("\n[HDBSCAN] available — running...")
    # Metric='euclidean' on normalized vectors ≈ cosine. HDBSCAN can be very effective for synonyms.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                min_samples=HDBSCAN_MIN_SAMPLES,
                                metric='euclidean',
                                cluster_selection_method='eom')
    labels_hdb = clusterer.fit_predict(embeddings)
    n_lbl = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
    noise = float(np.mean(labels_hdb == -1)) if labels_hdb is not None else 0.0
    print(f"[HDBSCAN] clusters={n_lbl:,}, noise={noise:0.1%}")
else:
    print("\n[HDBSCAN] not installed — skipping this method (pip install hdbscan to try it).")

# -------------------------------------------------------
# BUILD MAPPINGS (for both methods) + TIDY TABLES
# -------------------------------------------------------
def build_mapping(cands, labels, counter):
    """Pick 'head' as most frequent term in cluster; others become its synonyms."""
    groups = defaultdict(list)
    for tok, lab in zip(cands, labels):
        groups[int(lab)].append(tok)

    rows = []
    mapping = {}
    for lab, terms in groups.items():
        if lab == -1:  # HDBSCAN noise cluster (ignore in mapping)
            continue
        if len(terms) == 1:
            head = terms[0]
            syns = []
        else:
            head = max(terms, key=lambda t: (counter[t], -len(t)))
            syns = [t for t in terms if t != head]
        mapping[head] = sorted(syns, key=lambda s: (-counter[s], len(s)))
        for t in terms:
            rows.append({
                "algo": "agg" if labels is labels_agg else "hdb",
                "cluster_id": lab,
                "head": head,
                "term": t,
                "is_head": t == head,
                "freq": int(counter[t]),
            })
    df_out = pd.DataFrame(rows).sort_values(["head", "is_head", "freq"], ascending=[True, False, False])
    return mapping, df_out

syn_map_agg, clusters_agg = build_mapping(candidates, labels_agg, vocab_counter)
print(f"\n[Agg] mapped heads: {len(syn_map_agg):,}  (some heads may have 0 synonyms if singleton)")

syn_map_hdb, clusters_hdb = ({}, pd.DataFrame())
if labels_hdb is not None:
    syn_map_hdb, clusters_hdb = build_mapping(candidates, labels_hdb, vocab_counter)
    print(f"[HDB] mapped heads: {len(syn_map_hdb):,}")

# -------------------------------------------------------
# PREVIEW: show a few clusters with their synonyms
# -------------------------------------------------------
def preview(mapping, n=40, title="[preview]"):
    items = sorted(mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    print(f"\n{title} Showing {min(n, len(items))} of {len(items)} heads (sorted by #synonyms):")
    for head, syns in items[:n]:
        if syns:
            print(f"{head:20s} : {', '.join(syns)}")
        else:
            print(f"{head:20s} : (singleton)")

preview(syn_map_agg, SHOW_PREVIEW_CLUSTERS, title="[Agglomerative preview]")
if syn_map_hdb:
    preview(syn_map_hdb, SHOW_PREVIEW_CLUSTERS, title="[HDBSCAN preview]")

# -------------------------------------------------------
# OPTIONAL: Save JSON (set SAVE_JSON=True)
# -------------------------------------------------------
if SAVE_JSON:
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(syn_map_hdb or syn_map_agg, f, indent=2, ensure_ascii=False)
    print(f"\nSaved synonym_map → {JSON_PATH}")

# -------------------------------------------------------
# HOW TO USE THE MAP TO NORMALIZE TEXT
# -------------------------------------------------------
# This function replaces any matched term with its head.
# Note: It operates on lemmas; for chunks, it uses the underscore form.
def normalize_text_with_map(text, mapping):
    rep = {}
    for head, syns in mapping.items():
        for s in syns:
            rep[s] = head
        # also allow head to map to itself (idempotent)
        rep[head] = head

    # Build a fast lookup set
    keys = set(rep.keys())

    doc = nlp(text)
    out_tokens = []
    i = 0
    while i < len(doc):
        # try chunk match first (longer wins): 4→2 tokens
        matched = False
        if INCLUDE_NOUN_CHUNKS and doc.has_annotation("DEP"):
            for L in range(min(CHUNK_MAX_LEN, len(doc)-i), CHUNK_MIN_LEN-1, -1):
                toks = [t for t in doc[i:i+L] if t.is_alpha and not t.is_stop]
                if len(toks) == L:
                    key = "_".join((t.lemma_.lower() if LOWERCASE else t.lemma_) for t in toks)
                    if key in keys:
                        out_tokens.append(rep[key].replace("_", " "))
                        i += L
                        matched = True
                        break
        if matched:
            continue

        # single-token fallback
        t = doc[i]
        if t.is_alpha:
            lem = t.lemma_.lower() if LOWERCASE else t.lemma_
            out_tokens.append(rep.get(lem, t.text))
        else:
            out_tokens.append(t.text)
        i += 1

    return spacy.tokens.doc.Doc(nlp.vocab, words=out_tokens).text

# Example: normalize first 3 rows with the preferred map (HDB if available else Agg)
PREFERRED_MAP = syn_map_hdb if syn_map_hdb else syn_map_agg
print("\n[Normalize demo] First 3 rows before/after:")
for i in range(min(3, len(texts))):
    original = texts.iloc[i]
    normalized = normalize_text_with_map(original, PREFERRED_MAP)
    print(f"\n--- Row {i} ---")
    print("ORIG:", original[:240].replace("\n", " "))
    print("NORM:", normalized[:240].replace("\n", " "))

# -------------------------------------------------------
# WHAT TO TUNE / HOW TO APPROACH
# -------------------------------------------------------
# 1) If clusters look too loose/tight with Agglomerative:
#    - Run the threshold sweep above and pick a value that gives
#      a reasonable #clusters and low %singletons (eyeball a few).
#    - Typical good values for SBERT: 0.15–0.30 (lower = tighter).
#
# 2) Prefer SBERT over spaCy vectors if possible:
#    - pip install sentence-transformers
#    - These embeddings usually capture synonymy better than static vectors.
#
# 3) Try HDBSCAN if you can install it:
#    - pip install hdbscan
#    - Pros: finds variable-density clusters and leaves outliers as noise (-1).
#    - Tune min_cluster_size (3–8) and min_samples (1–5).
#
# 4) Include noun chunks if phrases matter (e.g., "engine mount"):
#    - Keep CHUNK_MIN_LEN/CHUNK_MAX_LEN small (2–3) to avoid noise.
#
# 5) Use corpus frequency (MIN_FREQ) to drop rare junk terms and to choose heads.
#
# 6) Inspect with the previews and clusters_agg / clusters_hdb DataFrames:
#    - clusters_agg.head()  # tidy table: head, term, freq, cluster_id
#    - Use this to audit before applying normalization to all text.
#
# 7) Export the map (SAVE_JSON=True) and use normalize_text_with_map(...) to
#    standardize incoming text (e.g., for downstream grouping, rules, or ML).
#
# 8) If you need even better quality:
#    - Swap in a stronger SBERT like "all-mpnet-base-v2" (slower but better).
#    - Add domain-specific phrases via a curated synonym seed list and
#      merge clusters that share a seed head.
#
# Happy clustering!