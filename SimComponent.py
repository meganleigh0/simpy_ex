# --- Imports -----------------------------------------------------------------
import re, math, string, random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# --- Config (edit here) ------------------------------------------------------
SEED                = 42
CONTEXT_WINDOW      = 5           # symmetric window size around a token
MIN_WORD_LEN        = 3
MIN_WORD_FREQ       = 20          # keep words seen at least this many times
MAX_CONTEXTS_PER_WORD = 60        # cap contexts collected per unique word
MAX_FEATURES        = 30000       # tf-idf vocab limit
NGRAM_RANGE         = (1, 2)      # use bigrams too
TOKEN_PATTERN       = r"[a-zA-Z][a-zA-Z\-']+"
TOP_K_WORDS_PER_TOPIC = 40
MIN_MEMBERSHIP      = 0.18        # soft-membership threshold for words→topics
RELATIVE_KEEP       = 0.60        # keep topics >= this * max topic score for word

# NMF search space (we’ll pick best by coherence + stability)
N_TOPICS_GRID       = [15, 20, 25, 30]
NMF_MAX_ITER        = 600
ALPHA_W             = 0.0         # l1 on W (sparsity); can try 0.1 - 0.5
ALPHA_H             = 0.0         # l1 on H

# Optional: add domain stopwords here
CUSTOM_STOPS = {
    "verbatim","following","unit","assembly","removed","installed","warning",
    "lamp","system","reading","readings","switch","replaced","pressure"
}

random.seed(SEED)
np.random.seed(SEED)

# --- Helpers: text cleaning ---------------------------------------------------
STOP = {
    # english stop set (small, you can expand)
    "the","a","an","and","or","if","is","was","were","are","be","been","being",
    "to","of","for","in","on","at","by","it","this","that","with","as","from",
    "not","no","but","so","than","then","there","here","into","out","off",
}
STOP |= CUSTOM_STOPS

def sentence_split(text: str):
    # conservative sentence splitter (keeps abbreviations intact reasonably well)
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+|\n+", text or "") if s.strip()]

def tokenize(text: str):
    toks = re.findall(TOKEN_PATTERN, (text or "").lower())
    toks = [t.strip("-'") for t in toks]
    toks = [t for t in toks if len(t) >= MIN_WORD_LEN and t not in STOP]
    # very light lemmatization-ish endings (keeps domain terms intact)
    lem = []
    for t in toks:
        if t.endswith("ing") and len(t) > 5: t = t[:-3]
        elif t.endswith("ed") and len(t) > 4: t = t[:-2]
        elif t.endswith("s") and len(t) > 4 and not t.endswith("ss"): t = t[:-1]
        lem.append(t)
    return lem

# --- Step 1: Build full sentences per did ------------------------------------
# Expecting a frame with columns like: did, token, sentence_beg, place, sentence_end
# If you already have full_sentences, you can skip to Step 2.
def build_sentences(df_tokens: pd.DataFrame) -> pd.DataFrame:
    df = df_tokens.copy()
    df["token"] = df["token"].fillna("").astype(str)
    # stable ordering of tokens inside a sentence via (did, sentence_beg, place, sentence_end)
    sents = (
        df.sort_values(["did", "sentence_beg", "place", "sentence_end"])
          .groupby(["did","sentence_beg","sentence_end"])["token"]
          .agg(lambda x: " ".join(x))
          .reset_index()
    )
    full = (
        sents.groupby("did")["token"]
             .apply(lambda x: " ".join(x))
             .reset_index()
             .rename(columns={"token":"sentence"})
    )
    return full

# --- Step 2: Build context rows (word, did, context) -------------------------
def build_context_rows(df_full: pd.DataFrame):
    rows = []
    word_counts = Counter()

    # pass 1: count words to form vocab
    for did, text in df_full[["did","sentence"]].itertuples(index=False):
        for sent in sentence_split(text):
            for tok in tokenize(sent):
                word_counts[tok] += 1

    vocab_words = {w for w,c in word_counts.items() if c >= MIN_WORD_FREQ}
    per_word_added = Counter()

    # pass 2: collect contexts, cap per word
    for did, text in df_full[["did","sentence"]].itertuples(index=False):
        for sent in sentence_split(text):
            toks = tokenize(sent)
            if not toks: 
                continue
            for i, w in enumerate(toks):
                if w not in vocab_words: 
                    continue
                if per_word_added[w] >= MAX_CONTEXTS_PER_WORD:
                    continue
                lo = max(0, i - CONTEXT_WINDOW)
                hi = min(len(toks), i + CONTEXT_WINDOW + 1)
                ctx = [t for t in toks[lo:i] + toks[i+1:hi] if t not in STOP]
                if not ctx: 
                    continue
                rows.append((w, did, " ".join(ctx)))
                per_word_added[w] += 1

    ctx_df = pd.DataFrame(rows, columns=["word","did","context"])
    return ctx_df, sorted(vocab_words)

# --- Step 3: Vectorize contexts (TF-IDF + SVD denoising) ---------------------
def vectorize_contexts(ctx_df: pd.DataFrame):
    tfidf = TfidfVectorizer(
        stop_words=None,         # already filtered
        max_features=MAX_FEATURES,
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        ngram_range=NGRAM_RANGE,
        dtype=np.float32
    )
    X = tfidf.fit_transform(ctx_df["context"])
    # denoise w/ SVD (keeps long-range structure; safer clustering/nmf)
    # Keep enough components but not too many (rule of thumb: sqrt(#features))
    n_svd = min(256, max(50, int(math.sqrt(X.shape[1]))))
    svd = TruncatedSVD(n_components=n_svd, random_state=SEED)
    X_svd = svd.fit_transform(X)
    X_svd = normalize(X_svd)   # cosine geometry
    return X, X_svd, tfidf, svd

# --- Coherence & Stability utilities -----------------------------------------
def top_terms_from_H(H_row, feature_names, k=20):
    idx = np.argsort(H_row)[::-1][:k]
    return [feature_names[i] for i in idx]

def umass_coherence(top_terms, td_matrix, vocab_index):
    # u_mass approximation using term-document (contexts) counts
    eps = 1e-9
    scores = []
    for i in range(1, len(top_terms)):
        for j in range(i):
            wi, wj = top_terms[i], top_terms[j]
            ci = td_matrix[:, vocab_index.get(wi, -1)].sum() if wi in vocab_index else 0
            cij = 0
            if wi in vocab_index and wj in vocab_index:
                col_i = vocab_index[wi]; col_j = vocab_index[wj]
                cij = (td_matrix[:, col_i].multiply(td_matrix[:, col_j])>0).sum()
            if cij == 0 or ci == 0: 
                scores.append(-5.0)
            else:
                scores.append(np.log((cij + eps) / ci))
    return float(np.mean(scores)) if scores else -10.0

def model_quality(nmf_model, X_tfidf, feature_names):
    W = nmf_model.transform(X_tfidf)
    H = nmf_model.components_
    # coherence (u_mass) computed on sparse matrix
    vocab_index = {t:i for i,t in enumerate(feature_names)}
    coh = []
    for t in range(H.shape[0]):
        terms = top_terms_from_H(H[t], feature_names, k=15)
        coh.append(umass_coherence(terms, X_tfidf, vocab_index))
    coh = float(np.mean(coh))
    # stability: cosine similarity among topics vs a small noise resample
    # (quick + dirty stability proxy)
    rng = np.random.default_rng(SEED)
    mask = rng.choice(X_tfidf.shape[0], size=int(0.85*X_tfidf.shape[0]), replace=False)
    W2 = nmf_model.fit_transform(X_tfidf[mask])
    H2 = nmf_model.components_
    sim = cosine_similarity(H, H2).max(axis=1).mean()
    return {"coherence_umass": coh, "stability": float(sim)}

# --- Step 4: Fit NMF with a small search over topics -------------------------
def fit_best_nmf(X_tfidf, feature_names, topic_grid=N_TOPICS_GRID):
    results = []
    best = None
    for k in topic_grid:
        nmf = NMF(
            n_components=k,
            init="nndsvda",
            random_state=SEED,
            max_iter=NMF_MAX_ITER,
            beta_loss="kullback-leibler",
            solver="mu",
            alpha_W=ALPHA_W, l1_ratio=1.0,
            alpha_H=ALPHA_H
        )
        W = nmf.fit_transform(X_tfidf)
        q = model_quality(nmf, X_tfidf, feature_names)
        score = 0.6*q["coherence_umass"] + 0.4*q["stability"]
        results.append({"k":k, **q, "score":score, "model":nmf, "W":W})
        if (best is None) or (score > best["score"]):
            best = results[-1]
    return best, pd.DataFrame([{k:v for k,v in r.items() if k!="model" and k!="W"} for r in results])

# --- Step 5: Aggregate to word-level soft memberships ------------------------
def word_topic_memberships(ctx_df, W_doc_topic, n_topics):
    # sum context-topic weights per word, then normalize
    sums = defaultdict(lambda: np.zeros(n_topics, dtype=np.float32))
    counts = Counter()
    for (w, _did), row_w in zip(ctx_df[["word","did"]].itertuples(index=False, name=None), W_doc_topic):
        sums[w] += row_w
        counts[w] += 1
    words = []
    vecs = []
    for w, v in sums.items():
        words.append(w)
        vecs.append(v / max(counts[w], 1))
    M = np.vstack(vecs)  # [n_words, n_topics]
    M = normalize(M, norm="l1", axis=1)
    return np.array(words), M

# --- Step 6: Assign words to topics (soft) & summaries -----------------------
def assign_words_soft(words, M, min_membership=MIN_MEMBERSHIP, relative_keep=RELATIVE_KEEP):
    assignments = defaultdict(list)  # topic -> list of (word,score)
    unassigned = []
    for i, w in enumerate(words):
        vec = M[i]
        max_s = float(vec.max())
        keep_idx = np.where((vec >= min_membership) & (vec >= relative_keep * max_s))[0]
        if len(keep_idx) == 0:
            unassigned.append(w)
            continue
        for t in keep_idx:
            assignments[t].append((w, float(vec[t])))
    for t in assignments:
        assignments[t].sort(key=lambda x: x[1], reverse=True)
    return assignments, unassigned

def topic_summaries(H, feature_names, top_k=20):
    summaries = []
    for t in range(H.shape[0]):
        idx = np.argsort(H[t])[::-1][:top_k]
        terms = [feature_names[i] for i in idx]
        summaries.append((t, terms))
    return summaries

# --- Optional: cluster contexts directly as a check (cosine agglom) ----------
def cluster_contexts_agglomerative(embeddings, distance_threshold=1.0-0.35):
    # average linkage + cosine distance; threshold ~0.65 similarity
    clust = AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine', linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clust.fit_predict(embeddings)
    return labels

# --- RUN ---------------------------------------------------------------------
# 1) If you already created full_sentences earlier, reuse it; otherwise:
# DATA = pd.read_csv("assets/clustering_data_source.csv")
# full_sentences = build_sentences(DATA)
# Expecting two columns: did, sentence
# If you already have `full_sentences`, comment the two lines above.

full_sentences = full_sentences  # <-- reuse your existing variable from earlier cell

# 2) Build contexts
context_df, vocab_words = build_context_rows(full_sentences)
print(f"Built {len(context_df):,} (word,context) rows for {len(vocab_words):,} vocab words.")

# 3) Vectorize contexts and denoise
X_tfidf, X_svd, tfidf, svd = vectorize_contexts(context_df)
feature_names = np.array(tfidf.get_feature_names_out())

# 4) Fit NMF with small search over K
best, nmf_search_table = fit_best_nmf(X_tfidf, feature_names, N_TOPICS_GRID)
nmf_model, W_doc_topic = best["model"], best["W"]
H_topic_term = nmf_model.components_
print("NMF selection (higher is better):\n", nmf_search_table.sort_values("score", ascending=False))

# 5) Word-level soft memberships
words, M_word_topic = word_topic_memberships(context_df, W_doc_topic, H_topic_term.shape[0])
assignments, unassigned = assign_words_soft(words, M_word_topic)

# 6) Topic summaries & outputs
summaries = topic_summaries(H_topic_term, feature_names, top_k=TOP_K_WORDS_PER_TOPIC)

print(f"\nWords with ≥1 cluster: {sum(len(v) for v in assignments.values()):,}")
print(f"Unassigned (low-signal): {len(unassigned):,}")

# --- Pretty tables -----------------------------------------------------------
# A) topic → top descriptors
topic_labels = []
for t, terms in summaries:
    label = ", ".join(terms[:8])
    topic_labels.append({"topic": t, "label": label, "top_terms": ", ".join(terms)})

topic_labels_df = pd.DataFrame(topic_labels).sort_values("topic")

# B) word memberships (long, ranked)
rows = []
for t, pairs in assignments.items():
    for w, score in pairs[:5000]:   # cap for display
        rows.append({"topic": t, "word": w, "score": score})
word_memberships_df = pd.DataFrame(rows).sort_values(["topic","score"], ascending=[True,False])

# C) optional: cluster contexts (sanity check)
# labels = cluster_contexts_agglomerative(X_svd, distance_threshold=1.0-0.35)
# print("Agglomerative clusters on contexts:", np.unique(labels).size)

# --- Example: show / save outputs -------------------------------------------
# topic_labels_df.head(20)
# word_memberships_df.head(20)

# Save CSVs (uncomment to write)
# topic_labels_df.to_csv("topic_labels.csv", index=False)
# word_memberships_df.to_csv("word_memberships.csv", index=False)