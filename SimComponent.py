# --- Word Usage Clustering (single cell) --------------------------------------
# Clusters unique words by the *contexts* they appear in. Words can belong to
# multiple clusters (soft membership), and low-signal words can remain unassigned.

import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# ------------------------------------------------------------------------------
# 0) CONFIG (tweak these to taste)
# ------------------------------------------------------------------------------
CONTEXT_WINDOW = 5          # +/- tokens around a target word to define its context
MIN_WORD_LEN = 3            # ignore very short tokens
MIN_WORD_FREQ = 20          # only model words that appear at least this many times
MAX_CONTEXTS_PER_WORD = 60  # cap contexts per word to keep runtime sane
N_TOPICS = 20               # number of usage clusters to learn
MAX_FEATURES = 30000        # TF-IDF max features for contexts
MIN_MEMBERSHIP = 0.18       # absolute min topic score to consider an assignment
RELATIVE_KEEP = 0.60        # keep topics within this fraction of a word's max score
TOP_K_WORDS_PER_TOPIC = 40  # for reporting which words best represent each topic
TOKEN_PATTERN = r"[a-zA-Z][a-zA-Z\-']+"  # keep alphabetic words with hyphen/'

# ------------------------------------------------------------------------------
# 1) Basic text helpers
# ------------------------------------------------------------------------------
_stop = set(ENGLISH_STOP_WORDS)

def sentence_split(text: str):
    # lightweight sentence splitter
    return re.split(r"(?<=[\.\?\!])\s+|\n+", text)

def tokenize(text: str):
    # lower + regex + basic filtering
    toks = re.findall(TOKEN_PATTERN, (text or "").lower())
    return [t for t in toks if len(t) >= MIN_WORD_LEN]

# ------------------------------------------------------------------------------
# 2) Build (word, DID, context_text) rows by sliding window over sentences
# ------------------------------------------------------------------------------
def build_context_rows(df: pd.DataFrame):
    rows = []
    word_counts = Counter()

    # first pass: count words (for MIN_WORD_FREQ filtering later)
    for _, r in df[["DID", "OG"]].itertuples(index=False):
        for sent in sentence_split(r):
            toks = tokenize(sent)
            for tok in toks:
                if tok not in _stop:
                    word_counts[tok] += 1

    vocab_words = {w for w, c in word_counts.items() if c >= MIN_WORD_FREQ}

    # second pass: collect contexts, capped per word
    per_word_added = Counter()
    for did, text in df[["DID", "OG"]].itertuples(index=False):
        for sent in sentence_split(text):
            toks = tokenize(sent)
            if not toks:
                continue
            for i, w in enumerate(toks):
                if w in _stop or w not in vocab_words:
                    continue
                if per_word_added[w] >= MAX_CONTEXTS_PER_WORD:
                    continue
                lo = max(0, i - CONTEXT_WINDOW)
                hi = min(len(toks), i + CONTEXT_WINDOW + 1)
                context_tokens = toks[lo:i] + toks[i+1:hi]
                # drop stopwords inside the context to sharpen signal
                context_tokens = [t for t in context_tokens if t not in _stop]
                if not context_tokens:
                    continue
                context_text = " ".join(context_tokens)
                rows.append((w, did, context_text))
                per_word_added[w] += 1

    ctx_df = pd.DataFrame(rows, columns=["word", "DID", "context"])
    return ctx_df, sorted(vocab_words)

# Expect an existing DataFrame 'data' with columns DID, OG
assert isinstance(data, pd.DataFrame) and {"DID","OG"}.issubset(data.columns), \
    "Expected a DataFrame named `data` with columns ['DID','OG']."

context_df, vocab_words = build_context_rows(data)
print(f"Built {len(context_df):,} (word,context) samples for {len(vocab_words):,} words.")

# If very large, consider sampling contexts to speed up experimentation:
# context_df = context_df.sample(n=min(len(context_df), 150_000), random_state=1)

# ------------------------------------------------------------------------------
# 3) TF-IDF of contexts and NMF topic model (usage clusters)
# ------------------------------------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=MAX_FEATURES,
    token_pattern=TOKEN_PATTERN,
    lowercase=True,
    ngram_range=(1,2)  # include some bigrams to capture local collocations
)
X = tfidf.fit_transform(context_df["context"])

nmf = NMF(
    n_components=N_TOPICS,
    init="nndsvda",
    random_state=42,
    max_iter=500,
    alpha_W=0.0,
    alpha_H=0.0,
    l1_ratio=0.0
)
W = nmf.fit_transform(X)   # (num_contexts x N_TOPICS): topic mix per context
H = nmf.components_        # (N_TOPICS x vocab_size): top terms per topic

# Normalize W so topic weights are comparable
W = normalize(W, norm="l1", axis=1)

# ------------------------------------------------------------------------------
# 4) Aggregate context-topic mixtures up to the *word* level (soft membership)
# ------------------------------------------------------------------------------
# We average topic mixtures over all contexts of a given word.
word_topic_sum = defaultdict(lambda: np.zeros(N_TOPICS, dtype=np.float32))
word_counts = Counter()

for (w, _did), row_w in zip(context_df[["word","DID"]].itertuples(index=False, name=None), W):
    word_topic_sum[w] += row_w
    word_counts[w] += 1

words = []
word_topic = []
for w in word_topic_sum:
    avg = word_topic_sum[w] / word_counts[w]
    words.append(w)
    word_topic.append(avg)

word_topic = np.vstack(word_topic)  # (num_words x N_TOPICS)

# ------------------------------------------------------------------------------
# 5) Assign words to clusters with soft rules (multi-membership + unassigned)
# ------------------------------------------------------------------------------
assignments = defaultdict(list)  # topic -> list[(word, score)]
unassigned = []

for i, w in enumerate(words):
    vec = word_topic[i]
    max_score = float(vec.max())
    if max_score < MIN_MEMBERSHIP:
        unassigned.append(w)
        continue
    # keep topics close to the best one, and above absolute floor
    keep_idx = np.where((vec >= MIN_MEMBERSHIP) & (vec >= RELATIVE_KEEP * max_score))[0]
    if len(keep_idx) == 0:
        unassigned.append(w)
        continue
    for t in keep_idx:
        assignments[t].append((w, float(vec[t])))

# sort assignments by strength per topic
for t in assignments:
    assignments[t].sort(key=lambda x: x[1], reverse=True)

print(f"\nWords with at least one cluster: {sum(len(v) for v in assignments.values()):,}")
print(f"Unassigned words (low-signal):   {len(unassigned):,}")

# ------------------------------------------------------------------------------
# 6) Human-readable summaries
# ------------------------------------------------------------------------------
# a) Top *terms* (context vocabulary) that define each topic (to label clusters)
feature_names = np.array(tfidf.get_feature_names_out())
topic_summaries = []
for t in range(N_TOPICS):
    top_term_idx = np.argsort(H[t])[::-1][:20]
    top_terms = feature_names[top_term_idx].tolist()
    topic_summaries.append((t, top_terms))

print("\n=== Topic label hints (top context terms) ===")
for t, terms in topic_summaries:
    print(f"[Topic {t:02d}] " + ", ".join(terms))

# b) Top words (from your corpus) that belong to each topic
print("\n=== Top words per topic (by membership) ===")
for t in range(N_TOPICS):
    if t not in assignments or len(assignments[t]) == 0:
        continue
    top_words = assignments[t][:TOP_K_WORDS_PER_TOPIC]
    preview = ", ".join(f"{w}({s:.2f})" for w, s in top_words[:20])
    print(f"[Topic {t:02d}] {preview}")

# ------------------------------------------------------------------------------
# 7) Export artifacts for downstream use
# ------------------------------------------------------------------------------
# (a) Word-topic membership matrix
word_topic_df = pd.DataFrame(word_topic, index=words, columns=[f"topic_{i:02d}" for i in range(N_TOPICS)])
word_topic_df.index.name = "word"

# (b) Assignments (long form)
rows = []
for t, items in assignments.items():
    for w, s in items:
        rows.append((w, t, s))
assignments_df = pd.DataFrame(rows, columns=["word", "topic", "score"]).sort_values(["topic","score"], ascending=[True, False])

# (c) Unassigned words
unassigned_df = pd.DataFrame({"word": sorted(unassigned)})

# Uncomment to persist locally if desired:
# word_topic_df.to_csv("word_topic_membership.csv")
# assignments_df.to_csv("word_cluster_assignments.csv", index=False)
# unassigned_df.to_csv("words_unassigned.csv", index=False)

# Quick peeks
print("\n=== Sample of word -> topic scores ===")
display(word_topic_df.head(20))

print("\n=== Sample assignments (first 10 per topic) ===")
display(assignments_df.groupby("topic").head(10))

print("\n=== Unassigned sample ===")
display(unassigned_df.head(50))

# ------------------------------------------------------------------------------
# Notes:
# - Increase N_TOPICS for finer clusters; decrease to coarsen.
# - Raise MIN_MEMBERSHIP to be stricter (more words become unassigned).
# - To capture *senses* more explicitly, you can first cluster contexts per word
#   (e.g., KMeans on context TF-IDF for each word) to create word-sense rows,
#   then run NMF on those; the above keeps things simple and fast to start.
# - Swap NMF for TruncatedSVD + KMeans if you prefer hard clusters; keep soft
#   membership by converting distances to similarities and thresholding.
# ------------------------------------------------------------------------------