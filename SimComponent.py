# --- Final NMF model using K=20 and important word filtering ---

import re, math, numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# -------------------
# Config
# -------------------
INPUT_PATH = "assets/clustering_data_source.csv"
IMPORTANT_WORDS_PATH = "assets/important_words.txt"

TEXT_COL = "sentence"
LOWERCASE = True
STOP_WORDS = "english"
NGRAM_RANGE = (1, 2)
MIN_DF = 5
MAX_FEATURES = 10_000
RANDOM_STATE = 42
K = 20
MAX_ITER = 400
TOP_TERMS_PER_TOPIC = 20

# -------------------
# Load important words list
# -------------------
with open(IMPORTANT_WORDS_PATH, "r") as f:
    important_words = set([w.strip().lower() for w in re.split(r"[,\\s]+", f.read()) if w.strip()])
print(f"Loaded {len(important_words)} important words.")

# -------------------
# Load and combine sentences
# -------------------
DATA = pd.read_csv(INPUT_PATH)
DATA["token"] = DATA["token"].fillna("").astype(str)

sentences = (
    DATA.sort_values(["did", "sentence_beg", "place"])
        .groupby(["did", "sentence_beg", "sentence_end"])
        .agg({"token": lambda s: " ".join(s)})
        .reset_index()
)

full_sentences = (
    sentences.groupby("did")["token"].apply(lambda s: " ".join(s))
             .reset_index().rename(columns={"token": TEXT_COL})
)

docs = full_sentences[TEXT_COL].astype(str).fillna("")
print(f"Loaded {len(docs)} documents for training.")

# -------------------
# TF-IDF vectorization
# -------------------
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
print(f"TF-IDF matrix shape: {X.shape}")

# -------------------
# NMF model (K=20)
# -------------------
nmf = NMF(
    n_components=K, 
    init="nndsvd", 
    random_state=RANDOM_STATE, 
    max_iter=MAX_ITER
)
W = nmf.fit_transform(X)
H = nmf.components_
Wn = normalize(W, norm="l2", axis=1)
labels = Wn.argmax(axis=1)

# -------------------
# Filter topics: only keep important words
# -------------------
topic_terms = []
for k in range(K):
    topic_weights = H[k]
    sorted_idx = np.argsort(topic_weights)[::-1]
    filtered_idx = [i for i in sorted_idx if terms[i].split()[0].lower() in important_words]
    top_idx = filtered_idx[:TOP_TERMS_PER_TOPIC]
    topic_terms.append([terms[i] for i in top_idx])

topic_df = pd.DataFrame({
    "topic": [f"Topic {i:02d}" for i in range(K)],
    "top_terms": [", ".join(tt) for tt in topic_terms]
})

# -------------------
# Assign documents to topics
# -------------------
docs_df = full_sentences.copy()
docs_df["topic"] = labels
docs_df["topic_confidence"] = W.max(axis=1)

# -------------------
# Save outputs
# -------------------
topic_df.to_csv("nmf_filtered_topics.csv", index=False)
docs_df.to_csv("nmf_filtered_doc_assignments.csv", index=False)
print("Saved filtered topic and document assignment files.")

print("\nTop terms per topic (filtered):")
display(topic_df.head(10))