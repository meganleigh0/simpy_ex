# ---------------------------------------------------------------------------
# Discover context‑specific synonyms and normalize failure‑report text
# ---------------------------------------------------------------------------
#
# Assumes: `df` exists and df['trimmed'] contains the raw text.
# Requires: pip install spacy scikit‑learn && python -m spacy download en_core_web_md
#
import pandas as pd
import spacy
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import re

# ----------------------------- CONFIG --------------------------------------
MIN_FREQ = 5          # ignore lemmas that appear fewer times than this
DIST_THRESHOLD = 0.25 # smaller ⇒ tighter synonym clusters
VECTOR_MODEL = "en_core_web_md"

# --------------------- 1. Tokenize, lemmatize, count ------------------------
nlp = spacy.load(VECTOR_MODEL, disable=["parser", "ner"])

vocab_counter = Counter()
for doc in nlp.pipe(df["trimmed"].astype(str).values, batch_size=500, n_process=-1):
    vocab_counter.update(
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop
    )

# candidate lemmas that are frequent enough
candidates = [w for w, f in vocab_counter.items() if f >= MIN_FREQ]
if not candidates:
    raise ValueError("No tokens meet MIN_FREQ threshold—lower MIN_FREQ or check data.")

# --------------------- 2. Get vectors & drop OOV tokens ---------------------
vectors = np.vstack([nlp.vocab[w].vector for w in candidates])
has_vector = np.linalg.norm(vectors, axis=1) > 0
vectors    = vectors[has_vector]
candidates = [w for w, keep in zip(candidates, has_vector) if keep]

# --------------------- 3. Cluster by cosine distance ------------------------
clustering = AgglomerativeClustering(
    affinity="cosine",
    linkage="average",
    distance_threshold=DIST_THRESHOLD,
    n_clusters=None
).fit(vectors)

labels = clustering.labels_

# ---------------- 4. Build canonical_to_synonyms mapping --------------------
groups = {}
for token, label in zip(candidates, labels):
    groups.setdefault(label, []).append(token)

canonical_to_synonyms = {}
for lemmas in groups.values():
    if len(lemmas) < 2:        # singleton cluster = no synonyms
        continue
    canonical = max(lemmas, key=lambda t: vocab_counter[t])  # most‑common word
    synonyms  = [l for l in lemmas if l != canonical]
    canonical_to_synonyms[canonical] = synonyms

# ---------------- 5. Compile replacement regex & normalize ------------------
variant_to_canonical = {
    v: c for c, syns in canonical_to_synonyms.items() for v in syns
}

escaped = sorted(map(re.escape, variant_to_canonical.keys()), key=len, reverse=True)
pattern = re.compile(rf"(?<!\w)({'|'.join(escaped)})(?!\w)", flags=re.IGNORECASE)

def normalize(text: str) -> str:
    if not isinstance(text, str) or not escaped:
        return text
    text = text.lower()
    return pattern.sub(lambda m: variant_to_canonical[m.group(0).lower()], text)

df["normalized"] = df["trimmed"].apply(normalize)

# ---------------- 6. (Optional) inspect what was learned --------------------
print("Top 20 automatically discovered synonym groups:")
for canon, syns in list(canonical_to_synonyms.items())[:20]:
    print(f"  {canon}  ←  {', '.join(syns)}")
