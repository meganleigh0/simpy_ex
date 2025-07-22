# ---------------------------------------------------------------------------
# Auto-discover context-specific synonyms and normalize failure-report text
# ---------------------------------------------------------------------------
# Assumes df['trimmed'] exists.  Requires:
#   pip install spacy scikit-learn && python -m spacy download en_core_web_md
#
import pandas as pd
import spacy
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import re

# ----------------------------- CONFIG --------------------------------------
MIN_FREQ        = 5          # ignore lemmas that appear < MIN_FREQ times
DIST_THRESHOLD  = 0.25       # tighten / loosen synonym clusters
VECTOR_MODEL    = "en_core_web_md"

# --------------------- 1. Tokenize, lemmatize, count ------------------------
nlp = spacy.load(VECTOR_MODEL, disable=["parser", "ner"])

vocab_counter = Counter()
for doc in nlp.pipe(df["trimmed"].astype(str).values, batch_size=500, n_process=-1):
    vocab_counter.update(
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop
    )

candidates = [w for w, f in vocab_counter.items() if f >= MIN_FREQ]
if not candidates:
    raise ValueError("No tokens meet MIN_FREQ threshold—lower MIN_FREQ or check data.")

# --------------------- 2. Get vectors & drop OOV tokens ---------------------
vectors     = np.vstack([nlp.vocab[w].vector for w in candidates])
has_vector  = np.linalg.norm(vectors, axis=1) > 0
vectors     = vectors[has_vector]
candidates  = [w for w, keep in zip(candidates, has_vector) if keep]

# --------------------- 3. Cluster by cosine distance ------------------------
# Newer scikit-learn uses `metric`, not `affinity`
clustering = AgglomerativeClustering(
    metric="cosine",           # <- FIXED
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
    if len(lemmas) < 2:
        continue
    canonical = max(lemmas, key=lambda t: vocab_counter[t])  # most frequent
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
    return pattern.sub(lambda m: variant_to_canonical[m.group(0).lower()],
                       text.lower())

df["normalized"] = df["trimmed"].apply(normalize)

# ---------------- 6. Quick sanity check -------------------------------------
print("Discovered synonym groups (top 20):")
for canon, syns in list(canonical_to_synonyms.items())[:20]:
    print(f"{canon:<12} ← {', '.join(syns)}")
