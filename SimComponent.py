# === FAST WORD CLUSTERING: MULTI-METHOD COMPARISON (single cell) ===
# Outputs (in ./out):
#   summary.csv                                   -> runtime, coverage, silhouette per method
#   word_clusters_<method>_k<k>.csv               -> labels + examples for each method tried

import re, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

# --------------------- CONFIG (tweak for speed/quality) ---------------------
TEXT_COL         = "og"
MAX_FEATURES     = 30000        # cap vocabulary
MIN_DF           = 2            # ignore singletons
N_COMPONENTS     = 100          # LSA dimensions (50–200 is typical)
NN_FOR_MASK      = 6            # neighborhood size for "clusterable" check
TARGET_COVERAGE  = 0.70         # aim ~70% words assigned (within 60–80%)
SAMPLE_FOR_SCORE = 2000         # sample size for silhouette scoring
K_GRID_FACTOR    = 1.0          # multiply the auto k guesses by this
OUT_DIR          = Path("out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

assert TEXT_COL in DATA.columns, f"DATA must have a '{TEXT_COL}' column."

# --------------------- 1) Clean text ---------------------
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)  # 1200rpm -> 1200 rpm
    s = re.sub(r"[^a-zA-Z\- ]+", " ", s)        # keep letters & hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s

texts = DATA[TEXT_COL].astype(str).map(clean_text).tolist()

# --------------------- 2) TF-IDF ---------------------
vect = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    max_df=0.95,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
)
X = vect.fit_transform(texts)                         # docs x terms
terms = np.array(vect.get_feature_names_out())
doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()

# --------------------- 3) LSA term embeddings ---------------------
svd = TruncatedSVD(n_components=min(N_COMPONENTS, max(10, X.shape[1]//2)), random_state=0)
svd.fit(X)
Y_all = svd.components_.T * svd.singular_values_      # term vectors
Y_all = normalize(Y_all)                              # unit-length -> cosine == dot

# --------------------- 4) Pick clusterable words (hit ~70% target) ---------------------
nbrs = NearestNeighbors(n_neighbors=min(NN_FOR_MASK, len(terms)), metric="cosine").fit(Y_all)
dist, _ = nbrs.kneighbors(Y_all)
nn_sim = 1 - dist[:, 1]
# choose threshold so coverage ~TARGET_COVERAGE (clipped to 0.60–0.80 window)
q = np.quantile(nn_sim, 1 - TARGET_COVERAGE)
mask = nn_sim >= q
coverage = mask.mean()
if coverage < 0.60 or coverage > 0.80:
    # nudge inside band
    q = np.quantile(nn_sim, 0.40) if coverage < 0.60 else np.quantile(nn_sim, 0.20)
    mask = nn_sim >= q
cluster_idx = np.where(mask)[0]
Y = Y_all[cluster_idx]
terms_used = terms[cluster_idx]
doc_freq_used = doc_freq[cluster_idx]

# --------------------- 5) Helper: choose K & simple evaluators ---------------------
m = len(Y)
if m < 10: raise ValueError("Not enough clusterable words; relax MIN_DF or lower threshold.")

def auto_k_grid(n):
    base = max(6, int(np.sqrt(n)))
    grid = sorted(set([max(2, int(base * s * K_GRID_FACTOR)) for s in (0.5, 1.0, 1.5, 2.0)]))
    return [k for k in grid if k < n]

def sample_for_score(Y, labels, max_n=SAMPLE_FOR_SCORE):
    n = len(Y)
    if n <= max_n:
        return silhouette_score(Y, labels, metric="cosine")
    idx = np.random.RandomState(0).choice(n, size=max_n, replace=False)
    return silhouette_score(Y[idx], labels[idx], metric="cosine")

def coverage_threshold(sims, target=0.70, band=(0.60, 0.80)):
    # choose threshold to keep ~target fraction with band tolerance
    q = np.quantile(sims, 1 - target)
    cov = (sims >= q).mean()
    if cov < band[0]: q = np.quantile(sims, 1 - band[0])
    if cov > band[1]: q = np.quantile(sims, 1 - band[1])
    return q

def finalize_labels_from_centers(Y, labels, centers):
    # cosine sim to own center; drop below threshold to get ~target coverage
    centers = normalize(centers)
    sims = np.sum(Y * centers[labels], axis=1)        # since all unit-norm
    thr = coverage_threshold(sims, TARGET_COVERAGE)
    keep = sims >= thr
    lab_fin = labels.copy()
    lab_fin[~keep] = -1
    return lab_fin, sims, thr

# --------------------- 6) Methods ---------------------
results = []
labels_by_method = {}

# (A) MiniBatchKMeans (fast, scalable)
from sklearn.cluster import MiniBatchKMeans
k_grid = auto_k_grid(m)
for k in k_grid:
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=2048, n_init=10)
    labels = km.fit_predict(Y)
    sscore = sample_for_score(Y, labels)
    lab_fin, sims, thr = finalize_labels_from_centers(Y, labels, km.cluster_centers_)
    t1 = time.time()
    assigned = (lab_fin >= 0).mean()*100
    results.append(["MiniBatchKMeans", k, round(sscore,4), round(assigned,1), round(t1-t0,2), thr])
    labels_by_method[("MiniBatchKMeans", k)] = (lab_fin, sims)

# (B) BIRCH (linear-time)
from sklearn.cluster import Birch
for k in k_grid:
    t0 = time.time()
    br = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
    labels = br.fit_predict(Y)
    # compute simple centers from assignments
    centers = np.vstack([normalize(Y[labels==i].mean(axis=0).reshape(1,-1))[0] for i in range(k)])
    sscore = sample_for_score(Y, labels)
    lab_fin, sims, thr = finalize_labels_from_centers(Y, labels, centers)
    t1 = time.time()
    assigned = (lab_fin >= 0).mean()*100
    results.append(["BIRCH", k, round(sscore,4), round(assigned,1), round(t1-t0,2), thr])
    labels_by_method[("BIRCH", k)] = (lab_fin, sims)

# (C) Agglomerative on a small sample, then assign rest to nearest center
def safe_agglom(Ysub, k):
    # try metric then affinity; if both fail, precomputed on the sample
    try:
        model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine")
        return model.fit_predict(Ysub)
    except TypeError:
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="cosine")
            return model.fit_predict(Ysub)
        except TypeError:
            D = cosine_distances(Ysub)
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed")
            except TypeError:
                model = AgglomerativeClustering(n_clusters=k, linkage="average", affinity="precomputed")
            return model.fit_predict(D)

sub_n = min(1500, m)  # keep sample small -> fast O(n^2)
sub_idx = np.random.RandomState(0).choice(m, size=sub_n, replace=False)
Ysub = Y[sub_idx]
for k in k_grid:
    t0 = time.time()
    labels_sub = safe_agglom(Ysub, k)
    # centers from sample
    centers = np.vstack([normalize(Ysub[labels_sub==i].mean(axis=0).reshape(1,-1))[0] for i in range(k)])
    # assign all points to nearest center (cosine)
    sims_all = cosine_similarity(Y, centers)
    labels_all = sims_all.argmax(axis=1)
    sscore = sample_for_score(Y, labels_all)
    lab_fin, sims, thr = finalize_labels_from_centers(Y, labels_all, centers)
    t1 = time.time()
    assigned = (lab_fin >= 0).mean()*100
    results.append(["Agglo(sample→assign)", k, round(sscore,4), round(assigned,1), round(t1-t0,2), thr])
    labels_by_method[("Agglo(sample→assign)", k)] = (lab_fin, sims)

# --------------------- 7) Summarize & export ---------------------
summary = pd.DataFrame(results, columns=["method","k","silhouette_cosine(sample)","assigned_%","fit_seconds","sim_threshold"])
summary = summary.sort_values(["silhouette_cosine(sample)","assigned_%"], ascending=[False, False])
summary_path = OUT_DIR / "summary.csv"
summary.to_csv(summary_path, index=False)
print("Summary written:", summary_path)
display(summary.head(10))

# Save a labeled CSV per (method,k) tried
def save_labels(method, k, labels, sims):
    df = pd.DataFrame({
        "word": terms_used,
        "doc_freq": doc_freq_used,
        "nn_similarity": nn_sim[cluster_idx],
        "cluster": labels,
        "sim_to_center": sims,
    }).sort_values(["cluster","doc_freq","word"], ascending=[True, False, True])
    out = OUT_DIR / f"word_clusters_{method.replace('/','_')}_k{k}.csv"
    df.to_csv(out, index=False)
    return out

paths = []
for (method, k), (lab_fin, sims) in labels_by_method.items():
    paths.append(save_labels(method, k, lab_fin, sims))

print("Wrote label files:", len(paths))
for p in paths[:5]:
    print("  ", p)