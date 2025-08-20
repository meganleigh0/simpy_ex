# --- FULL WORD CLUSTERING ANALYSIS PIPELINE (compatible across sklearn versions) ---
# Make sure DATA is already loaded like:
# DATA = pd.read_json("data/fm_mine.json")

# === Imports ===
import re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

# Optional (PowerPoint output, if installed)
try:
    from pptx import Presentation
    from pptx.util import Inches
    pptx_available = True
except ImportError:
    pptx_available = False

# === 1. Clean the text ===
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)  # split 1200rpm -> 1200 rpm
    s = re.sub(r"[^a-zA-Z\- ]+", " ", s)        # keep words & hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s

texts = DATA["og"].astype(str).map(clean_text).tolist()

# === 2. TF-IDF vectorization ===
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",  # words length >= 3
    max_df=0.95,
    min_df=2
)
X = vectorizer.fit_transform(texts)
terms = np.array(vectorizer.get_feature_names_out())

# === 3. Latent semantic space (LSA) ===
svd = TruncatedSVD(n_components=100, random_state=0)
svd.fit(X)
term_vectors = svd.components_.T * svd.singular_values_
term_vectors = normalize(term_vectors)

# === 4. Nearest neighbor similarity ===
nbrs = NearestNeighbors(n_neighbors=5, metric="cosine").fit(term_vectors)
distances, _ = nbrs.kneighbors(term_vectors)
nn_sim = 1 - distances[:, 1]  # cosine similarity with 1st neighbor

# keep 60–80% of words (choose threshold adaptively)
threshold = np.quantile(nn_sim, 0.3)
clusterable_idx = np.where(nn_sim >= threshold)[0]
Y = term_vectors[clusterable_idx]

# === 5. Choose number of clusters using precomputed distances ===
D = cosine_distances(Y)
best_k, best_sil = None, -1
for k in range(5, min(40, len(Y))):
    try:
        model = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
        labels_tmp = model.fit_predict(D)
        if len(set(labels_tmp)) > 1:
            sil = silhouette_score(Y, labels_tmp, metric="cosine")
            if sil > best_sil:
                best_sil, best_k = sil, k
    except:
        continue
if best_k is None:
    best_k = 10

# === 6. Final clustering ===
model = AgglomerativeClustering(n_clusters=best_k, affinity="precomputed", linkage="average")
labels = model.fit_predict(D)

# assign cluster labels back to the vocab
cluster_labels = np.full(len(terms), -1)
cluster_labels[clusterable_idx] = labels

# === 7. Build results DataFrame ===
df_clusters = pd.DataFrame({
    "word": terms,
    "nn_similarity": nn_sim,
    "cluster": cluster_labels
}).sort_values(["cluster", "word"])

# === 8. Save outputs ===
out_dir = Path("out"); out_dir.mkdir(exist_ok=True)
df_clusters.to_csv(out_dir / "word_clusters.csv", index=False)

# === 9. Plot cluster sizes ===
cluster_sizes = df_clusters[df_clusters["cluster"] >= 0]["cluster"].value_counts().sort_index()
plt.bar(cluster_sizes.index, cluster_sizes.values)
plt.title("Cluster Sizes")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Words")
plt.savefig(out_dir / "cluster_sizes.png")
plt.close()

# === 10. (Optional) PowerPoint report ===
if pptx_available:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Word Clustering Report"
    slide.placeholders[1].text = (
        f"Unique words: {len(terms)}\n"
        f"Clusterable words: {len(clusterable_idx)}\n"
        f"Clusters: {best_k}"
    )
    prs.save(out_dir / "word_cluster_report.pptx")

print("Done! Files written to:", out_dir)