# Fix for older scikit-learn versions: use 'affinity' instead of 'metric' in AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=best_k, affinity="cosine", linkage="average")
labels = model.fit_predict(Y)

# Continue the pipeline from label assignment onward
cluster_labels_full = np.full(len(terms), -1, dtype=int)
cluster_labels_full[clusterable_idx] = labels

df_clusters = pd.DataFrame({
    "word": terms,
    "doc_freq": doc_freq,
    "nn_similarity": nn_sim,
    "cluster": cluster_labels_full
}).sort_values(["cluster", "doc_freq", "word"], ascending=[True, False, True])

text_series = DATA["og"].astype(str)
def sample_contexts(word, max_examples=2, max_len=140):
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    ex = []
    for txt in text_series.iloc[: min(len(text_series), 10000)]:
        if pattern.search(txt):
            s = pattern.sub(f"[{word.upper()}]", txt)
            ex.append(s if len(s) <= max_len else (s[: max_len - 1] + "…"))
        if len(ex) >= max_examples:
            break
    return ex

examples_col1, examples_col2 = [], []
for w in df_clusters["word"]:
    ex = sample_contexts(w)
    examples_col1.append(ex[0] if len(ex) > 0 else "")
    examples_col2.append(ex[1] if len(ex) > 1 else "")

df_clusters["example_1"] = examples_col1
df_clusters["example_2"] = examples_col2

out_dir = Path("/mnt/data"); out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "word_clusters.csv"
df_clusters.to_csv(csv_path, index=False)

assigned = (df_clusters["cluster"] >= 0).sum()
unique_words = len(df_clusters)
coverage_pct = 100.0 * assigned / unique_words

cluster_sizes = df_clusters[df_clusters["cluster"] >= 0]["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 4.5))
plt.bar(range(len(cluster_sizes)), cluster_sizes.values)
plt.title("Cluster Sizes")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Words")
chart_path = out_dir / "cluster_sizes.png"
plt.tight_layout()
plt.savefig(chart_path)
plt.close()

ppt_path = out_dir / "word_cluster_report.pptx"
if pptx_available:
    prs = Presentation()
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Word Clustering Report"
    slide.placeholders[1].text = (
        f"Documents: {len(DATA)}\n"
        f"Unique words (after filtering): {unique_words}\n"
        f"Clusterable words: {int((df_clusters['cluster']>=0).sum())} ({(df_clusters['cluster']>=0).mean():.1%} of vocab)\n"
        f"Clusters found: {model.n_clusters}\n"
        f"Final coverage (assigned to clusters): {coverage_pct:.1f}%"
    )

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Methods"
    body = slide.shapes.placeholders[1].text_frame
    body.text = (
        "• Cleaned text: lowercase, removed punctuation/numbers, kept hyphenated tokens.\n"
        "• Built TF‑IDF (unigrams, stopwords removed, min_df adaptive).\n"
        "• Reduced to latent semantic space with TruncatedSVD (LSA).\n"
        "• Estimated local similarity via nearest neighbors (cosine).\n"
        f"• Selected clusterable words by similarity ≥ {threshold:.2f} to target 60–80% coverage.\n"
        f"• Agglomerative clustering (average‑linkage, cosine) with k={model.n_clusters} chosen by silhouette."
    )

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Pitfalls Avoided / Considerations"
    body = slide.shapes.placeholders[1].text_frame
    body.text = (
        "• Polysemy: low‑similarity words left unclustered (cluster = −1).\n"
        "• Rare terms & typos: filtered using min_df and pre‑cleaned spellings.\n"
        "• Synonyms/abbreviations: embeddings group similar terms but may still split/merge.\n"
        "• Numeric artifacts/units: numbers stripped; consider custom tokenization for units.\n"
        "• Human-in-the-loop review recommended before using clusters downstream."
    )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Cluster Size Distribution"
    left = Inches(0.5); top = Inches(1.2); height = Inches(4.5)
    slide.shapes.add_picture(str(chart_path), left, top, height=height)

    # Show a few clusters
    tf = prs.slides.add_slide(prs.slide_layouts[1]).shapes.placeholders[1].text_frame
    prs.slides[-1].shapes.title.text = "Sample Clusters & Top Words"
    tf.clear()
    for cid in sorted(cluster_sizes.index.tolist())[:12]:
        words_in_cluster = df_clusters[df_clusters["cluster"] == cid].nlargest(10, "doc_freq")["word"].tolist()
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"Cluster {cid}: " + ", ".join(words_in_cluster)
        p.level = 0

    prs.save(str(ppt_path))
else:
    ppt_path = out_dir / "word_cluster_report.txt"
    with open(ppt_path, "w", encoding="utf-8") as f:
        f.write("pptx module not available; wrote a text summary instead.\n")

print({
    "csv_out": str(csv_path),
    "report_out": str(ppt_path),
    "chart": str(chart_path),
    "coverage_percent_assigned": round(coverage_pct, 2),
    "n_clusters": int(model.n_clusters),
})

from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Preview of word clusters", df_clusters.head(30))