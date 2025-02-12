import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv("your_data.csv")
print("Initial data sample:")
print(df.head())

# If your columns are: did, report_text, symptom, pname
# we'll focus on 'symptom' for cleaning and clustering.

############################################################################
# 2. Basic Normalization
############################################################################

def basic_normalization(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text

df['norm_symptom'] = df['symptom'].apply(basic_normalization)

# Check how many unique symptoms after basic normalization
print("Unique symptoms (just normalized):", df['norm_symptom'].nunique())

############################################################################
# 3. TF-IDF Vectorization
############################################################################
# We'll create a TF-IDF matrix from the 'norm_symptom' column.

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)  
# ngram_range=(1,2) helps capture bigrams
# min_df=2 ignores very rare terms that appear only once.

X = vectorizer.fit_transform(df['norm_symptom'])

print("TF-IDF matrix shape:", X.shape)
# Example: (2640, ???) depending on how many terms survive min_df

############################################################################
# 4. Find a Good Number of Clusters (Optional)
############################################################################
# If you don't know how many clusters to use, you can do a quick 
# silhouette-based or elbow method approach. We'll do a simple silhouette sample.

possible_ks = [5, 10, 15, 20, 30, 40]  # tweak based on data size
scores = []

for k in possible_ks:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels_temp = kmeans_temp.fit_predict(X)
    score_temp = silhouette_score(X, labels_temp)
    scores.append(score_temp)
    print(f"K={k}, silhouette={score_temp:.3f}")

# Plot silhouette to help decide
plt.plot(possible_ks, scores, marker='o')
plt.title("Silhouette Scores for Different K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()

# Let's pick a K based on the best silhouette or business logic.
# Suppose we see K=15 is decent, or K=20, etc.
best_k = 15  # example choice

############################################################################
# 5. K-Means Clustering with chosen K
############################################################################
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

print("Cluster assignments done. Number of clusters:", best_k)

############################################################################
# 6. Choose a Representative Label per Cluster
############################################################################
# A common approach: pick the most frequent "norm_symptom" in each cluster
# or pick the symptom with the highest average TF-IDF within that cluster.

cluster_to_symptoms = {}
for idx, row in df.iterrows():
    c = row['cluster']
    symptom = row['norm_symptom']
    if c not in cluster_to_symptoms:
        cluster_to_symptoms[c] = []
    cluster_to_symptoms[c].append(symptom)

# Let's define a simple function to find the "most frequent symptom" in each cluster
cluster_reps = {}
for c, symptoms in cluster_to_symptoms.items():
    freq = pd.Series(symptoms).value_counts()
    # The top symptom is freq.index[0]
    representative = freq.index[0]
    cluster_reps[c] = representative

# Another approach might be to find the symptom string that is closest to the cluster centroid in TF-IDF space.
# We'll do the simpler approach here.

# Now map each cluster to its representative
def get_cluster_label(c):
    return cluster_reps[c]

df['cluster_label'] = df['cluster'].apply(get_cluster_label)

############################################################################
# 7. (Optional) Final Fuzzy Matching to a Known Symptom List
############################################################################
# If you have a curated list, map these cluster representatives to that list.
from fuzzywuzzy import process

valid_symptoms = [
    "fever", "cough", "headache", "dizziness", "vomiting", "nausea",
    "fatigue", "chills", "sore throat", "shortness of breath",
    "chest pain", "diarrhea", "constipation", "rash", "body aches",
    "migraine", "arrhythmia", "hypertension"
    # etc.
]

def fuzzy_map_to_valid(sym, valid_list=valid_symptoms, threshold=80):
    if not sym:
        return sym
    best_match, score = process.extractOne(sym, valid_list)
    if score >= threshold:
        return best_match
    return sym

df['final_symptom'] = df['cluster_label'].apply(fuzzy_map_to_valid)

############################################################################
# 8. Check Results
############################################################################
print("Unique raw symptoms before:", df['symptom'].nunique())
print("Unique final symptoms after cluster+map:", df['final_symptom'].nunique())

# Let's see some example transformations
print(df[['symptom','norm_symptom','cluster','cluster_label','final_symptom']].head(30))

############################################################################
# 9. Visualization
############################################################################
# a) Compare unique counts
unique_before = df['symptom'].nunique()
unique_after = df['final_symptom'].nunique()

plt.bar(["Before","After"], [unique_before, unique_after], color=['red','green'])
plt.title("Unique Symptom Count: Before vs. After Clustering & Mapping")
plt.ylabel("Count of Unique Symptoms")
plt.show()

# b) Frequency of final symptoms
plt.figure(figsize=(10,6))
sns.countplot(y=df['final_symptom'], order=df['final_symptom'].value_counts().index)
plt.title("Symptom Frequency (After Clustering and Mapping)")
plt.xlabel("Count")
plt.ylabel("Symptom")
plt.tight_layout()
plt.show()
