import pandas as pd
import re
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual file path

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Function to clean text: remove special chars, extra spaces, and normalize wording
def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""
    text = text.lower().strip()  # Lowercase and trim spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    return text

# Apply text cleaning
df["symptom"] = df["symptom"].apply(clean_text)
df["fmode"] = df["fmode"].apply(clean_text)

# Load Sentence-BERT model for better semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode symptoms and failure modes into vector space
symptom_embeddings = model.encode(df["symptom"].unique(), convert_to_tensor=True)
fmode_embeddings = model.encode(df["fmode"].unique(), convert_to_tensor=True)

# Compute similarity matrix
symptom_sim_matrix = cosine_similarity(symptom_embeddings.cpu().detach().numpy())
fmode_sim_matrix = cosine_similarity(fmode_embeddings.cpu().detach().numpy())

# Apply hierarchical clustering
num_clusters = 20  # Adjust based on dataset size
symptom_clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="complete").fit(1 - symptom_sim_matrix)
fmode_clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="complete").fit(1 - fmode_sim_matrix)

# Map clusters back to dataframe
symptom_mapping = dict(zip(df["symptom"].unique(), symptom_clusters.labels_))
fmode_mapping = dict(zip(df["fmode"].unique(), fmode_clusters.labels_))

df["symptom_cluster"] = df["symptom"].map(symptom_mapping)
df["fmode_cluster"] = df["fmode"].map(fmode_mapping)

# Save refined dataset
df.to_csv("refined_dataset.csv", index=False)
print("Refined dataset saved as 'refined_dataset.csv'")

# Display cluster mapping
import ace_tools as tools
tools.display_dataframe_to_user(name="Refined Symptom & Failure Mode Clusters", dataframe=df)