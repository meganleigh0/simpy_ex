import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from fuzzywuzzy import fuzz, process

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual file path

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Function to clean text further
def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Apply text cleaning
df["symptom"] = df["symptom"].apply(clean_text)
df["fmode"] = df["fmode"].apply(clean_text)

# Ensure that if failure = 0, symptom should be empty
df.loc[df["failure"] == 0, "symptom"] = ""

# Group similar failure modes and symptoms using fuzzy matching
def match_similar_strings(text_series, threshold=85):
    unique_texts = text_series.unique()
    grouped_texts = {}
    
    for text in unique_texts:
        best_match = process.extractOne(text, grouped_texts.keys(), scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            grouped_texts[best_match[0]].append(text)
        else:
            grouped_texts[text] = [text]

    # Create mapping of similar texts to representative form
    text_mapping = {text: representative for representative, matches in grouped_texts.items() for text in matches}
    return text_mapping

# Apply fuzzy matching to symptoms and failure modes
symptom_mapping = match_similar_strings(df["symptom"])
fmode_mapping = match_similar_strings(df["fmode"])

# Replace with grouped values
df["symptom"] = df["symptom"].map(symptom_mapping)
df["fmode"] = df["fmode"].map(fmode_mapping)

# Vectorize text using TF-IDF for clustering
vectorizer = TfidfVectorizer()
X_symptom = vectorizer.fit_transform(df["symptom"])
X_fmode = vectorizer.fit_transform(df["fmode"])

# Cluster similar symptoms and failure modes
num_clusters = 5  # Adjust based on dataset size
kmeans_symptom = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans_fmode = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)

df["symptom_cluster"] = kmeans_symptom.fit_predict(X_symptom)
df["fmode_cluster"] = kmeans_fmode.fit_predict(X_fmode)

# Analyze failure correlations
failure_analysis = df.groupby(["fmode_cluster", "symptom_cluster"])["failure"].sum().reset_index()

# Save the refined dataset
df.to_csv("refined_dataset.csv", index=False)
print("Refined dataset saved as 'refined_dataset.csv'")

# Display failure mode and symptom analysis results
import ace_tools as tools
tools.display_dataframe_to_user(name="Refined Failure Mode & Symptom Analysis", dataframe=failure_analysis)