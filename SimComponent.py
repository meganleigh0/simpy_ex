import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual file path

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Function to clean text (removes special characters, extra spaces)
def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Apply cleaning to both 'symptom' and 'fmode' columns
df["symptom"] = df["symptom"].apply(clean_text)
df["fmode"] = df["fmode"].apply(clean_text)

# Ensure that if failure = 0, symptom should be empty
df.loc[df["failure"] == 0, "symptom"] = ""

# Check unique values after cleaning
unique_symptoms = df["symptom"].value_counts()
unique_fmodes = df["fmode"].value_counts()

print("Top unique symptoms:\n", unique_symptoms.head(20))
print("Top unique failure modes:\n", unique_fmodes.head(20))

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

# Analyze which failure modes are most linked to specific symptoms
failure_analysis = df.groupby(["fmode_cluster", "symptom_cluster"])["failure"].sum().reset_index()

# Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")

# Display failure analysis results
import ace_tools as tools
tools.display_dataframe_to_user(name="Failure Mode & Symptom Analysis", dataframe=failure_analysis)