import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv("your_dataset.csv")  # Replace with actual file path

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Clean text function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Apply text cleaning
df["fail_mode_and_symptoms"] = df["fail_mode_and_symptoms"].apply(clean_text)

# If failure = 0, ensure symptoms are empty
df.loc[df["failure"] == 0, "fail_mode_and_symptoms"] = ""

# Identify unique fail modes and symptoms
unique_symptoms = df["fail_mode_and_symptoms"].value_counts()
print("Top unique symptoms:\n", unique_symptoms.head(20))

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["fail_mode_and_symptoms"])

# Cluster similar symptoms
num_clusters = 5  # Adjust based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["symptom_cluster"] = kmeans.fit_predict(X)

# Analyze which symptoms most often result in failure
failure_symptom_counts = df[df["failure"] == 1]["symptom_cluster"].value_counts()
print("Clusters associated with failures:\n", failure_symptom_counts)

# Save the cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")