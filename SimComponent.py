import pandas as pd
import numpy as np
import re
from collections import Counter

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK tools
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics import edit_distance

nltk.download('punkt')
nltk.download('stopwords')

# Optional: if you want spaCy for better tokenization/lemmatization:
# import spacy
# nlp = spacy.load("en_core_web_sm")

##############################################################################
# 1. Load data
##############################################################################
# Replace "your_data.csv" with your actual file path
df = pd.read_csv("your_data.csv")

print("Data sample:")
print(df.head())

##############################################################################
# 2. Basic normalization
##############################################################################
def basic_normalization(text):
    if pd.isna(text):
        return ""
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # strip whitespace
    text = text.strip()
    return text

df['norm_symptom'] = df['symptom'].apply(basic_normalization)

##############################################################################
# 3. Tokenize (and optionally Lemmatize) the symptoms
##############################################################################

# Option A: Pure NLTK tokenization
stop_words = set(stopwords.words('english'))

def tokenize_nltk(text):
    tokens = word_tokenize(text)
    # optionally remove stopwords if needed
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# Option B: spaCy-based tokenization + lemmatization (uncomment if needed)
# def tokenize_spacy(text):
#     doc = nlp(text)
#     tokens = []
#     for token in doc:
#         if not token.is_stop and not token.is_punct and not token.is_space:
#             # Use the lemma if you want the root form (helps unify forms):
#             tokens.append(token.lemma_.lower())
#     return tokens

def tokenize_and_lemmatize(text):
    # Here we just do NLTK tokenization. If you want spaCy, replace the line below.
    tokens = tokenize_nltk(text)
    return tokens

df['tokens'] = df['norm_symptom'].apply(tokenize_and_lemmatize)

##############################################################################
# 4. Build a vocabulary from the data (or use an external list)
##############################################################################
# We'll create a vocabulary of all tokens that appear with some minimum frequency 
# so we don't keep super-rare or spurious tokens in the vocabulary.

all_tokens = []
for toks in df['tokens']:
    all_tokens.extend(toks)

counter = Counter(all_tokens)

# Let's keep tokens that appear at least twice, for example:
min_freq = 2
vocab = {token for token, freq in counter.items() if freq >= min_freq}

print(f"Vocabulary size (min freq={min_freq}):", len(vocab))

##############################################################################
# 5. Edit-Distance-Based Correction
##############################################################################
def correct_tokens_with_edit_distance(tokens, vocabulary, max_distance=2):
    """
    For each token not in 'vocabulary', find the closest match (by edit distance).
    If edit distance < max_distance, replace it; otherwise keep original.
    """
    corrected_tokens = []
    for t in tokens:
        if t in vocabulary:
            corrected_tokens.append(t)
        else:
            # Find best candidate in vocabulary
            best_match = None
            best_dist = float('inf')
            for candidate in vocabulary:
                dist = edit_distance(t, candidate)
                if dist < best_dist:
                    best_dist = dist
                    best_match = candidate
                # Optional optimization: break early if dist==0 or dist>max_distance
                if best_dist == 0:
                    break
            # If the best distance is below threshold, replace
            if best_dist <= max_distance:
                corrected_tokens.append(best_match)
            else:
                # keep original if no good match found
                corrected_tokens.append(t)
    return corrected_tokens

df['tokens_corrected'] = df['tokens'].apply(lambda toks: correct_tokens_with_edit_distance(toks, vocab, max_distance=2))

# Rejoin tokens into a single string
df['corrected_symptom'] = df['tokens_corrected'].apply(lambda toks: " ".join(toks))

##############################################################################
# (Optional) 6. Fuzzy match to a curated symptom list
##############################################################################
# If you do have a known list of symptom terms, we can unify them further.
# We'll use fuzzywuzzy for demonstration:
from fuzzywuzzy import process

valid_symptoms = [
    "fever", "cough", "headache", "dizziness", "vomiting", "nausea",
    "fatigue", "chills", "sore throat", "shortness of breath",
    "chest pain", "diarrhea", "constipation", "rash", "body aches",
    "migraine", "arrhythmia", "hypertension"
    # ... etc ...
]

def fuzzy_match_symptom(symptom_str, valid_list=valid_symptoms, threshold=80):
    """
    For multi-word strings, you could:
      - fuzzy match the entire string
      - or match each token/phrase individually
    """
    if not symptom_str:
        return symptom_str
    best_match, score = process.extractOne(symptom_str, valid_list)
    if score >= threshold:
        return best_match
    else:
        return symptom_str

df['final_symptom'] = df['corrected_symptom'].apply(fuzzy_match_symptom)

##############################################################################
# 7. Inspect and Compare
##############################################################################

print("Number of unique original symptoms:", df['symptom'].nunique())
print("Number of unique final symptoms:", df['final_symptom'].nunique())

# Look at some examples side-by-side
print(df[['symptom','norm_symptom','tokens','tokens_corrected','corrected_symptom','final_symptom']].head(20))

##############################################################################
# 8. Visualization
##############################################################################

# a) Bar plot of final symptom frequencies (if not too large)
plt.figure(figsize=(10,6))
sns.countplot(y=df['final_symptom'], order=df['final_symptom'].value_counts().index)
plt.title("Symptom Frequency (After Cleaning)")
plt.xlabel("Count")
plt.ylabel("Symptom")
plt.tight_layout()
plt.show()

# b) Unique count comparison
unique_before = df['symptom'].nunique()
unique_after = df['final_symptom'].nunique()

plt.figure(figsize=(6,4))
plt.bar(["Before", "After"], [unique_before, unique_after], color=['red','green'])
plt.title("Unique Symptom Count: Before vs. After Cleaning")
plt.ylabel("Count of Unique Symptoms")
plt.show()
