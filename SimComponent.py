import pandas as pd
import re
import nltk
from spellchecker import SpellChecker
from fuzzywuzzy import process
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('punkt')

# 1. Read data
df = pd.read_csv("your_data.csv")

# 2. Basic normalization
def basic_normalization(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip()

df['norm_symptom'] = df['symptom'].apply(basic_normalization)

# 3. Tokenize and spell-check
spell = SpellChecker()

# Optionally add medical terms
medical_terms = ["hypertension", "arrhythmia", "bradycardia", "otitis", 
                 "arthritis", "bronchitis", "nausea", "cough", "diarrhea", 
                 "fever", "migraine", "appendicitis", "dizziness", "rash", ...]
for term in medical_terms:
    spell.word_frequency.add(term.lower())

def spell_correct_symptom(s):
    tokens = nltk.word_tokenize(s)
    corrected_tokens = [spell.correction(t) for t in tokens]
    return " ".join(corrected_tokens)

df['spellchecked_symptom'] = df['norm_symptom'].apply(spell_correct_symptom)

# 4. Fuzzy match to known symptom list
valid_symptoms = [
    "fever", "cough", "headache", "dizziness", "vomiting", "nausea",
    "fatigue", "chills", "sore throat", "shortness of breath",
    "chest pain", "diarrhea", "constipation", "rash", "body aches",
    "migraine", "arrhythmia", "hypertension"  # etc.
]

def fuzzy_match_symptom(s, valid_list=valid_symptoms, threshold=80):
    if not s:
        return ""
    best_match, score = process.extractOne(s, valid_list)
    return best_match if score >= threshold else s

df['final_symptom'] = df['spellchecked_symptom'].apply(fuzzy_match_symptom)

# 5. Check results
print("Before cleaning unique symptoms:", df['symptom'].nunique())
print("After cleaning unique symptoms:", df['final_symptom'].nunique())

# 6. Visualization
plt.figure(figsize=(8, 4))
unique_before = df['symptom'].nunique()
unique_after = df['final_symptom'].nunique()
plt.bar(["Before", "After"], [unique_before, unique_after], color=['red','green'])
plt.title("Unique Symptom Count: Before vs After")
plt.ylabel("Count of Unique Symptoms")
plt.show()

# Frequency distribution after cleaning
plt.figure(figsize=(10, 6))
sns.countplot(y=df['final_symptom'], order=df['final_symptom'].value_counts().index)
plt.title("Symptom Frequency (After Cleaning)")
plt.xlabel("Count")
plt.ylabel("Symptom")
plt.tight_layout()
plt.show()
