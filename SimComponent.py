import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from fuzzywuzzy import process

# 1. Load data
df = pd.read_csv("your_data.csv")

# 2. Inspect before cleaning
print(df.head())
print("Unique symptoms (before):", df['symptom'].nunique())

# 3. Define cleaning approach
common_misspellings = {
    "feverr": "fever", 
    "fevr": "fever",
    # ... add more ...
}

def clean_symptom_text(s):
    if pd.isna(s):
        return ""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = s.strip()
    if s in common_misspellings:
        s = common_misspellings[s]
    return s

valid_symptoms = ["fever", "cough", "headache", "dizziness", "vomiting", "nausea", 
                  "fatigue", "chills", "sore throat", "shortness of breath", 
                  "rash", "diarrhea", "chest pain", "body aches"]
def fuzzy_match_to_valid(s, valid_list=valid_symptoms, threshold=80):
    if s == "":
        return ""
    best_match, score = process.extractOne(s, valid_list)
    return best_match if score >= threshold else s

# 4. Apply cleaning and fuzzy matching
df['clean_symptom'] = df['symptom'].apply(clean_symptom_text)
df['final_symptom'] = df['clean_symptom'].apply(fuzzy_match_to_valid)

# 5. Inspect after cleaning
print("Unique symptoms (after):", df['final_symptom'].nunique())

# 6. Visualization

# a) Frequency bar chart
plt.figure(figsize=(10, 6))
sns.countplot(y=df['final_symptom'], order=df['final_symptom'].value_counts().index)
plt.title("Symptom Frequency (After Cleaning)")
plt.xlabel("Count")
plt.ylabel("Symptom")
plt.tight_layout()
plt.show()

# b) Before vs after unique counts
unique_before = df['symptom'].nunique()
unique_after = df['final_symptom'].nunique()

plt.figure(figsize=(6, 4))
plt.bar(["Before Cleaning", "After Cleaning"], [unique_before, unique_after], color=['red','green'])
plt.title("Comparison of Unique Symptom Counts")
plt.ylabel("Number of Unique Symptoms")
plt.show()
