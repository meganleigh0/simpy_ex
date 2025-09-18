import pandas as pd

# Load the CSV
DATA = pd.read_csv("clustering_data_source.csv")

# Ensure tokens are always strings
DATA["token"] = DATA["token"].fillna("").astype(str)

# Step 1: Rebuild sentences (group by did + sentence boundaries)
sentences = (
    DATA.sort_values(["did", "sentence_beg", "place"])
        .groupby(["did", "sentence_beg", "sentence_end"])
        .agg({"token": lambda x: " ".join(x)})
        .reset_index()
)

# Step 2: Rebuild full text per did (all sentences joined)
full_sentences = (
    sentences.groupby("did")["token"]
             .apply(lambda x: " ".join(x))
             .reset_index()
             .rename(columns={"token": "sentence"})
)

# Show result
for i, row in full_sentences.iterrows():
    print(f"{row['did']}: {row['sentence']}")