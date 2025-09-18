import pandas as pd

# Read CSV
df = pd.read_csv("clustering_data_source.csv")

# Group by `did` and sentence boundaries, then join tokens in order
# We'll sort by `place` to keep token order correct
sentences = (
    df.sort_values(["did", "sentence_beg", "place"])
      .groupby(["did", "sentence_beg", "sentence_end"])
      .agg({"token": lambda x: " ".join(x)})
      .reset_index()
)

# Combine into a single text string per `did`
full_sentences = (
    sentences.groupby("did")["token"]
             .apply(lambda x: " ".join(x))
             .reset_index()
             .rename(columns={"token": "sentence"})
)

# Show result
for i, row in full_sentences.iterrows():
    print(f"{row['did']}: {row['sentence']}")