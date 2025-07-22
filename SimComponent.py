# --------------- tighten threshold & POS filter -----------------
GOOD_POS   = {"NOUN", "ADJ", "VERB"}          # adjust as needed
MIN_FREQ   = 5
DIST_THRES = 0.18

vocab_counter = Counter()
for doc in nlp.pipe(df["trimmed"].astype(str), batch_size=500, n_process=-1):
    vocab_counter.update(
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop
           and tok.pos_ in GOOD_POS
    )

# … keep rest of pipeline identical but use DIST_THRES in AgglomerativeClustering