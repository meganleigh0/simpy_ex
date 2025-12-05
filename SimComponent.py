# --- FIXED LSP HANDLING (closest date <= LSP) ---
valid = ev.index[ev.index <= lsp_date]

if len(valid) == 0:
    lsp_effective = ev.index.max()   # fallback
else:
    lsp_effective = valid.max()      # closest date <= LSP

row_lsp = ev.loc[lsp_effective]

SPI_LSP = row_lsp["BCWP"] / row_lsp["BCWS"] if row_lsp["BCWS"] else np.nan
CPI_LSP = row_lsp["BCWP"] / row_lsp["ACWP"] if row_lsp["ACWP"] else np.nan