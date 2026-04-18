# %% [markdown]
# ---
# ## 5. Decomposizione dell'embedding FM: componente composizionale vs contestuale
#
# L'embedding di ciascun FM viene decomposto in due sottospazi ortogonali tramite
# regressione Ridge (k=6):
#
# - **Proiezione** (`proj`): parte dell'embedding *spiegabile* dalla frequenza dei k-mer.
#   Rappresenta il contenuto puramente composizionale catturato dal FM.
# - **Residuo** (`resid`): parte *non spiegabile* dai k-mer.
#   Rappresenta il segnale contestuale/posizionale appreso durante il pretraining.
#
# Per ciascuna delle tre rappresentazioni (proiezione, residuo, full embedding)
# viene addestrato un RF separato e misurato il MCC sul test set.
#
# **Domanda**: il vantaggio del FM sul k-mer è spiegato dalla componente contestuale
# (residuo) o emerge dall'embedding come sistema integrato?

# %%
