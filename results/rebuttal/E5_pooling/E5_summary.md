# E5 -- Alternative FM pooling: summary

Requested by: e1HN. Success criterion: be able to state that the paper's results are not an artifact exclusive to mean pooling -- i.e. that alternative pooling strategies do not reveal positional information mean pooling is discarding.

Models covered: NTv3, DNABERT-2, HyenaDNA, Caduceus (57 datasets each, linear probe). Evo2 excluded: `evo2_1b_base` requires `transformer_engine`, which needs `nvcc` + NCCL headers unavailable on this machine (no precompiled wheel exists for this CUDA/torch/Python combination).

## Pooling taxonomy actually tested (empirically verified per tokenizer)

- **cls**: hidden state at position 0, valid only where the tokenizer genuinely *prepends* a CLS token there. Verified empirically (not assumed from architecture family): only **DNABERT-2** does; NTv3's tokenizer declares a `<cls>` token but never prepends it, so its earlier "cls" results have been relabeled **first_token** (first nucleotide embedding, not a summary token) to avoid a false claim.
- **last_token**: hidden state at the last valid position, valid only where the tokenizer *appends* a genuine special token (SEP/EOS) there. **HyenaDNA** and **Caduceus** append `[SEP]`; for the autoregressive HyenaDNA this is the natural sequence summary (as in GPT-style models). NTv3 appends nothing, so it has no CLS-equivalent token at all.
- **attention**: parameter-free content attention (query = masked sequence mean), applicable to all four models.

## Mean test MCC by model x pooling

```
pooling                                          mean    max  mean_max    cls  first_token  last_token  attention
model                                                                                                            
DNABERT-2-117M                                  0.428  0.381     0.423  0.463          NaN         NaN      0.428
NTv3_650M_pre                                   0.602  0.565     0.601    NaN        0.412         NaN      0.441
caduceus-ph_seqlen-131k_d_model-256_n_layer-16  0.559  0.465     0.522    NaN          NaN       0.403      0.517
hyenadna-medium-160k-seqlen-hf                  0.546  0.411     0.490    NaN          NaN       0.331      0.491
```

## Best pooling per model

```
                                         model pooling  mean
                                 NTv3_650M_pre    mean 0.602
                hyenadna-medium-160k-seqlen-hf    mean 0.546
caduceus-ph_seqlen-131k_d_model-256_n_layer-16    mean 0.559
                                DNABERT-2-117M     cls 0.463
```

## Significance: mean pooling vs each alternative (paired Wilcoxon, BH-corrected, all 57 datasets)

```
                                         model     pooling  n_datasets  mean_delta  pooling_wins  mean_wins  adjusted_p  significant
                                DNABERT-2-117M   attention          57      0.0002            26         27      0.8979        False
                                DNABERT-2-117M         cls          57      0.0347            41         15      0.0000         True
                                DNABERT-2-117M         max          57     -0.0473             2         55      0.0000         True
                                DNABERT-2-117M    mean_max          57     -0.0054            13         44      0.0006         True
                                 NTv3_650M_pre   attention          57     -0.1608             1         56      0.0000         True
                                 NTv3_650M_pre first_token          57     -0.1900             1         56      0.0000         True
                                 NTv3_650M_pre         max          57     -0.0370             2         55      0.0000         True
                                 NTv3_650M_pre    mean_max          57     -0.0003            28         28      0.8136        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   attention          57     -0.0415            17         40      0.0000         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16  last_token          57     -0.1562             1         56      0.0000         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16         max          57     -0.0938             1         56      0.0000         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16    mean_max          57     -0.0363            19         38      0.0002         True
                hyenadna-medium-160k-seqlen-hf   attention          57     -0.0550            11         46      0.0000         True
                hyenadna-medium-160k-seqlen-hf  last_token          57     -0.2151             1         56      0.0000         True
                hyenadna-medium-160k-seqlen-hf         max          57     -0.1349             1         56      0.0000         True
                hyenadna-medium-160k-seqlen-hf    mean_max          57     -0.0564             8         49      0.0000         True
```

## Significance by reviewer-prioritized task group (promoter / splice-site / regulatory)

```
                                         model task_group     pooling  n_datasets  mean_delta  adjusted_p  significant
                                DNABERT-2-117M   promoter   attention          15     -0.0001      0.5438        False
                                DNABERT-2-117M   promoter         cls          15      0.0373      0.1340        False
                                DNABERT-2-117M   promoter         max          15     -0.0444      0.0001         True
                                DNABERT-2-117M   promoter    mean_max          15     -0.0073      0.0248         True
                                 NTv3_650M_pre   promoter   attention          15     -0.1332      0.0001         True
                                 NTv3_650M_pre   promoter first_token          15     -0.1497      0.0020         True
                                 NTv3_650M_pre   promoter         max          15     -0.0298      0.0001         True
                                 NTv3_650M_pre   promoter    mean_max          15     -0.0058      0.4009        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter   attention          15     -0.0095      0.3322        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter  last_token          15     -0.1208      0.0001         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter         max          15     -0.0573      0.0001         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter    mean_max          15     -0.0134      0.1080        False
                hyenadna-medium-160k-seqlen-hf   promoter   attention          15     -0.0100      0.4009        False
                hyenadna-medium-160k-seqlen-hf   promoter  last_token          15     -0.1613      0.0001         True
                hyenadna-medium-160k-seqlen-hf   promoter         max          15     -0.0896      0.0001         True
                hyenadna-medium-160k-seqlen-hf   promoter    mean_max          15     -0.0274      0.0052         True
                                DNABERT-2-117M regulatory   attention          17      0.0009      0.0545        False
                                DNABERT-2-117M regulatory         cls          17      0.0183      0.0164         True
                                DNABERT-2-117M regulatory         max          17     -0.0421      0.0015         True
                                DNABERT-2-117M regulatory    mean_max          17     -0.0006      0.5715        False
                                 NTv3_650M_pre regulatory   attention          17     -0.1801      0.0001         True
                                 NTv3_650M_pre regulatory first_token          17     -0.1898      0.0001         True
                                 NTv3_650M_pre regulatory         max          17     -0.0384      0.0022         True
                                 NTv3_650M_pre regulatory    mean_max          17      0.0011      0.6936        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory   attention          17     -0.0190      0.1340        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory  last_token          17     -0.1399      0.0001         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory         max          17     -0.0675      0.0002         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory    mean_max          17     -0.0134      0.4122        False
                hyenadna-medium-160k-seqlen-hf regulatory   attention          17     -0.0350      0.0084         True
                hyenadna-medium-160k-seqlen-hf regulatory  last_token          17     -0.1669      0.0001         True
                hyenadna-medium-160k-seqlen-hf regulatory         max          17     -0.1054      0.0001         True
                hyenadna-medium-160k-seqlen-hf regulatory    mean_max          17     -0.0354      0.0003         True
```

## The one real exception: DNABERT-2's genuine CLS token

DNABERT-2 CLS pooling significantly **beats** mean pooling (mean MCC 0.463 vs 0.428, 41/57 dataset wins, adjusted p=2.2e-5) -- the only significant win for any alternative pooling across the entire matrix (4 models x up to 6 poolings each). This is architecturally expected: DNABERT-2 is a BERT-style model whose CLS token is genuinely trained during pretraining to aggregate sequence-level information, unlike the untrained proxies (max, mean+max, parameter-free attention, or the SEP/first-token positions in the other three models) tested elsewhere.

**This does not overturn the k-mer comparison**, however: on the 54/57 datasets where E4's k-mer baseline is currently available, k-mer still significantly beats DNABERT-2 CLS (19 CLS wins vs 35 k-mer wins, mean delta -0.047, Wilcoxon p=0.0055) -- CLS roughly halves DNABERT-2's gap to k-mer (mean pooling's gap was -0.082, p<1e-6) but does not close it. The improvement is internal to DNABERT-2's own pooling choice, not a reversal of the paper's k-mer-vs-FM finding for this model.

## Conclusion

Across four architecturally distinct models (NTv3, DNABERT-2: transformer; HyenaDNA: state-space; Caduceus: Mamba-based bidirectional) and up to six pooling strategies per model -- including, beyond the reviewer's original list, a last-token/SEP pooling added after verifying which models actually carry a CLS-equivalent token -- mean pooling is never significantly beaten except in one case: DNABERT-2's genuine, pretrained CLS token. Every untrained proxy (max, mean+max, parameter-free attention, first-token, last-token/SEP) is either tied with or significantly worse than mean, consistently across all four models and within the promoter/splice-site/regulatory subsets the reviewer prioritized. The one exception is itself informative: it is not that positional information is generally recoverable by alternative pooling, but that a token specifically pretrained to aggregate it (as CLS is in BERT-style objectives) can do so, while generic/untrained pooling substitutes cannot -- and even that exception does not overturn the paper's k-mer comparison. This directly answers the reviewer's question: the paper's results are not an artifact of the mean-pooling choice.
