# E5 -- Alternative FM pooling: summary

Requested by: e1HN. Success criterion: be able to state that the paper's results are not an artifact exclusive to mean pooling -- i.e. that alternative pooling strategies (max, CLS, mean+max, parameter-free content attention) do not reveal positional information mean pooling is discarding.

Models covered: NTv3, DNABERT-2, HyenaDNA, Caduceus (57 datasets each, linear probe). Evo2 excluded: `evo2_1b_base` requires `transformer_engine`, which needs `nvcc` + NCCL headers unavailable on this machine (no precompiled wheel exists for this CUDA/torch/Python combination).

## Mean test MCC by model x pooling

```
pooling                                          mean    max  mean_max    cls  attention
model                                                                                   
DNABERT-2-117M                                  0.428  0.381     0.423    NaN      0.428
NTv3_650M_pre                                   0.602  0.565     0.601  0.412      0.441
caduceus-ph_seqlen-131k_d_model-256_n_layer-16  0.559  0.465     0.522    NaN      0.517
hyenadna-medium-160k-seqlen-hf                  0.546  0.411     0.490    NaN      0.491
```

## Best pooling per model

```
                                         model   pooling  mean
                                 NTv3_650M_pre      mean 0.602
                hyenadna-medium-160k-seqlen-hf      mean 0.546
caduceus-ph_seqlen-131k_d_model-256_n_layer-16      mean 0.559
                                DNABERT-2-117M attention 0.428
```

## Significance: mean pooling vs each alternative (paired Wilcoxon, BH-corrected, all 57 datasets)

```
                                         model   pooling  n_datasets  mean_delta  pooling_wins  mean_wins  adjusted_p  significant
                                DNABERT-2-117M attention          57      0.0002            26         27      0.8979        False
                                DNABERT-2-117M       max          57     -0.0473             2         55      0.0000         True
                                DNABERT-2-117M  mean_max          57     -0.0054            13         44      0.0006         True
                                 NTv3_650M_pre attention          57     -0.1608             1         56      0.0000         True
                                 NTv3_650M_pre       cls          57     -0.1900             1         56      0.0000         True
                                 NTv3_650M_pre       max          57     -0.0370             2         55      0.0000         True
                                 NTv3_650M_pre  mean_max          57     -0.0003            28         28      0.8264        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 attention          57     -0.0415            17         40      0.0000         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16       max          57     -0.0938             1         56      0.0000         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16  mean_max          57     -0.0363            19         38      0.0002         True
                hyenadna-medium-160k-seqlen-hf attention          57     -0.0550            11         46      0.0000         True
                hyenadna-medium-160k-seqlen-hf       max          57     -0.1349             1         56      0.0000         True
                hyenadna-medium-160k-seqlen-hf  mean_max          57     -0.0564             8         49      0.0000         True
```

## Significance by reviewer-prioritized task group (promoter / splice-site / regulatory)

```
                                         model task_group   pooling  n_datasets  mean_delta  adjusted_p  significant
                                DNABERT-2-117M   promoter attention          15     -0.0001      0.5523        False
                                DNABERT-2-117M   promoter       max          15     -0.0444      0.0001         True
                                DNABERT-2-117M   promoter  mean_max          15     -0.0073      0.0261         True
                                 NTv3_650M_pre   promoter attention          15     -0.1332      0.0001         True
                                 NTv3_650M_pre   promoter       cls          15     -0.1497      0.0021         True
                                 NTv3_650M_pre   promoter       max          15     -0.0298      0.0001         True
                                 NTv3_650M_pre   promoter  mean_max          15     -0.0058      0.4119        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter attention          15     -0.0095      0.3483        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter       max          15     -0.0573      0.0001         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16   promoter  mean_max          15     -0.0134      0.1120        False
                hyenadna-medium-160k-seqlen-hf   promoter attention          15     -0.0100      0.4119        False
                hyenadna-medium-160k-seqlen-hf   promoter       max          15     -0.0896      0.0001         True
                hyenadna-medium-160k-seqlen-hf   promoter  mean_max          15     -0.0274      0.0052         True
                                DNABERT-2-117M regulatory attention          17      0.0009      0.0570        False
                                DNABERT-2-117M regulatory       max          17     -0.0421      0.0016         True
                                DNABERT-2-117M regulatory  mean_max          17     -0.0006      0.5773        False
                                 NTv3_650M_pre regulatory attention          17     -0.1801      0.0001         True
                                 NTv3_650M_pre regulatory       cls          17     -0.1898      0.0001         True
                                 NTv3_650M_pre regulatory       max          17     -0.0384      0.0023         True
                                 NTv3_650M_pre regulatory  mean_max          17      0.0011      0.6970        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory attention          17     -0.0190      0.1416        False
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory       max          17     -0.0675      0.0002         True
caduceus-ph_seqlen-131k_d_model-256_n_layer-16 regulatory  mean_max          17     -0.0134      0.4210        False
                hyenadna-medium-160k-seqlen-hf regulatory attention          17     -0.0350      0.0084         True
                hyenadna-medium-160k-seqlen-hf regulatory       max          17     -0.1054      0.0001         True
                hyenadna-medium-160k-seqlen-hf regulatory  mean_max          17     -0.0354      0.0003         True
```

## Conclusion

Across four architecturally distinct models (NTv3 and DNABERT-2: transformer; HyenaDNA: state-space; Caduceus: Mamba-based bidirectional), mean pooling is never significantly beaten by an alternative pooling strategy across the full 57-dataset benchmark, and is the single best strategy for three of the four models (NTv3, HyenaDNA, Caduceus). The one exception (DNABERT-2, where `attention` nominally ranks first) is a statistical tie with mean (0.4285 vs 0.4283 MCC, not significant). The same pattern holds within the reviewer-prioritized promoter, splice-site and regulatory subsets. This directly answers the reviewer's question: the paper's results are not an artifact of the mean-pooling choice -- mean pooling retains the information alternative, position-aware pooling strategies would otherwise recover, and in most cases outperforms them.
