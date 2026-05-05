#!/bin/bash

echo "Waiting for Caduceus regression to complete..."

model_tag="caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

# Poll every 30 seconds until both FM results exist for both tasks
while true; do
    if [ ! -f results/regression_records.csv ]; then
        echo "[$(date '+%H:%M:%S')] Waiting for results file..."
        sleep 30
        continue
    fi

    result=$(conda run -n cgr_bench python3 << 'PYEOF' 2>/dev/null
import pandas as pd
df = pd.read_csv("results/regression_records.csv")
bRNA = df[df["dataset"] == "bulk_rna_expression"]
cage = df[df["dataset"] == "cage_prediction"]
if len(bRNA) > 0 and len(cage) > 0:
    model_col = "fm_caduceus-ph_seqlen-131k_d_model-256_n_layer-16_R2"
    if model_col in df.columns:
        b_val = bRNA[model_col].values[0]
        c_val = cage[model_col].values[0]
        if pd.notna(b_val) and pd.notna(c_val):
            print("READY")
PYEOF
    )

    if [ "$result" = "READY" ]; then
        echo "✓ Caduceus regression complete!"
        break
    fi
    echo "[$(date '+%H:%M:%S')] Still waiting for Caduceus to finish..."
    sleep 30
done

echo ""
echo "Launching Evo2 regression..."
nohup conda run -n cgr_bench python3 -u \
    src/scripts/regression/train_lra_regression.py \
    --model evo2_1b_base \
    --tasks cage_prediction bulk_rna_expression \
    --k-values 4 5 6 \
    --fm-batch-size 8 \
    > logs/regression_evo2.log 2>&1 &

RPID=$!
echo "Evo2 regression PID: $RPID"
sleep 2
ps -p $RPID > /dev/null && echo "✓ Evo2 regression started" || echo "✗ Evo2 regression failed"

echo ""
echo "Waiting for Evo2 regression to complete..."
wait $RPID 2>/dev/null || true
echo "✓ Evo2 regression done"

echo ""
echo "Launching Evo2 decomposition (k=6 only)..."
nohup /home/oem/miniconda3/envs/cgr_bench/bin/python3 -u \
    src/scripts/decomposition/train_decomposition.py \
    --model evo2_1b_base \
    --data-root /data/genomic_bench/dna_foundation_benchmark/ \
    --k-values 6 \
    --no-cleanup \
    > logs/decomp_evo2_k6.log 2>&1 &

DPID=$!
echo "Evo2 decomposition PID: $DPID"
sleep 2
ps -p $DPID > /dev/null && echo "✓ Evo2 decomposition started" || echo "✗ Evo2 decomposition failed"

echo ""
echo "Waiting for all Evo2 experiments to complete..."
wait $DPID 2>/dev/null || true
echo "✓ Evo2 decomposition done"

echo ""
echo "All Evo2 experiments complete!"
