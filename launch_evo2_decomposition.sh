#!/bin/bash

echo "Waiting for Caduceus decomposition to complete..."

# Poll every 10 seconds until the Caduceus decomposition process exits
while ps -p 1996721 > /dev/null 2>&1; do
    sleep 10
done

echo "✓ Caduceus decomposition complete"
echo ""
echo "Launching Evo2 decomposition (k=6 only)..."

/home/oem/miniconda3/envs/cgr_bench/bin/python3 -u \
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
echo "Waiting for Evo2 decomposition to complete..."
wait $DPID
echo "✓ Evo2 decomposition done"

echo ""
echo "All Evo2 experiments complete!"
