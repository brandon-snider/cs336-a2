#! /bin/bash

# Profile mixed precision training vs full precision training

# Define model sizes and sequence lengths
SIZES=("sm" "md" "lg" "xl" "2.7b")
# SEQ_LENS=(128 256 512 1024)
SEQ_LENS=(512)
OUT_DIR="out/profiling/mixed"

for size in "${SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        uv run nsys profile \
            -o ${OUT_DIR}/bf16-${size}-${seq_len} \
            python -m cs336_systems.profile \
                --size ${size} \
                --seq-len ${seq_len} \
                --mode forward_backward \
                --mixed || echo "Error in bf16 forward-backward for size=${size}, seq_len=${seq_len}, continuing..."
    done
done

for size in "${SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        uv run nsys profile \
            -o ${OUT_DIR}/fp32-f-${size}-${seq_len} \
            python -m cs336_systems.profile \
                --size ${size} \
                --seq-len ${seq_len} \
                --mode forward_backward || echo "Error in fp32 forward-backward for size=${size}, seq_len=${seq_len}, continuing..."
    done
done