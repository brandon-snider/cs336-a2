#! /bin/bash

# How to run:
# - Run all: ./benchmark.sh
# - Flags:
#   --exclude-f: Exclude forward pass
#   --exclude-fb: Exclude forward-backward pass

# Define model sizes and sequence lengths
SIZES=("small" "medium" "large" "xl" "2.7B")
SEQ_LENS=(128 256 512 1024)

# Parse command line arguments
EXCLUDE_F=false
EXCLUDE_FB=false

for arg in "$@"; do
    case $arg in
        --exclude-f)
            EXCLUDE_F=true
            ;;
        --exclude-fb)
            EXCLUDE_FB=true
            ;;
    esac
done

trap "echo 'Benchmark interrupted'; exit 1" INT

# Forward only
if [ "$EXCLUDE_F" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run python -m cs336_systems.benchmark_2 \
                --size ${size} \
                --seq-len ${seq_len} \
                --no-backward || echo "Error in forward-only for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi

# Forward-backward
if [ "$EXCLUDE_FB" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run python -m cs336_systems.benchmark_2 \
                --size ${size} \
                --seq-len ${seq_len} || echo "Error in forward-backward for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi
