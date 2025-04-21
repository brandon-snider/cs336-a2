#! /bin/bash

# How to run:
# - Run all: ./cs336_systems/benchmark.sh --output out/benchmark.txt
#   - Will benchmark forward-only and forward-backward for all model sizes and sequence lengths
#
# Use a custom number of warmup steps:
#   - ./cs336_systems/benchmark.sh --output out/benchmark.txt --warmup 10
#   - Default is 5
#
# Flags reference:
#   --exclude-f: Exclude forward pass
#   --exclude-fb: Exclude forward-backward pass
#   --output FILE: Send output to FILE (also shows in terminal)
#   --warmup N: Number of warmup iterations (default: 5)

# Define model sizes and sequence lengths
SIZES=("sm" "md" "lg" "xl" "2.7b")
SEQ_LENS=(128 256 512 1024)

# Parse command line arguments
EXCLUDE_F=false
EXCLUDE_FB=false
OUTPUT_FILE=""
WARMUP=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exclude-f)
            EXCLUDE_F=true
            shift
            ;;
        --exclude-fb)
            EXCLUDE_FB=true
            shift
            ;;
        --output)
            if [[ $# -gt 1 ]]; then
                OUTPUT_FILE="$2"
                shift 2
            else
                echo "Error: --output requires a filename"
                exit 1
            fi
            ;;
        --warmup)
            if [[ $# -gt 1 ]]; then
                WARMUP="$2"
                shift 2
            else
                echo "Error: --warmup requires a number"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up output redirection if requested
if [ -n "$OUTPUT_FILE" ]; then
    exec > >(tee -a "$OUTPUT_FILE") 2>&1
fi

trap "echo 'Benchmark interrupted'; exit 1" INT

# Forward only
# if [ "$EXCLUDE_F" = false ]; then
#     for size in "${SIZES[@]}"; do
#         for seq_len in "${SEQ_LENS[@]}"; do
#             uv run python -m cs336_systems.benchmark \
#                 --size ${size} \
#                 --seq-len ${seq_len} \
#                 --warmup ${WARMUP} \
#                 --no-backward || echo "Error in forward-only for size=${size}, seq_len=${seq_len}, continuing..."
#         done
#     done
# fi

# Forward-backward
if [ "$EXCLUDE_FB" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run python -m cs336_systems.benchmark \
                --size ${size} \
                --seq-len ${seq_len} \
                --warmup ${WARMUP} || echo "Error in forward-backward for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi