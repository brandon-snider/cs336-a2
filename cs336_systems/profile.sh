#! /bin/bash

# How to run:
# - Run all: ./profile.sh
# - Flags:
#   --exclude-f: Exclude forward pass
#   --exclude-fb: Exclude forward-backward pass
#   --exclude-fbs: Exclude forward-backward-step pass

# Define model sizes and sequence lengths
SIZES=("small" "medium" "large" "xl" "2.7B")
SEQ_LENS=(128 256 512 1024)

# Parse command line arguments
EXCLUDE_F=false
EXCLUDE_FB=false
EXCLUDE_FBS=false

for arg in "$@"; do
    case $arg in
        --exclude-f)
            EXCLUDE_F=true
            ;;
        --exclude-fb)
            EXCLUDE_FB=true
            ;;
        --exclude-fbs)
            EXCLUDE_FBS=true
            ;;
    esac
done

# Forward only
if [ "$EXCLUDE_F" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/forward-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --sizes ${size} \
                    --seq-lens ${seq_len} || echo "Error in forward-only for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi

# Forward-backward
if [ "$EXCLUDE_FB" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/forward-backward-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --sizes ${size} \
                    --seq-lens ${seq_len} \
                    --include-backward || echo "Error in forward-backward for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi

# Forward-backward-step
if [ "$EXCLUDE_FBS" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/forward-backward-step-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --sizes ${size} \
                    --seq-lens ${seq_len} \
                    --include-backward \
                    --include-step || echo "Error in forward-backward-step for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi