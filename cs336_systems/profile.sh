#! /bin/bash

# How to run:
# - Run all: ./cs336_systems/profile.sh
# - Flags:
#   --exclude-f: Exclude forward pass
#   --exclude-fb: Exclude forward-backward pass
#   --exclude-fbs: Exclude forward-backward-step pass
#   --exclude-sdpa: Exclude scaled_dot_product_attention profiling

# Define model sizes and sequence lengths
SIZES=("sm" "md" "lg" "xl" "2.7b")
SEQ_LENS=(128 256 512 1024)

# Parse command line arguments
EXCLUDE_F=false
EXCLUDE_FB=false
EXCLUDE_FBS=false
EXCLUDE_SDPA=false

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
        --exclude-sdpa)
            EXCLUDE_SDPA=true
            ;;
    esac
done

# Forward only
if [ "$EXCLUDE_F" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/f-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --size ${size} \
                    --seq-len ${seq_len} \
                    --mode forward || echo "Error in forward-only for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi

# Forward-backward
if [ "$EXCLUDE_FB" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/fb-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --size ${size} \
                    --seq-len ${seq_len} \
                    --mode forward_backward || echo "Error in forward-backward for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi

# Forward-backward-step
if [ "$EXCLUDE_FBS" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/fbs-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --size ${size} \
                    --seq-len ${seq_len} \
                    --mode train || echo "Error in forward-backward-step for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi

# Scaled_dot_product_attention
if [ "$EXCLUDE_SDPA" = false ]; then
    for size in "${SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            uv run nsys profile \
                -o out/profiling/sdpa-${size}-${seq_len} \
                python -m cs336_systems.profile \
                    --size ${size} \
                    --seq-len ${seq_len} \
                    --mode forward \
                    --annotate-sdpa || echo "Error in scaled_dot_product_attention for size=${size}, seq_len=${seq_len}, continuing..."
        done
    done
fi
