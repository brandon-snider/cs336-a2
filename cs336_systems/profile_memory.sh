#! /bin/bash

# Profile memory usage of forward and train passes
# Note: this script uses the benchmark.py script, not profile.py

# Define model sizes and sequence lengths
SIZES=("2.7b")
SEQ_LENS=(128 256 512)

# Parse command-line arguments
mixed_arg=""
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --mixed)
      mixed_arg="--mixed"
      shift # past argument
      ;;
    *)    # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

for size in "${SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        uv run -m cs336_systems.benchmark \
            --size ${size} \
            --seq-len ${seq_len} \
            --mode forward \
            --memory \
            --steps 1 ${mixed_arg} || echo "Error in forward for size=${size}, seq_len=${seq_len}, continuing..."
    done
done

for size in "${SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        uv run -m cs336_systems.benchmark \
            --size ${size} \
            --seq-len ${seq_len} \
            --mode train \
            --memory \
            --steps 1 ${mixed_arg} || echo "Error in train for size=${size}, seq_len=${seq_len}, continuing..."
    done
done