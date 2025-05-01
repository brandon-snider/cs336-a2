#!/bin/bash

#SBATCH --job-name=ddp_profiling
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:04:00
#SBATCH --output=ddp_profiling_%j.out
#SBATCH --error=ddp_profiling_%j.err

uv run nsys profile \
                -o out/profiling_ddp/naive \
                python -m cs336_systems.ddp_benchmarking \
                    --batch-sizes 16 \
                    --seq-lens 128

uv run nsys profile \
                -o out/profiling_ddp/flat \
                python -m cs336_systems.ddp_benchmarking \
                    --flat \
                    --batch-sizes 16 \
                    --seq-lens 128

uv run nsys profile \
                -o out/profiling_ddp/overlap_individual \
                python -m cs336_systems.ddp_benchmarking \
                    --overlap-individual \
                    --batch-sizes 16 \
                    --seq-lens 128