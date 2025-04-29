#!/bin/bash

#SBATCH --job-name=opt_sharding_memory
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:04:00
#SBATCH --output=out/slurm/opt_sharding_memory_%j.out
#SBATCH --error=out/slurm/opt_sharding_memory_%j.err

uv run -m cs336_systems.ddp_benchmarking --batch-sizes 16 32 --seq-lens 128 --shard-optimizer --report-memory
uv run -m cs336_systems.ddp_benchmarking --batch-sizes 16 32 --seq-lens 128 --report-memory