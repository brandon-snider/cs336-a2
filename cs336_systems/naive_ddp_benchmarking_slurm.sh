#!/bin/bash

#SBATCH --job-name=naive_ddp_benchmarking
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:03:00
#SBATCH --output=naive_ddp_benchmarking_%j.out
#SBATCH --error=naive_ddp_benchmarking_%j.err

uv run -m cs336_systems.ddp_benchmarking --batch-sizes 2 4 8 16 32 --seq-lens 128