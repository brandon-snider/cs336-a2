#!/bin/bash

#SBATCH --job-name=benchmark_all_reduce
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=00:03:00
#SBATCH --output=benchmark_all_reduce_%j.out
#SBATCH --error=benchmark_all_reduce_%j.err

uv run -m cs336_systems.benchmark_all_reduce --world-sizes 2 4 6