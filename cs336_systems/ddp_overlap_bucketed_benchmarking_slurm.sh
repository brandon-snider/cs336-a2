#!/bin/bash

#SBATCH --job-name=ddp_overlap_bucketed_benchmarking
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:05:00
#SBATCH --output=ddp_overlap_bucketed_benchmarking_%j.out
#SBATCH --error=ddp_overlap_bucketed_benchmarking_%j.err

uv run -m cs336_systems.ddp_benchmarking --overlap-bucketed --bucket-sizes-mb 1000 100 10 1 --batch-sizes 16 --seq-lens 128