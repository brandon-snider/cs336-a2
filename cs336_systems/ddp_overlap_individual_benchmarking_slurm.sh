#!/bin/bash

#SBATCH --job-name=ddp_overlap_individual_benchmarking
#SBATCH --partition=a2
#SBATCH --qos=a2-qos
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=00:01:30
#SBATCH --output=ddp_overlap_individual_benchmarking_%j.out
#SBATCH --error=ddp_overlap_individual_benchmarking_%j.err

uv run -m cs336_systems.naive_ddp_benchmarking --overlap-individual --batch-sizes 16 --seq-lens 128