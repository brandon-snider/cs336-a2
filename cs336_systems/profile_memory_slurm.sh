#!/bin/bash

#SBATCH --job-name=profile_memory
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=profile_memory_%j.out
#SBATCH --error=profile_memory_%j.err

./cs336_systems/profile_memory.sh "$@"