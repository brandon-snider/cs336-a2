#!/bin/bash

#SBATCH --job-name=profile_mixed
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=profile_mixed_%j.out
#SBATCH --error=profile_mixed_%j.err

./cs336_systems/profile_mixed.sh "$@"