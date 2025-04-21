#!/bin/bash

#SBATCH --job-name=profile
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err

./cs336_systems/profile.sh "$@"