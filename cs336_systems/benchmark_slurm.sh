#!/bin/bash

#SBATCH --job-name=benchmark
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

./cs336_systems/benchmark.sh --output out/benchmark-warmup-5.txt --warmup 5
./cs336_systems/benchmark.sh --output out/benchmark-warmup-0.txt --warmup 0
./cs336_systems/benchmark.sh --output out/benchmark-warmup-10.txt --warmup 10