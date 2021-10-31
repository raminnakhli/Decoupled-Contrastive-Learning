#!/bin/bash
#SBATCH --job-name=simclr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH -o %j.out
#SBATCH -e %j.out

DATASET=CIFAR10
BATCH_SIZE=128
LOSS="dclw"
TEMP=0.1

python train.py \
  --batch_size $BATCH_SIZE \
  --epochs 100 \
  --feature_dim 128 \
  --loss $LOSS \
  --temperature $TEMP

python test.py \
  --batch_size 64 \
  --epochs 100 \
  --model_path "results/128_${TEMP}_200_${BATCH_SIZE}_100_${LOSS}_model.pth"
