#!/bin/bash
#SBATCH  -J bert2bert
#SBATCH -c 16
#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -o ./err_log/%x_%j.out
#SBATCH -e ./err_log/%x_%j.err

enable_lmod
module load container_env pytorch-gpu/1.10.0
#module load container_env tensorflow-gpu/2.4.1


crun -p ~/envs/nlp python -u bert2bert_finetune.py \
