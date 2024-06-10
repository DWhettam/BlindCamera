#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=DCASE-resnet50
#SBATCH --mem=100G
#SBATCH --account=cosc014882
#SBATCH -t 06-24:00:00
module load CUDA


source /mnt/storage/home/qc19291/miniconda3/bin/activate DCASE

python -u EPIC_resnet.py --epochs 100 --ngpus 2 --window_size 10 --hop_length 5  --batch_size 128 --pretrained True --learning_rate 0.001 --label_type verb
