#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=DCASE-resnet50
#SBATCH --mem=100G
#SBATCH -t 24:00:00
module load CUDA


source /mnt/storage/home/qc19291/anaconda3/bin/activate BlindCamera

python -u EPIC_resnet.py --epochs 100 --ngpus 2 --window_size 10 --hop_length 5  --batch_size 128 --pretrained True --augment True --label_type verb > verb_fixed_len.out
