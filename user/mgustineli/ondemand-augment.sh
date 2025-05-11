#!/bin/bash
set -xe

# print system info
echo "Hostname: $(hostname)"
echo "Number of CPUs: $(nproc)"
echo "Available memory: $(free -h)"

# activate the environment
source ~/scratch/fungiclef/.venv/bin/activate

# check GPU availability
python -c "import torch; print(torch.cuda.is_available())"  # Check if PyTorch can access the GPU
nvidia-smi                                                  # Check GPU usage

# define paths
scratch_data_dir=$(realpath ~/scratch/fungiclef/data)
project_data_dir=/storage/coda1/p-dsgt_clef2025/0/shared/fungiclef/data

# run the Python script
fungiclef preprocessing augment \
    $project_data_dir/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv \
    $project_data_dir/dataset/images/FungiTastic-FewShot/train/fullsize \
    $project_data_dir/dataset/processed/train_augment_serialized_v2.parquet \
    --scale-factor 3 \
