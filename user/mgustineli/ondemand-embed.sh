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
dataset_name=train # train, val, test

# run the Python script
fungiclef embed workflow \
    $project_data_dir/dataset/processed/${dataset_name}_serialized.parquet \
    $project_data_dir/embeddings/${dataset_name}_embed.parquet \
    --cpu-count 4 \
    --batch-size 64 \
