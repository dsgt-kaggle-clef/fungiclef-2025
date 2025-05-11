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
project_dir=/storage/coda1/p-dsgt_clef2025/0/shared/fungiclef
dataset_name=train # train, val, test

# run the Python script
fungiclef embed workflow \
    $project_dir/data/dataset/processed/${dataset_name}_augment_serialized_v2.parquet \
    $project_dir/data/embeddings/plantclef/${dataset_name}_augment_embed_v2.parquet \
    --cpu-count 4 \
    --batch-size 128 \
