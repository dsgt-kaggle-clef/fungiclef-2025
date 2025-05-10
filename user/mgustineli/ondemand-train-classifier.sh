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
train_parquet="train_serialized"
train_embed="train_embed"
val_parquet="val_serialized"
val_embed="val_embed"

# run the Python script
fungiclef prediction train \
    $project_data_dir/dataset/processed/${train_parquet}.parquet \
    $project_data_dir/embeddings/plantclef/${train_embed}.parquet \
    $project_data_dir/dataset/processed/${val_parquet}.parquet \
    $project_data_dir/embeddings/plantclef/${val_embed}.parquet \
    --cpu-count 4 \
    --batch-size 64 \
    --max-epochs 10 \
    --learning-rate 0.001 \
    --output-model-path "model" \
