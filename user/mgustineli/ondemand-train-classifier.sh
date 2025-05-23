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
embedding_dir="plantclef" # plantclef or dinov2
model_name="${embedding_dir}-classifier-mixup-v1" # model name
model_type="mixup" # "linear" or "mixup"

# run the Python script
fungiclef prediction train \
    $project_data_dir/dataset/processed/${train_parquet}.parquet \
    $project_data_dir/embeddings/$embedding_dir/${train_embed}.parquet \
    $project_data_dir/dataset/processed/${val_parquet}.parquet \
    $project_data_dir/embeddings/$embedding_dir/${val_embed}.parquet \
    --model-type $model_type \
    --cpu-count 4 \
    --batch-size  256 \
    --max-epochs 10 \
    --learning-rate 0.0005 \
    --output-model-path "model" \
    --model-name $model_name \
    --embedding-col "embeddings" \
    --early-stopping-patience 3 \
