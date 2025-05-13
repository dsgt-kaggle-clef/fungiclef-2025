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
# models:
# - vit_base_patch14_reg4_dinov2.lvd142m
# - hf-hub:BVRA/beit_base_patch16_384.in1k_ft_df24_384
# - hf-hub:BVRA/vit_base_patch16_224.in1k_ft_df24_224
embedding_dir=vit_base_patch16_224
model_type="linear" # "linear" or "mixup"
model_name="vit-base-${model_type}-v1" # model name

# run the Python script
fungiclef prediction train \
    $project_data_dir/dataset/processed/${train_parquet}.parquet \
    $project_data_dir/embeddings/$embedding_dir/${train_embed}_v1.parquet \
    $project_data_dir/dataset/processed/${val_parquet}.parquet \
    $project_data_dir/embeddings/$embedding_dir/${val_embed}_v1.parquet \
    --model-type $model_type \
    --cpu-count 4 \
    --batch-size  256 \
    --max-epochs 100 \
    --learning-rate 0.0005 \
    --output-model-path "model" \
    --model-name $model_name \
    --embedding-col "embeddings" \
    --early-stopping-patience 3 \
