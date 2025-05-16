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
embedding_dir=vit_base_patch16_384
# models:
# - vit_base_patch14_reg4_dinov2.lvd142m
# - hf-hub:BVRA/beit_base_patch16_384.in1k_ft_df24_384
# - hf-hub:BVRA/vit_base_patch16_224.in1k_ft_df24_224
# - hf-hub:BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384
# - hf-hub:BVRA/swin_base_patch4_window12_384.in1k_ft_df24_384
model_name=hf-hub:BVRA/vit_base_patch16_384.in1k_ft_fungitastic_384
resize_size=384

# run the Python script
fungiclef embed workflow \
    $project_dir/data/dataset/processed/${dataset_name}_serialized.parquet \
    $project_dir/data/embeddings/$embedding_dir/${dataset_name}_embed_v1.parquet \
    --cpu-count 4 \
    --batch-size 128 \
    --model-name $model_name \
