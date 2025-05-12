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
model_name="multimodal-classifier-v1-epoch=24-val_loss=5.18.ckpt"
csv_filename="multimodal_linear_v1.csv" # prediction filename


# run the Python script
fungiclef multimodal predict \
    $project_dir/data/dataset/processed/test_serialized.parquet \
    $project_dir/data/embeddings/plantclef/test_embed.parquet \
    $project_dir/data/embeddings/bert/test_text_embed.parquet \
    $project_dir/model/classifier/$model_name \
    $project_dir/prediction/$csv_filename \
    --batch-size 64 \
    --cpu-count 4 \
    --embedding-col "embeddings" \
