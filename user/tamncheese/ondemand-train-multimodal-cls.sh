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
image_embed_dir="plantclef" # plantclef or dinov2
model_name="${image_embed_dir}-multipleLinearLayer-classifier-weightSampler-multiObjective_v5" # model name
multi_objective="True" # True or False for multi_objective loss, must use linear model_type for multi_objective (for now)

if [ "$multi_objective" = "True" ]; then
    multi_objective_flag="--multi-objective"
else
    multi_objective_flag="--no-multi-objective"
fi

# run the Python script
fungiclef multimodalmultiobjective train \
    $project_data_dir/dataset/processed/train_serialized.parquet \
    $project_data_dir/embeddings/plantclef/train_embed.parquet \
    $project_data_dir/embeddings/bert/train_text_embed.parquet \
    $project_data_dir/dataset/processed/val_serialized.parquet \
    $project_data_dir/embeddings/plantclef/val_embed.parquet \
    $project_data_dir/embeddings/bert/val_text_embed.parquet \
    --batch-size  256 \
    --cpu-count 4 \
    --max-epochs 100 \
    --learning-rate 0.0005 \
    --output-model-path "model" \
    --model-name $model_name \
    --embedding-col "embeddings" \
    --early-stopping-patience 3 \
    $multi_objective_flag\
