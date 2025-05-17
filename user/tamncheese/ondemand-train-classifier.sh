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
model_name="${embedding_dir}-classifier-linear-weightSampler-multiObjective" # model name
model_type="linear" # "linear" or "mixup"
multi_objective="True" # True or False for multi_objective loss, must use linear model_type for multi_objective (for now)

if [ "$multi_objective" = "True" ]; then
    multi_objective_flag="--multi-objective"
else
    multi_objective_flag="--no-multi-objective"
fi

# run the Python script
fungiclef prediction train \
    $project_data_dir/dataset/processed/${train_parquet}.parquet \
    $project_data_dir/embeddings/$embedding_dir/${train_embed}.parquet \
    $project_data_dir/dataset/processed/${val_parquet}.parquet \
    $project_data_dir/embeddings/$embedding_dir/${val_embed}.parquet \
    --model-type $model_type \
    --cpu-count 6 \
    --batch-size  256 \
    --max-epochs 50 \
    --learning-rate 0.0005 \
    --output-model-path "model" \
    --model-name $model_name \
    --embedding-col "embeddings" \
    --early-stopping-patience 3 \
    $multi_objective_flag\
