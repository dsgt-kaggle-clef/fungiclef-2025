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
test_parquet="test_serialized"
test_embed="test_embed"
model_name="plantclef-fungi-classifier-epoch=05-val_loss=5.18.ckpt"


# run the Python script
fungiclef prediction predict \
    $project_dir/data/dataset/processed/${test_parquet}.parquet \
    $project_dir/data/embeddings/plantclef/${test_embed}.parquet \
    $project_dir/model/classifier/$model_name \
    $project_dir/prediction/plantclef_prediction_v1.csv \
    --cpu-count 4 \
    --batch-size 64 \
    --embedding-col "embeddings" \
