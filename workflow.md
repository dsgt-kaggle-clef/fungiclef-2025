# Pipline 
all example commands listed here assumes you're in the root directory with the following structure:
```
root/
├── clef/fungiclef-2025/  # the github repository
├── scratch/fungiclef/     
    ├── dataset           # directory for storing the 2025 dataset
    ├── processed         # directory for serialized images + metadata parquet
    ├── embeddings        # directory for storing the embedding + metadata parquet
    ├── model             # directory for storing the models
        ├── various directories             # directories inside /model to save each submitted iteration

```

## Setup venv 
if using PACE, run this in terminal, not in an interactive session
```bash
source clef/fungiclef-2025/fungiclef/scripts/utils/slurm-venv.sh
```

## Download Dataset

```bash
kaggle competitions download -c fungi-clef-2025 -p /scratch/fungiclef
```
navigate to scratch/fungiclef
```bash
unzip fungi-clef-2025.zip -d temp_fungi
```
```bash
mv temp_fungi dataset
rm -rf temp_fungi
```

## Preprocessing
### Fix end of file errors in images
Pillow/OpenCV has issues opening some images straight from kaggle due to missing end of file marker.
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/fix_end_of_file.py 
```

### Serialize Images
Training set
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/serialize.py --csv scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv --image-dir scratch/fungiclef/dataset/images/FungiTastic-FewShot/train/fullsize --output scratch/fungiclef/processed/train_serialized.parquet
```

Validation set
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/serialize.py --csv scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv --image-dir scratch/fungiclef/dataset/images/FungiTastic-FewShot/val/fullsize --output scratch/fungiclef/processed/val_serialized.parquet
```

Test set
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/serialize.py --csv scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv --image-dir scratch/fungiclef/dataset/images/FungiTastic-FewShot/test/fullsize --output scratch/fungiclef/processed/test_serialized.parquet
```

### Generate class_mapping.txt
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/class_mapping.py --metadata scratch/fungiclef/dataset/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv --output clef/fungiclef-2025/fungiclef/class_mapping.txt
```

## Obtain Embeddings
One set of example commands for DinoV2-base model, another set of commands for vit timm model (plantclef model)

DinoV2
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/processed/train_serialized.parquet --output scratch/fungiclef/embeddings/train_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6
```

```bash
python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/processed/val_serialized.parquet --output scratch/fungiclef/embeddings/val_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6
```

```bash
python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/processed/test_serialized.parquet --output scratch/fungiclef/embeddings/test_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6
```

ViT from timm
```bash
python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/processed/train_serialized.parquet --output scratch/fungiclef/embeddings/plantclef/train_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6 --model-name vit_base_patch14_reg4_dinov2.lvd142m
```

```bash
python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/processed/val_serialized.parquet --output scratch/fungiclef/embeddings/plantclef/val_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6 --model-name vit_base_patch14_reg4_dinov2.lvd142m
```

```bash
python clef/fungiclef-2025/fungiclef/preprocessing/embedding.py --input scratch/fungiclef/processed/test_serialized.parquet --output scratch/fungiclef/embeddings/plantclef/test_embeddings.parquet --model-name facebook/dinov2-base --batch-size 64 --num-workers 6 --model-name vit_base_patch14_reg4_dinov2.lvd142m
```

## Train the Model
One set of example commands for DinoV2-base model, another set of commands for vit timm model

DinoV2
```bash
python clef/fungiclef-2025/fungiclef/prediction/train.py --train scratch/fungiclef/embeddings/dinov2/train_embeddings.parquet --val scratch/fungiclef/embeddings/dinov2/val_embeddings.parquet --batch-size 64 --max-epochs 20 --output-dir scratch/fungiclef/model/base_model --learning-rate 1e-3
```
ViT from timm
```bash
python clef/fungiclef-2025/fungiclef/prediction/train.py --train scratch/fungiclef/embeddings/plantclef/train_embeddings.parquet --val scratch/fungiclef/embeddings/plantclef/val_embeddings.parquet --batch-size 64 --max-epochs 20 --output-dir scratch/fungiclef/model/plantclef --learning-rate 1e-3
```

## Generate Predictions
One set of example commands for DinoV2-base model, another set of commands for vit timm model

DinoV2
```bash
python clef/fungiclef-2025/fungiclef/prediction/predict.py --test scratch/fungiclef/embeddings/dinov2/test_embeddings.parquet --model scratch/fungiclef/model/base_model/base_fungi-classifier-epoch=05-val_loss=5.17.ckpt --output clef/fungiclef-2025/fungiclef/prediction/base_predictions.csv
```
ViT from timm
```bash
python clef/fungiclef-2025/fungiclef/prediction/predict.py --test scratch/fungiclef/embeddings/plantclef/test_embeddings.parquet --model scratch/fungiclef/model/plantclef/plantclef_fungi-classifier-epoch=04-val_loss=5.16.ckpt --output clef/fungiclef-2025/fungiclef/prediction/plantclef_predictions.csv
```
