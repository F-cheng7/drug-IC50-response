# drug-IC50-response

## 1. Project Overview

This repository provides the implementation of a multimodal drug response prediction framework.

The model supports the following tasks:

- Single-drug IC50 prediction
- Drug-combination response prediction
- Joint training of single-drug and drug-combination tasks
- Few-shot adaptation

The framework integrates molecular structure information, drug textual descriptions, cell-line transcriptomic features, and task-specific information for drug response prediction.

## 2. Environment Setup

Please install the required dependencies using:

```bash
pip install -r requirements.txt
```

## 3. Dataset Download

The datasets used in this project are available from Google Drive:

```text
https://drive.google.com/drive/folders/18sCdttV41dc3g7DMiUg7LOxzkQwqwe9j?usp=sharing
```

Please download the datasets and place them in your local dataset directory.

In the example commands, the dataset directory is:

```text
datasets/
```

Expected dataset files include:

```text
single_drug.csv
double_drug.csv
single_fewshot_filled_rounded.csv
```

If you use a different local path, please modify the corresponding dataset path in the training commands.

## 4. Pretrained BERT

This project uses `bert-base-uncased` as the drug text encoder.

You can download the pretrained BERT model from Hugging Face:

```text
https://huggingface.co/bert-base-uncased
```

Please place the downloaded BERT model under:

```text
pretrained/bert-base-uncased/
```

The expected structure is:

```text
pretrained/
└── bert-base-uncased/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

If you place the BERT model in another directory, please modify the `--text_model_name` argument accordingly.

## 5. Training Examples

Only core command examples are provided here.  
For complete hyperparameter settings, please refer to `commands.md`.

### Single-drug IC50 Prediction

```bash
python code/main.py \
  --mode single \
  --csv_path datasets/single_drug.csv \
  --preprocessed_dir cache/cache_single \
  --text_model_name pretrained/bert-base-uncased \
  --batch_size 8 \
  --epochs 30 \
  --lr 5e-5 \
  --device cuda \
  --ckpt_dir checkpoints/checkpoints_single_seed42
```

### Drug-combination Response Prediction

```bash
python code/main.py \
  --mode combo \
  --csv_path datasets/double_drug.csv \
  --preprocessed_dir cache/cache_combo \
  --text_model_name pretrained/bert-base-uncased \
  --batch_size 8 \
  --epochs 40 \
  --lr 3e-5 \
  --device cuda \
  --ckpt_dir checkpoints/checkpoints_combo_seed42_scale7 \
  --combo_label_transform signed_log1p \
  --combo_label_scale 7.0
```

### Joint Training

```bash
python code/main.py \
  --mode joint \
  --joint_single_csv_path datasets/single_drug.csv \
  --joint_combo_csv_path datasets/double_drug.csv \
  --preprocessed_dir cache/cache_joint \
  --text_model_name pretrained/bert-base-uncased \
  --batch_size 8 \
  --epochs 40 \
  --lr 4e-5 \
  --device cuda \
  --ckpt_dir checkpoints/checkpoints_joint_seed42_scale7 \
  --combo_label_transform signed_log1p \
  --combo_label_scale 7.0
```

### Few-shot Adaptation

Before few-shot adaptation, please first train the single-drug model and prepare the pretrained checkpoint.

Expected checkpoint path:

```text
checkpoints/checkpoints_single_seed42/single_best_model.pt
```

Example command for 32-shot adaptation:

```bash
python code/main.py \
  --mode single \
  --csv_path datasets/single_drug.csv \
  --few_shot_csv_path data/single_fewshot_filled_rounded.csv \
  --preprocessed_dir cache/cache_single \
  --text_model_name pretrained/bert-base-uncased \
  --epochs 0 \
  --device cuda \
  --ckpt_dir checkpoints/checkpoints_fewshot_seed42_shot32 \
  --resume_from checkpoints/checkpoints_single_seed42/single_best_model.pt \
  --few_shot_epochs 30 \
  --few_shot_lr 5e-4 \
  --few_shot_batch_size 8 \
  --few_shot_support_ratio 0.25196850393700787
```

## 6. Outputs

Training outputs are saved to the directory specified by `--ckpt_dir`.

Example output directories:

```text
checkpoints/checkpoints_single_seed42
checkpoints/checkpoints_combo_seed42_scale7
checkpoints/checkpoints_joint_seed42_scale7
checkpoints/checkpoints_fewshot_seed42_shot32
```

Typical outputs include:

```text
model checkpoints
training logs
prediction results
evaluation metrics
few-shot adaptation results
```

Cache files generated during preprocessing are saved under the directory specified by `--preprocessed_dir`, for example:

```text
cache/cache_single
cache/cache_combo
cache/cache_joint
```
