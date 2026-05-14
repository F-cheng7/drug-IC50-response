# Training Commands

This file provides the complete training commands used in this project.

Please run all commands from the root directory of this repository.

Expected project structure:

```text
.
├── code/
│   ├── main.py
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── ...
├── datasets/
│   ├── single_drug.csv
│   ├── double_drug.csv
│   └── single_fewshot_filled_rounded.csv
├── pretrained/
│   └── bert-base-uncased/
├── cache/
├── checkpoints/
├── requirements.txt
├── README.md
└── commands.md
```

## 1. Single-drug IC50 Prediction

```bash
python code/main.py \
  --mode single \
  --csv_path datasets/single_drug.csv \
  --preprocessed_dir cache/cache_single \
  --text_model_name pretrained/bert-base-uncased \
  --max_atoms 40 \
  --max_text_len 96 \
  --text_cache_batch_size 32 \
  --train_ratio 0.8 \
  --batch_size 8 \
  --epochs 30 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --device cuda \
  --seed 42 \
  --ckpt_dir checkpoints/checkpoints_single_seed42 \
  --evaluate_every 1 \
  --patience 10 \
  --hidden_dim 128 \
  --egnn_layers 1 \
  --num_layers 2 \
  --n_heads 4 \
  --dropout 0.1 \
  --feature_dropout 0.1
```

## 2. Drug-combination Response Prediction

```bash
python code/main.py \
  --mode combo \
  --csv_path datasets/double_drug.csv \
  --preprocessed_dir cache/cache_combo \
  --text_model_name pretrained/bert-base-uncased \
  --max_atoms 40 \
  --max_text_len 96 \
  --text_cache_batch_size 32 \
  --train_ratio 0.8 \
  --batch_size 8 \
  --epochs 40 \
  --lr 3e-5 \
  --weight_decay 2e-4 \
  --num_workers 0 \
  --device cuda \
  --seed 42 \
  --ckpt_dir checkpoints/checkpoints_combo_seed42_scale7 \
  --evaluate_every 1 \
  --patience 12 \
  --hidden_dim 256 \
  --egnn_layers 2 \
  --num_layers 3 \
  --n_heads 4 \
  --dropout 0.12 \
  --feature_dropout 0.12 \
  --combo_label_transform signed_log1p \
  --combo_label_scale 7.0
```

## 3. Joint Training

```bash
python code/main.py \
  --mode joint \
  --joint_single_csv_path datasets/single_drug.csv \
  --joint_combo_csv_path datasets/double_drug.csv \
  --preprocessed_dir cache/cache_joint \
  --text_model_name pretrained/bert-base-uncased \
  --max_atoms 40 \
  --max_text_len 96 \
  --text_cache_batch_size 32 \
  --train_ratio 0.8 \
  --batch_size 8 \
  --epochs 40 \
  --lr 4e-5 \
  --weight_decay 2e-4 \
  --num_workers 0 \
  --device cuda \
  --seed 42 \
  --ckpt_dir checkpoints/checkpoints_joint_seed42_scale7 \
  --evaluate_every 1 \
  --patience 12 \
  --hidden_dim 256 \
  --egnn_layers 2 \
  --num_layers 3 \
  --n_heads 4 \
  --dropout 0.12 \
  --feature_dropout 0.12 \
  --combo_label_transform signed_log1p \
  --combo_label_scale 7.0
```

## 4. Few-shot Adaptation

Few-shot adaptation requires a pretrained single-drug model checkpoint.

Please first run the single-drug training command and make sure the following checkpoint exists:

```text
checkpoints/checkpoints_single_seed42/single_best_model.pt
```

### 4.1 32-shot Adaptation

```bash
python code/main.py \
  --mode single \
  --csv_path datasets/single_drug.csv \
  --few_shot_csv_path datasets/single_fewshot_filled_rounded.csv \
  --preprocessed_dir cache/cache_single \
  --text_model_name pretrained/bert-base-uncased \
  --max_atoms 40 \
  --max_text_len 96 \
  --text_cache_batch_size 32 \
  --train_ratio 0.8 \
  --batch_size 8 \
  --epochs 0 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --device cuda \
  --seed 42 \
  --ckpt_dir checkpoints/checkpoints_fewshot_seed42_shot32 \
  --evaluate_every 1 \
  --patience 10 \
  --resume_from checkpoints/checkpoints_single_seed42/single_best_model.pt \
  --hidden_dim 128 \
  --egnn_layers 1 \
  --num_layers 2 \
  --n_heads 4 \
  --dropout 0.1 \
  --feature_dropout 0.1 \
  --few_shot_epochs 30 \
  --few_shot_lr 5e-4 \
  --few_shot_batch_size 8 \
  --few_shot_support_ratio 0.25196850393700787
```

### 4.2 101-shot Adaptation

```bash
python code/main.py \
  --mode single \
  --csv_path datasets/single_drug.csv \
  --few_shot_csv_path datasets/single_fewshot_filled_rounded.csv \
  --preprocessed_dir cache/cache_single \
  --text_model_name pretrained/bert-base-uncased \
  --max_atoms 40 \
  --max_text_len 96 \
  --text_cache_batch_size 32 \
  --train_ratio 0.8 \
  --batch_size 8 \
  --epochs 0 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --device cuda \
  --seed 42 \
  --ckpt_dir checkpoints/checkpoints_fewshot_seed42 \
  --evaluate_every 1 \
  --patience 10 \
  --resume_from checkpoints/checkpoints_single_seed42/single_best_model.pt \
  --hidden_dim 128 \
  --egnn_layers 1 \
  --num_layers 2 \
  --n_heads 4 \
  --dropout 0.1 \
  --feature_dropout 0.1 \
  --few_shot_epochs 30 \
  --few_shot_lr 5e-4 \
  --few_shot_batch_size 8 \
  --few_shot_support_ratio 0.8
```

## 5. Notes

- All commands should be executed from the root directory of this repository.
- The datasets should be downloaded from Google Drive and placed under `datasets/`.
- The pretrained BERT model should be placed under `pretrained/bert-base-uncased/`.
- Training caches will be saved under `cache/`.
- Model checkpoints and prediction outputs will be saved under `checkpoints/`.
- If you use a CPU-only environment, replace `--device cuda` with `--device cpu`.
