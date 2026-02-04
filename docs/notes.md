# Dissertation notes & experimental setup

Approach
- We use VADER to create weak/silver labels on AG News headlines and descriptions. The supervised DistilBERT model is then trained on these pseudo-labeled samples.

Label mapping
- VADER compound <= -0.05 -> negative
- -0.05 < compound < 0.05 -> neutral
- compound >= 0.05 -> positive

Evaluation
- Use a held-out portion (default 10%) as validation during training.
- Report accuracy and macro-F1 (included in training script compute_metrics).

Reproducibility
- Training script accepts `--sample` for quick replication without long runs.
- Use `--epochs 1` and `--sample 2000` for demo runs; for dissertation runs increase dataset and epochs.

Limitations
- Weak labeling introduces noise; discuss potential methods to improve labels (human labeling, data augmentation, multi-lexicon ensembling, or use of a sentiment-labeled dataset such as IMDB or SST for transfer learning).

Next steps
- Add an evaluation script that compares supervised vs. VADER baseline on consistent test split and outputs confusion matrices, precision/recall, and examples of disagreement.
