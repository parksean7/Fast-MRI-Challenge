#!/bin/bash
# Train Script for Anatomy Classifier
# Phase 1 (8 epoch) yes val, no aug
# Phase 2 (4 epoch) no val, no aug

python run/train_classifier_cnn.py \
  -b 1 \
  -e 8 \
  -r 500 \
  -n "anatomy_classifier_cnn" \
  -t '../Data/train/' \
  -v '../Data/val/' \
  --train-phase 'phase_1' \
  --lr 1e-4 \
  --optim-type 'AdamW' \
  --weight-decay 1e-6 \
  --accumulation-steps 4 \
  --scheduler-type 'cosine_annealing' \
  --eta-min 1e-6 \
  --seed 42 \
  --mask-mode 'original' \
  --val-mask-mode 'original' && \
python run/train_classifier_cnn.py \
  -b 1 \
  -e 4 \
  -r 500 \
  -n "anatomy_classifier_cnn" \
  --resume '../result/anatomy_classifier_cnn/phase_1/model_epoch_7.pt' \
  -t '../Data/train/' \
  -v '../Data/val/' \
  --no-validation \
  --train-phase 'phase_2' \
  --lr 3e-6 \
  --optim-type 'AdamW' \
  --weight-decay 1e-6 \
  --accumulation-steps 2 \
  --scheduler-type 'cosine_annealing' \
  --eta-min 1e-6 \
  --seed 42 \
  --mask-mode 'original' \
  --val-mask-mode 'original'  