#!/bin/bash
# Train Script for Brain Expert (SAGVarNET)
# Only Feed Brain Data (--data-anatomy-mode 1)
# Phase 1 (15 epoch): warmup

python run/train_expert.py \
  -b 1 \
  -e 11 \
  -r 400 \
  -n "brain_expert" \
  --resume '../result/brain_expert/phase_1/model_epoch_3.pt' \
  -t '../Data/train/' \
  -v '../Data/val/' \
  --data-anatomy-mode 1 \
  --no-validation \
  --train-phase 'phase_1_resume' \
  --sens-chans 4 \
  --sens-pools 4 \
  --num-cascades 8 \
  --chans 20 \
  --pools 4 \
  --n-buffer 4 \
  --use-checkpoint \
  --save-interval 1 \
  --lr 4.1472e-5 \
  --optim-type 'AdamW' \
  --weight-decay 1e-6 \
  --scheduler-type 'step' \
  --scheduler-step 1 \
  --scheduler-gamma 1.2 \
  --accumulation-steps 2 \
  --loss-type 'SSIM_L1' \
  --ssim-weight 0.84 \
  --l1-weight 0.16 \
  --seed 42 \
  --mask-mode 'original'