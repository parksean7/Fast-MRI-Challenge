#!/bin/bash
# Train Script for Brain Expert (SAGVarNET)
# Only Feed Brain Data (--data-anatomy-mode 1)
# Phase 1 (15 epoch): warmup
# Phase 2 (15 epoch)

python run/train_expert.py \
  -b 1 \
  -e 7 \
  -r 400 \
  -n "brain_expert" \
  --resume '../result/brain_expert/phase_2/model_epoch_7.pt' \
  -t '../Data/train/' \
  -v '../Data/val/' \
  --data-anatomy-mode 1 \
  --no-validation \
  --train-phase 'phase_2_resume' \
  --sens-chans 4 \
  --sens-pools 4 \
  --num-cascades 8 \
  --chans 20 \
  --pools 4 \
  --n-buffer 4 \
  --use-checkpoint \
  --lr 2.31e-4 \
  --optim-type 'AdamW' \
  --weight-decay 1e-6 \
  --scheduler-type 'cosine_annealing' \
  --eta-min 1.5e-4 \
  --accumulation-steps 2 \
  --loss-type 'SSIM_L1' \
  --ssim-weight 0.84 \
  --l1-weight 0.16 \
  --seed 42 \
  \
  --mask-mode augment \
  --mask-types-prob 0.45 0.0 0.05 0.45 0.05 \
  --mask-acc-probs "4:0.5 8:0.5" \
  --mask-aug-schedule exp \
  --mask-aug-start-prob 0.279 \
  --mask-aug-end-prob 0.3 \
  --mask-aug-delay 0 \
  \
  --mr-aug-on \
  --mr-aug-start-prob 0.279 \
  --mr-aug-end-prob 0.3 \
  --mr-aug-schedule exp \
  --mr-aug-delay 0 \
  \
  --mr-aug-weight-fliph 1.0 \
  --mr-aug-weight-scaling 0.5 \
  --mr-aug-weight-noise 0.5 \
  --mr-aug-weight-gibbs 0.7 \
  --mr-aug-weight-brightness 0.5 \
  --mr-aug-weight-contrast 1.0 \
  --mr-aug-weight-biasfield 0.7 \
  \
  --mr-aug-max-scaling 0.1 \
  \
  --mr-aug-noise-type rician \
  --mr-aug-noise-std-min 0.002 \
  --mr-aug-noise-std-max 0.02 \
  \
  --mr-aug-gibbs-truncation-min 0.75 \
  --mr-aug-gibbs-truncation-max 0.95 \
  --mr-aug-gibbs-type circular \
  --mr-aug-gibbs-anisotropy 0.3 \
  \
  --mr-aug-max-brightness 0.1 \
  --mr-aug-max-contrast 0.2 \
  \
  --mr-aug-biasfield-order 3 \
  --mr-aug-biasfield-min 0.9 \
  --mr-aug-biasfield-max 1.1 \
  --mr-aug-bias-stripe-prob 0.6 \
  --mr-aug-bias-stripe-freq-min 8.0 \
  --mr-aug-bias-stripe-freq-max 25.0 \
  --mr-aug-bias-stripe-amp-min 0.008 \
  --mr-aug-bias-stripe-amp-max 0.025