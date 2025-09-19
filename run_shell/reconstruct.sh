
python run/reconstruct.py \
  -b 1 \
  -n 'final_model_moe' \
  -p '../Data/leaderboard' \
  --model-path '../result/final_model_moe/best_model.pt' \
  \
  --brain-sens-chans 4 \
  --brain-sens-pools 4 \
  --brain-num-cascades 8 \
  --brain-chans 20 \
  --brain-pools 4 \
  --brain-n-buffer 4 \
  \
  --knee-sens-chans 4 \
  --knee-sens-pools 4 \
  --knee-num-cascades 8 \
  --knee-chans 20 \
  --knee-pools 4 \
  --knee-n-buffer 4