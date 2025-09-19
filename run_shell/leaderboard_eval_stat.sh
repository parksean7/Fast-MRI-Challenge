#!/bin/bash

# Enhanced leaderboard evaluation with detailed statistics and per-file SSIM values
python run/leaderboard_eval_stat.py \
  -lp '../Data/leaderboard' \
  -yp '../result/brain_expert/phase_5/reconstructions_leaderboard' \
  -o '../result/brain_expert/phase_5/leaderboard_statistics.txt' \
  --detailed \
  --save-per-file \
  --per-file-dir '../result/brain_expert/phase_5'

# Alternative: Basic statistics without detailed output
# python run/leaderboard_eval_stat.py \
#   -lp '../Data/leaderboard' \
#   -yp '../result/varnet_with_augment_250719/reconstructions_leaderboard'

# Alternative: Save statistics to file without detailed output
# python run/leaderboard_eval_stat.py \
#   -lp '../Data/leaderboard' \
#   -yp '../result/varnet_with_augment_250719/reconstructions_leaderboard' \
#   -o 'leaderboard_stats.txt'

# Alternative: Save only per-file SSIM values without other detailed output
# python run/leaderboard_eval_stat.py \
#   -lp '../Data/leaderboard' \
#   -yp '../result/varnet_with_augment_250719/reconstructions_leaderboard' \
#   --save-per-file \
#   --per-file-dir 'ssim_results'