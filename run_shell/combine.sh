
python run/combine.py \
  -n "final_model_moe" \
  --classifier-cnn-path '../result/anatomy_classifier_cnn/phase_2/model_epoch_1.pt' \
  --brain-expert-path '../result/brain_expert/phase_5/model_epoch_4.pt' \
  --knee-expert-path '../result/knee_expert/phase_6/model_epoch_2.pt' \
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