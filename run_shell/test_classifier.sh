# Test For anatomyClassifier_Shape
# python run/test_classifier.py \
#   -b 1 \
#   -p '../Data/leaderboard' \
#   --classifier 'shape'

# # Test For anatomyClassifier_Intensity
# python run/test_classifier.py \
#   -b 1 \
#   -p '../Data/leaderboard' \
#   --classifier 'intensity'

# Test For anatomyClassifier_CNN
python run/test_classifier.py \
  -b 1 \
  -p '../Data/leaderboard' \
  --classifier 'cnn' \
  --model-path '../result/anatomy_classifier_cnn/extra_phase_1/model_epoch_6.pt' 