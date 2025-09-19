import argparse
from pathlib import Path
import os, sys

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
print(sys.path)

from utils.learning.test_part_router import forward
import time


def parse():
    parser = argparse.ArgumentParser(description='Test anatomyRouter on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-p', '--path-data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    parser.add_argument('--model-path-cnn', type=Path, required=True)

    # Data
    parser.add_argument("--input-key", type=str, default='kspace', help='Name of input key')
    parser.add_argument('--data-anatomy-mode', type=int, default=0, help='Data filtering mode (0=baseline, 1=brain only, 2=knee only)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    start_time = time.time()
    
    # Check if model checkpoint exists
    if not os.path.exists(args.model_path_cnn):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path_cnn}")
    print(f"Using anatomyClassifier model (1): {args.model_path_cnn}")
    
    # acc4
    args.data_path = args.path_data / "acc4"
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')