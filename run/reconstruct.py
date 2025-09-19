import argparse
from pathlib import Path
import os, sys

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
print(sys.path)

from utils.learning.test_part import forward
import time


def parse():
    parser = argparse.ArgumentParser(description='Test MoE SAGVarNet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net-name', type=Path, default='final_model_moe', help='Name of network')
    parser.add_argument('-p', '--path-data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    parser.add_argument('--model-path', type=Path, required=True, help='Path to MoE SAGVarNet model checkpoint file')

    # Brain Expert (SAG VarNet) specific parameters
    parser.add_argument('--brain-sens-chans', type=int, required=True)
    parser.add_argument('--brain-sens-pools', type=int, required=True)
    parser.add_argument('--brain-chans', type=int, required=True)
    parser.add_argument('--brain-pools', type=int, required=True)
    parser.add_argument('--brain-num-cascades', type=int, required=True)
    parser.add_argument('--brain-n-buffer', type=int, required=True)

    # Knee Expert (SAG VarNet) specific parameters
    parser.add_argument('--knee-sens-chans', type=int, required=True)
    parser.add_argument('--knee-sens-pools', type=int, required=True)
    parser.add_argument('--knee-chans', type=int, required=True)
    parser.add_argument('--knee-pools', type=int, required=True)
    parser.add_argument('--knee-num-cascades', type=int, required=True)
    parser.add_argument('--knee-n-buffer', type=int, required=True)

    # Data
    parser.add_argument("--input-key", type=str, default='kspace', help='Name of input key')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = Path('../result') / args.net_name

    start_time = time.time()
    
    # Check if model checkpoint exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    print(f"Using MoE_SAGVarNet model: {args.model_path}")
    
    # acc4
    args.data_path = args.path_data / "acc4"
    args.forward_dir = args.exp_dir / 'reconstructions_leaderboard' / "acc4"
    print(f"Processing acc4: {args.forward_dir}")
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = args.exp_dir / 'reconstructions_leaderboard' / "acc8"
    print(f"Processing acc8: {args.forward_dir}")
    forward(args)
    
    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3600 else print('Fail!')