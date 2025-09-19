import torch
import argparse
import shutil
import os, sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
print(sys.path)

from utils.learning.combine_part import combine

def parse():
    parser = argparse.ArgumentParser(description='Combine Networks Together',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-n', '--net-name', type=Path, default='final_model_moe', help='Name of network')

    # Model Paths
    parser.add_argument('--classifier-cnn-path', type=Path, required=True)
    parser.add_argument('--brain-expert-path', type=Path, required=True)
    parser.add_argument('--knee-expert-path', type=Path, required=True)

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

    args = parser.parse_args()
    return args


def create_exp_dir(args):
    """Create experiment directory structure"""
    args.exp_dir = Path('../result') / args.net_name
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {args.exp_dir}")


if __name__ == '__main__':
    print("Start Combining ...")
    
    args = parse()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, training will be very slow!")
    else:
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        print(f"Using GPU {args.GPU_NUM}")
    
    # Create experiment directories
    create_exp_dir(args)
    
    # Save arguments
    import json
    with open(args.exp_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    # Start training
    try:
        combine(args)
        print("Combining completed successfully!")
    except KeyboardInterrupt:
        print("Combining interrupted by user")
    except Exception as e:
        print(f"Combining failed with error: {e}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()