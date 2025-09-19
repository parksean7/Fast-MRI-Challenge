import torch
import argparse
import shutil
import os, sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
print(sys.path)

from utils.learning.train_part_expert import train
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train VarNet SAG on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-a', '--accumulation-steps', type=int, default=1, help='Gradient Accumulation Step Size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='moe_accnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    # SAG VarNet specific parameters
    parser.add_argument('--sens-chans', type=int, required=True)
    parser.add_argument('--sens-pools', type=int, required=True)
    parser.add_argument('--chans', type=int, required=True)
    parser.add_argument('--pools', type=int, required=True)
    parser.add_argument('--num-cascades', type=int, required=True)
    parser.add_argument('--n-buffer', type=int, required=True)
    parser.add_argument('--use-checkpoint', action='store_true', default=False)
     
    # Loss Funtion
    parser.add_argument('--loss-type', type=str, choices=['SSIM', 'SSIM_L1', 'EW_SSIM'], default='SSIM_L1', 
                       help='Loss Function Type: SSIM or SSIM + L1')
    parser.add_argument('--ssim-weight', type=float, default=0.8) # for SSIM_L1
    parser.add_argument('--l1-weight', type=float, default=0.2) # for SSIM_L1
    parser.add_argument('--ew_ssim_edge_method', type=str, default='hybrid') # for EW_SSIM
    parser.add_argument('--ew_ssim_edge_sigma', type=float, default=0.8) # for EW_SSIM
    parser.add_argument('--ew_ssim_use_pred_edges', action='store_true', default=False) # for EW_SSIM
    parser.add_argument('--ew_ssim_blend_ratio', type=float, default=0.8) # for EW_SSIM
    
    # Optimizer
    parser.add_argument('-l', '--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--optim-type', type=str, choices=['Adam', 'AdamW'], default=1e-4, help='Optimizer Tpe')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for AdamW optimizer')

    # Scheduler
    parser.add_argument('--scheduler-type', type=str, default='cosine_annealing')
    parser.add_argument('--eta-min', type=float, default=1e-6) # for consine-annealing only
    parser.add_argument('--scheduler-step', type=int, default=4) # for step only
    parser.add_argument('--scheduler-gamma', type=float, default=0.5) # for step only
    parser.add_argument('--scheduler-factor', type=float, default=0.5) # for reduce on plateau only
    parser.add_argument('--scheduler-patience', type=int, default=5) # for reduce on plateau only

    # Data parameters
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--data-anatomy-mode', type=int, default=0, help='Data filtering mode (0=baseline, 1=brain only, 2=knee only)')

    # Training mode arguments
    parser.add_argument('--no-validation', action='store_true', default=False,
                       help='Disable validation and use train+val datasets for training only')
    
    # Paths
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--train-phase', type=str, default=None)
    
    # Training mask augmentation arguments
    parser.add_argument('--mask-mode', type=str, choices=['original', 'augment'], default='original', 
                       help='Training mask mode: original (use given mask) or augment (use various masks)')
    parser.add_argument('--mask-types-prob', type=float, nargs=5, 
                       default=[1.0, 0.0, 0.0, 0.0, 0.0],
                       help='Training probabilities for 5 mask types: [random, equispaced, equispaced_fraction, magic, magic_fraction]')
    parser.add_argument('--mask-acc-probs', type=str, default='4:1.0',
                       help='Training acceleration probabilities in format "acc1:prob1 acc2:prob2 ..." (e.g., "4:0.4 6:0.1 8:0.5")')
    
    # Training mask augmentation scheduling arguments
    parser.add_argument('--mask-aug-schedule', type=str, choices=['constant', 'linear', 'exp'], default='constant',
                       help='Training mask augmentation probability scheduling type')
    parser.add_argument('--mask-aug-start-prob', type=float, default=0.0,
                       help='Training starting probability for mask augmentation (0.0-1.0)')
    parser.add_argument('--mask-aug-end-prob', type=float, default=1.0,
                       help='Training ending probability for mask augmentation (0.0-1.0)')
    parser.add_argument('--mask-aug-delay', type=int, default=0,
                       help='Training number of epochs before mask augmentation scheduling starts')
    
    # Validation mask augmentation arguments
    parser.add_argument('--val-mask-mode', type=str, choices=['original', 'augment'], default='original', 
                       help='Validation mask mode: original (use given mask) or augment (use various masks)')
    parser.add_argument('--val-mask-types-prob', type=float, nargs=5, 
                       default=[1.0, 0.0, 0.0, 0.0, 0.0],
                       help='Validation probabilities for 5 mask types: [random, equispaced, equispaced_fraction, magic, magic_fraction]')
    parser.add_argument('--val-mask-acc-probs', type=str, default='4:1.0',
                       help='Validation acceleration probabilities in format "acc1:prob1 acc2:prob2 ..." (e.g., "4:0.4 6:0.1 8:0.5")')
    
    # Validation mask augmentation scheduling arguments  
    parser.add_argument('--val-mask-aug-schedule', type=str, choices=['constant', 'linear', 'exp'], default='constant',
                       help='Validation mask augmentation probability scheduling type')
    parser.add_argument('--val-mask-aug-start-prob', type=float, default=0.0,
                       help='Validation starting probability for mask augmentation (0.0-1.0)')
    parser.add_argument('--val-mask-aug-end-prob', type=float, default=1.0,
                       help='Validation ending probability for mask augmentation (0.0-1.0)')
    parser.add_argument('--val-mask-aug-delay', type=int, default=0,
                       help='Validation number of epochs before mask augmentation scheduling starts')

    # Training MRAugment arguments
    parser.add_argument('--mr-aug-on', action='store_true', default=False,
                       help='Enable MRAugment data augmentation for training')
    parser.add_argument('--mr-aug-debug', action='store_true', default=False,
                       help='Enable debug prints for MRAugment pipeline')
    
    # Training scheduling arguments
    parser.add_argument('--mr-aug-schedule', type=str, choices=['constant', 'ramp', 'exp'], default='exp',
                       help='Training augmentation strength scheduling type')
    parser.add_argument('--mr-aug-delay', type=int, default=0,
                       help='Training number of epochs without augmentation at start')
    parser.add_argument('--mr-aug-start-prob', type=float, default=0.0,
                       help='Training starting augmentation strength [0-1]')
    parser.add_argument('--mr-aug-end-prob', type=float, default=0.5,
                       help='Training ending augmentation strength [0-1]')
    parser.add_argument('--mr-aug-exp-decay', type=float, default=5.0,
                       help='Training exponential decay coefficient for exp schedule')
    
    # Validation MRAugment arguments
    parser.add_argument('--val-mr-aug-on', action='store_true', default=False,
                       help='Enable MRAugment data augmentation for validation')
    parser.add_argument('--val-mr-aug-debug', action='store_true', default=False,
                       help='Enable debug prints for validation MRAugment pipeline')
    
    # Validation scheduling arguments
    parser.add_argument('--val-mr-aug-schedule', type=str, choices=['constant', 'ramp', 'exp'], default='exp',
                       help='Validation augmentation strength scheduling type')
    parser.add_argument('--val-mr-aug-delay', type=int, default=0,
                       help='Validation number of epochs without augmentation at start')
    parser.add_argument('--val-mr-aug-start-prob', type=float, default=0.0,
                       help='Validation starting augmentation strength [0-1]')
    parser.add_argument('--val-mr-aug-end-prob', type=float, default=0.5,
                       help='Validation ending augmentation strength [0-1]')
    parser.add_argument('--val-mr-aug-exp-decay', type=float, default=5.0,
                       help='Validation exponential decay coefficient for exp schedule')
    
    # Training technique probability weights (2024 winners: fliph/flipv/rot90=0.5, others=1.0)
    parser.add_argument('--mr-aug-weight-fliph', type=float, default=0.0,
                       help='Training probability weight for horizontal flip')
    parser.add_argument('--mr-aug-weight-flipv', type=float, default=0.0,
                       help='Training probability weight for vertical flip')
    parser.add_argument('--mr-aug-weight-rot90', type=float, default=0.0,
                       help='Training probability weight for 90-degree rotation (disabled by default to avoid dimension changes)')
    parser.add_argument('--mr-aug-weight-rotation', type=float, default=0.0,
                       help='Training probability weight for arbitrary rotation')
    parser.add_argument('--mr-aug-weight-translation', type=float, default=0.0,
                       help='Training probability weight for translation')
    parser.add_argument('--mr-aug-weight-scaling', type=float, default=0.0,
                       help='Training probability weight for scaling')
    parser.add_argument('--mr-aug-weight-shearing', type=float, default=0.0,
                       help='Training probability weight for shearing')
    
    # Training color augmentation probability weights
    parser.add_argument('--mr-aug-weight-brightness', type=float, default=0.0,
                       help='Training probability weight for brightness adjustment')
    parser.add_argument('--mr-aug-weight-contrast', type=float, default=0.0,
                       help='Training probability weight for contrast adjustment')
    
    # Training Gibbs ringing simulation probability weights  
    parser.add_argument('--mr-aug-weight-gibbs', type=float, default=0.0,
                       help='Training probability weight for Gibbs ringing simulation')
    
    # Training k-space noise injection probability weights
    parser.add_argument('--mr-aug-weight-noise', type=float, default=0.0,
                       help='Training probability weight for k-space noise injection')
    
    # Training coil sensitivity bias field probability weights  
    parser.add_argument('--mr-aug-weight-biasfield', type=float, default=0.0,
                       help='Training probability weight for coil sensitivity bias field')
    
    # Training streak artifact probability weights
    parser.add_argument('--mr-aug-weight-streaks', type=float, default=0.0,
                       help='Training probability weight for streak artifacts')
    
    # Training streak density parameters
    parser.add_argument('--mr-aug-streak-density-min', type=float, default=0.01,
                       help='Minimum density for streak artifacts')
    parser.add_argument('--mr-aug-streak-density-max', type=float, default=0.05,
                       help='Maximum density for streak artifacts')
    
    # Validation technique probability weights
    parser.add_argument('--val-mr-aug-weight-fliph', type=float, default=0.5,
                       help='Validation probability weight for horizontal flip')
    parser.add_argument('--val-mr-aug-weight-flipv', type=float, default=0.5,
                       help='Validation probability weight for vertical flip')
    parser.add_argument('--val-mr-aug-weight-rot90', type=float, default=0.0,
                       help='Validation probability weight for 90-degree rotation')
    parser.add_argument('--val-mr-aug-weight-rotation', type=float, default=0.5,
                       help='Validation probability weight for arbitrary rotation')
    parser.add_argument('--val-mr-aug-weight-translation', type=float, default=1.0,
                       help='Validation probability weight for translation')
    parser.add_argument('--val-mr-aug-weight-scaling', type=float, default=1.0,
                       help='Validation probability weight for scaling')
    parser.add_argument('--val-mr-aug-weight-shearing', type=float, default=1.0,
                       help='Validation probability weight for shearing')
    
    # Validation color augmentation probability weights
    parser.add_argument('--val-mr-aug-weight-brightness', type=float, default=0.5,
                       help='Validation probability weight for brightness adjustment')
    parser.add_argument('--val-mr-aug-weight-contrast', type=float, default=0.5,
                       help='Validation probability weight for contrast adjustment')
    
    # Validation Gibbs ringing simulation probability weights
    parser.add_argument('--val-mr-aug-weight-gibbs', type=float, default=0.6,
                       help='Validation probability weight for Gibbs ringing simulation')
    
    # Validation k-space noise injection probability weights
    parser.add_argument('--val-mr-aug-weight-noise', type=float, default=1.0,
                       help='Validation probability weight for k-space noise injection')
    
    # Validation coil sensitivity bias field probability weights
    parser.add_argument('--val-mr-aug-weight-biasfield', type=float, default=1.0,
                       help='Validation probability weight for coil sensitivity bias field')
    
    # Validation streak artifact probability weights
    parser.add_argument('--val-mr-aug-weight-streaks', type=float, default=0.0,
                       help='Validation probability weight for streak artifacts')
    
    # Validation streak density parameters
    parser.add_argument('--val-mr-aug-streak-density-min', type=float, default=0.01,
                       help='Validation minimum density for streak artifacts')
    parser.add_argument('--val-mr-aug-streak-density-max', type=float, default=0.05,
                       help='Validation maximum density for streak artifacts')
    
    # Training transformation limits (2024 winner defaults)
    parser.add_argument('--mr-aug-max-rotation', type=float, default=180.0,
                       help='Training maximum rotation angle in degrees')
    parser.add_argument('--mr-aug-max-scaling', type=float, default=0.25,
                       help='Training maximum scaling factor (0.25 = ±25 percent)')
    parser.add_argument('--mr-aug-max-translation-x', type=float, default=0.125,
                       help='Training maximum translation along x-axis as fraction of width')
    parser.add_argument('--mr-aug-max-translation-y', type=float, default=0.08,
                       help='Training maximum translation along y-axis as fraction of height')
    parser.add_argument('--mr-aug-max-shearing-x', type=float, default=15.0,
                       help='Training maximum shearing along x-axis in degrees')
    parser.add_argument('--mr-aug-max-shearing-y', type=float, default=15.0,
                       help='Training maximum shearing along y-axis in degrees')
    
    # Training color augmentation limits
    parser.add_argument('--mr-aug-max-brightness', type=float, default=0.2,
                       help='Training maximum brightness adjustment (±20% default)')
    parser.add_argument('--mr-aug-max-contrast', type=float, default=0.2,
                       help='Training maximum contrast adjustment (±20% default)')
    
    # Training Gibbs ringing simulation parameters
    parser.add_argument('--mr-aug-gibbs-truncation-min', type=float, default=0.6,
                       help='Training minimum k-space truncation factor for Gibbs ringing')
    parser.add_argument('--mr-aug-gibbs-truncation-max', type=float, default=0.9,
                       help='Training maximum k-space truncation factor for Gibbs ringing')
    parser.add_argument('--mr-aug-gibbs-type', type=str, choices=['circular', 'rectangular', 'elliptical'],
                       default='circular', help='Training Gibbs ringing truncation pattern type')
    parser.add_argument('--mr-aug-gibbs-anisotropy', type=float, default=0.3,
                       help='Training Gibbs ringing directional anisotropy (0.0=isotropic, 1.0=strong PE bias)')
    
    # Training k-space noise injection parameters
    parser.add_argument('--mr-aug-noise-type', type=str, choices=['gaussian', 'rician'], 
                       default='rician', help='Training k-space noise distribution type')
    parser.add_argument('--mr-aug-noise-std-min', type=float, default=0.005,
                       help='Training minimum noise standard deviation')
    parser.add_argument('--mr-aug-noise-std-max', type=float, default=0.02,
                       help='Training maximum noise standard deviation')
    
    # Training coil sensitivity bias field parameters
    parser.add_argument('--mr-aug-biasfield-order', type=int, default=3,
                       help='Training bias field polynomial order (higher = smoother)')
    parser.add_argument('--mr-aug-biasfield-min', type=float, default=0.9,
                       help='Training minimum bias field multiplier')
    parser.add_argument('--mr-aug-biasfield-max', type=float, default=1.1,
                       help='Training maximum bias field multiplier')
    
    # Training bias field stripe parameters
    parser.add_argument('--mr-aug-bias-stripe-prob', type=float, default=0.5,
                       help='Training probability of adding sinusoidal stripes to bias field')
    parser.add_argument('--mr-aug-bias-stripe-freq-min', type=float, default=5.0,
                       help='Training minimum stripe frequency in pixels per cycle')
    parser.add_argument('--mr-aug-bias-stripe-freq-max', type=float, default=40.0,
                       help='Training maximum stripe frequency in pixels per cycle')
    parser.add_argument('--mr-aug-bias-stripe-amp-min', type=float, default=0.005,
                       help='Training minimum stripe amplitude')
    parser.add_argument('--mr-aug-bias-stripe-amp-max', type=float, default=0.04,
                       help='Training maximum stripe amplitude')
    
    # Validation transformation limits
    parser.add_argument('--val-mr-aug-max-rotation', type=float, default=180.0,
                       help='Validation maximum rotation angle in degrees')
    parser.add_argument('--val-mr-aug-max-scaling', type=float, default=0.25,
                       help='Validation maximum scaling factor (0.25 = ±25 percent)')
    parser.add_argument('--val-mr-aug-max-translation-x', type=float, default=0.125,
                       help='Validation maximum translation along x-axis as fraction of width')
    parser.add_argument('--val-mr-aug-max-translation-y', type=float, default=0.08,
                       help='Validation maximum translation along y-axis as fraction of height')
    parser.add_argument('--val-mr-aug-max-shearing-x', type=float, default=15.0,
                       help='Validation maximum shearing along x-axis in degrees')
    parser.add_argument('--val-mr-aug-max-shearing-y', type=float, default=15.0,
                       help='Validation maximum shearing along y-axis in degrees')
    
    # Validation color augmentation limits
    parser.add_argument('--val-mr-aug-max-brightness', type=float, default=0.2,
                       help='Validation maximum brightness adjustment (±20% default)')
    parser.add_argument('--val-mr-aug-max-contrast', type=float, default=0.2,
                       help='Validation maximum contrast adjustment (±20% default)')
    
    # Validation Gibbs ringing simulation parameters  
    parser.add_argument('--val-mr-aug-gibbs-truncation-min', type=float, default=0.6,
                       help='Validation minimum k-space truncation factor for Gibbs ringing')
    parser.add_argument('--val-mr-aug-gibbs-truncation-max', type=float, default=0.9,
                       help='Validation maximum k-space truncation factor for Gibbs ringing')
    parser.add_argument('--val-mr-aug-gibbs-type', type=str, choices=['circular', 'rectangular', 'elliptical'],
                       default='circular', help='Validation Gibbs ringing truncation pattern type')
    parser.add_argument('--val-mr-aug-gibbs-anisotropy', type=float, default=0.3,
                       help='Validation Gibbs ringing directional anisotropy (0.0=isotropic, 1.0=strong PE bias)')
    
    # Validation k-space noise injection parameters
    parser.add_argument('--val-mr-aug-noise-type', type=str, choices=['gaussian', 'rician'],
                       default='rician', help='Validation k-space noise distribution type')
    parser.add_argument('--val-mr-aug-noise-std-min', type=float, default=0.005,
                       help='Validation minimum noise standard deviation')
    parser.add_argument('--val-mr-aug-noise-std-max', type=float, default=0.02,
                       help='Validation maximum noise standard deviation')
    
    # Validation coil sensitivity bias field parameters
    parser.add_argument('--val-mr-aug-biasfield-order', type=int, default=3,
                       help='Validation bias field polynomial order (higher = smoother)')
    parser.add_argument('--val-mr-aug-biasfield-min', type=float, default=0.9,
                       help='Validation minimum bias field multiplier')
    parser.add_argument('--val-mr-aug-biasfield-max', type=float, default=1.1,
                       help='Validation maximum bias field multiplier')
    
    # Validation bias field stripe parameters
    parser.add_argument('--val-mr-aug-bias-stripe-prob', type=float, default=0.5,
                       help='Validation probability of adding sinusoidal stripes to bias field')
    parser.add_argument('--val-mr-aug-bias-stripe-freq-min', type=float, default=5.0,
                       help='Validation minimum stripe frequency in pixels per cycle')
    parser.add_argument('--val-mr-aug-bias-stripe-freq-max', type=float, default=40.0,
                       help='Validation maximum stripe frequency in pixels per cycle')
    parser.add_argument('--val-mr-aug-bias-stripe-amp-min', type=float, default=0.005,
                       help='Validation minimum stripe amplitude')
    parser.add_argument('--val-mr-aug-bias-stripe-amp-max', type=float, default=0.04,
                       help='Validation maximum stripe amplitude')
    
    # Training interpolation and upsampling
    parser.add_argument('--mr-aug-interpolation-order', type=int, default=1, choices=[1, 3],
                       help='Training interpolation order: 1=bilinear, 3=bicubic')
    parser.add_argument('--mr-aug-upsample', action='store_true', default=False,
                       help='Training enable upsampling before interpolation to reduce aliasing')
    parser.add_argument('--mr-aug-upsample-factor', type=int, default=2,
                       help='Training upsampling factor if upsampling is enabled')
    parser.add_argument('--mr-aug-upsample-order', type=int, default=1, choices=[1, 3],
                       help='Training upsampling interpolation order')
    
    # Validation interpolation and upsampling
    parser.add_argument('--val-mr-aug-interpolation-order', type=int, default=1, choices=[1, 3],
                       help='Validation interpolation order: 1=bilinear, 3=bicubic')
    parser.add_argument('--val-mr-aug-upsample', action='store_true', default=False,
                       help='Validation enable upsampling before interpolation to reduce aliasing')
    parser.add_argument('--val-mr-aug-upsample-factor', type=int, default=2,
                       help='Validation upsampling factor if upsampling is enabled')
    parser.add_argument('--val-mr-aug-upsample-order', type=int, default=1, choices=[1, 3],
                       help='Validation upsampling interpolation order')
    
    # Training resolution limit
    parser.add_argument('--mr-aug-max-train-resolution', type=int, nargs=2, default=None,
                       help='Maximum training resolution [height width]. Images larger than this will be center cropped.')
    parser.add_argument('--val-mr-aug-max-train-resolution', type=int, nargs=2, default=None,
                       help='Maximum validation resolution [height width]. Images larger than this will be center cropped.')

    args = parser.parse_args()
    return args


def create_exp_dir(args):
    """Create experiment directory structure"""
    args.exp_dir = Path('../result') / args.net_name / args.train_phase
    args.val_dir = args.exp_dir / 'val'
    args.main_dir = args.exp_dir / '__info'
    args.val_loss_dir = args.exp_dir

    args.val_dir.mkdir(parents=True, exist_ok=True)
    args.main_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Loss logs directory: {args.val_loss_dir}")


def check_requirements(args):
    """Check if required files exist"""
    # Check if training data exists
    if not os.path.exists(args.data_path_train):
        raise FileNotFoundError(f"Training data not found at {args.data_path_train}")
    
    if not args.no_validation and not os.path.exists(args.data_path_val):
        raise FileNotFoundError(f"Validation data not found at {args.data_path_val}")
    
    print("✓ All required files found")


def print_config(args):
    """Print training configuration"""
    print("=" * 80)
    print("VarNet SAG Training Configuration")
    print("=" * 80)
    print(f"Network: {args.net_name}")
    print(f"GPU: {args.GPU_NUM}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient Accumulation Step: {args.accumulation_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"No validation: {args.no_validation}")
    print()
    print("Data:")
    print(f"  Train: {args.data_path_train}")
    if not args.no_validation:
        print(f"  Val: {args.data_path_val}")
    print(f"  Mask mode: {args.mask_mode}")
    print(f"  MR augment enabled: {args.mr_aug_on}")
    if hasattr(args, 'mr_aug_strength'):
        print(f"  MR augment strength: {args.mr_aug_strength}")
    print("=" * 80)


if __name__ == '__main__':
    print("Starting VarNet SAG Training...")
    
    args = parse()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, training will be very slow!")
    else:
        print(f"CUDA available: {torch.cuda.device_count()} GPUs")
        print(f"Using GPU {args.GPU_NUM}")
    
    # Set random seed
    seed_fix(args.seed)
    
    # Create experiment directories
    create_exp_dir(args)
    
    # Check requirements
    check_requirements(args)
    
    # Print configuration
    print_config(args)
    
    # Copy this script to experiment directory for reproducibility
    script_path = Path(__file__)
    shutil.copy2(script_path, args.main_dir / script_path.name)
    
    # Save arguments
    import json
    with open(args.main_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    # Start training
    try:
        train(args)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        torch.cuda.empty_cache()