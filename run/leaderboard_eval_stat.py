import argparse
import numpy as np
import h5py
import random
import glob
import os
import sys
import torch
import torch.nn.functional as F
import cv2 
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import utils
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(parent_dir)

from utils.common.loss_function import SSIMLoss


class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)
            
    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))
            
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return S.mean()


def compute_statistics(values):
    """Compute min, max, mean, std statistics for a list of values."""
    if not values:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'count': 0}
    
    values = np.array(values)
    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'count': len(values)
    }


def forward_with_stats(args):
    """
    Evaluate leaderboard data with detailed statistics.
    
    Returns detailed SSIM statistics for:
    - Overall (all slices)
    - acc4 brain, acc4 knee, acc8 brain, acc8 knee
    - Per-file SSIM values
    """
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    leaderboard_data = glob.glob(os.path.join(args.leaderboard_data_path,'*.h5'))
    if len(leaderboard_data) != 58:
        raise NotImplementedError('Leaderboard Data Size Should Be 58')
    
    your_data = glob.glob(os.path.join(args.your_data_path,'*.h5'))
    if len(your_data) != 58:
        raise NotImplementedError('Your Data Size Should Be 58')           
    
    # Store SSIM values for each category
    ssim_values = {
        'overall': [],
        'acc4_brain': [],
        'acc4_knee': [],
        'acc8_brain': [],
        'acc8_knee': []
    }
    
    # Store per-file SSIM values and metadata
    file_ssim_values = {}
    file_metadata = {}  # Store coil count and slice count for each file
    
    ssim_calculator = SSIM().to(device=device)
    
    with torch.no_grad():
        for part in ['brain_test', 'knee_test']:
            for i_subject in range(29):
                l_fname = os.path.join(args.leaderboard_data_path, part + str(i_subject+1) + '.h5')
                y_fname = os.path.join(args.your_data_path, part + str(i_subject+1) + '.h5')
                
                # Get filename for tracking
                filename = part + str(i_subject+1) + '.h5'
                file_ssim_values[filename] = []
                
                with h5py.File(l_fname, "r") as hf:
                    num_slices = hf['image_label'].shape[0]
                    # Get coil count from kspace data path
                    kspace_fname = l_fname.replace('/image/', '/kspace/')
                    with h5py.File(kspace_fname, "r") as kf:
                        coil_count = kf['kspace'].shape[1]  # shape is (slice, coil, height, width)
                    
                    file_metadata[filename] = {
                        'num_slices': num_slices,
                        'num_coils': coil_count
                    }
                
                for i_slice in range(num_slices):
                    with h5py.File(l_fname, "r") as hf:
                        target = hf['image_label'][i_slice]
                        mask = np.zeros(target.shape)
                        if part == 'knee_test':
                            mask[target>2e-5] = 1
                        elif part == 'brain_test':
                            mask[target>5e-5] = 1
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                        mask = cv2.dilate(mask, kernel, iterations=15)
                        mask = cv2.erode(mask, kernel, iterations=14)
                        
                        target = torch.from_numpy(target).to(device=device)
                        mask = (torch.from_numpy(mask).to(device=device)).type(torch.float)
                        maximum = hf.attrs['max']
                        
                    with h5py.File(y_fname, "r") as hf:
                        recon = hf[args.output_key][i_slice]
                        recon = torch.from_numpy(recon).to(device=device)
                        
                    # Calculate SSIM for this slice
                    slice_ssim = ssim_calculator(recon*mask, target*mask, maximum).cpu().numpy()
                    
                    # Add to overall
                    ssim_values['overall'].append(slice_ssim)
                    
                    # Add to per-file tracking
                    file_ssim_values[filename].append(float(slice_ssim))
                    
                    # Add to specific category based on acceleration and anatomy
                    # Determine acceleration from filename or path
                    if 'acc4' in str(args.your_data_path) or args.acceleration == 4:
                        if part == 'brain_test':
                            ssim_values['acc4_brain'].append(slice_ssim)
                        else:  # knee_test
                            ssim_values['acc4_knee'].append(slice_ssim)
                    elif 'acc8' in str(args.your_data_path) or args.acceleration == 8:
                        if part == 'brain_test':
                            ssim_values['acc8_brain'].append(slice_ssim)
                        else:  # knee_test
                            ssim_values['acc8_knee'].append(slice_ssim)
    
    return ssim_values, file_ssim_values, file_metadata


def print_statistics_table(stats_dict):
    """Print a formatted table of statistics."""
    print("\n" + "="*80)
    print("LEADERBOARD EVALUATION STATISTICS")
    print("="*80)
    
    # Header
    print(f"{'Category':<15} {'Count':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}")
    print("-"*80)
    
    # Print each category
    for category, stats in stats_dict.items():
        print(f"{category.replace('_', ' ').title():<15} "
              f"{stats['count']:<8} "
              f"{stats['min']:<8.4f} "
              f"{stats['max']:<8.4f} "
              f"{stats['mean']:<8.4f} "
              f"{stats['std']:<8.4f}")
    
    print("-"*80)
    print(f"Overall SSIM: {stats_dict['overall']['mean']:.4f} ± {stats_dict['overall']['std']:.4f}")
    print("="*80)


def save_statistics_to_file(stats_dict, output_file):
    """Save statistics to a file."""
    with open(output_file, 'w') as f:
        f.write("LEADERBOARD EVALUATION STATISTICS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"{'Category':<15} {'Count':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}\n")
        f.write("-"*60 + "\n")
        
        for category, stats in stats_dict.items():
            f.write(f"{category.replace('_', ' ').title():<15} "
                   f"{stats['count']:<8} "
                   f"{stats['min']:<8.4f} "
                   f"{stats['max']:<8.4f} "
                   f"{stats['mean']:<8.4f} "
                   f"{stats['std']:<8.4f}\n")
        
        f.write("-"*60 + "\n")
        f.write(f"Overall SSIM: {stats_dict['overall']['mean']:.4f} ± {stats_dict['overall']['std']:.4f}\n")


def save_per_file_ssim(file_ssim_acc4, file_ssim_acc8, file_metadata_acc4, file_metadata_acc8, output_dir):
    """Save per-file SSIM values to separate files."""
    import json
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save acc4 per-file SSIM values
    acc4_output = os.path.join(output_dir, 'ssim_per_file_acc4.json')
    with open(acc4_output, 'w') as f:
        json.dump(file_ssim_acc4, f, indent=2)
    print(f"Saved acc4 per-file SSIM values to: {acc4_output}")
    
    # Save acc8 per-file SSIM values  
    acc8_output = os.path.join(output_dir, 'ssim_per_file_acc8.json')
    with open(acc8_output, 'w') as f:
        json.dump(file_ssim_acc8, f, indent=2)
    print(f"Saved acc8 per-file SSIM values to: {acc8_output}")
    
    # Create combined file with summary statistics per file
    combined_output = os.path.join(output_dir, 'ssim_per_file_summary.txt')
    with open(combined_output, 'w') as f:
        f.write("PER-FILE SSIM SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # ACC4 files
        f.write("ACC4 FILES:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Filename':<20} {'Slices':<8} {'Coils':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
        f.write("-"*68 + "\n")
        
        for filename, ssim_values in sorted(file_ssim_acc4.items()):
            if ssim_values:  # Check if list is not empty
                ssim_array = np.array(ssim_values)
                coil_count = file_metadata_acc4.get(filename, {}).get('num_coils', 'N/A')
                f.write(f"{filename:<20} {len(ssim_values):<8} {coil_count:<8} "
                       f"{np.mean(ssim_array):<8.4f} {np.std(ssim_array):<8.4f} "
                       f"{np.min(ssim_array):<8.4f} {np.max(ssim_array):<8.4f}\n")
        
        f.write("\n")
        
        # ACC8 files
        f.write("ACC8 FILES:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Filename':<20} {'Slices':<8} {'Coils':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}\n")
        f.write("-"*68 + "\n")
        
        for filename, ssim_values in sorted(file_ssim_acc8.items()):
            if ssim_values:  # Check if list is not empty
                ssim_array = np.array(ssim_values)
                coil_count = file_metadata_acc8.get(filename, {}).get('num_coils', 'N/A')
                f.write(f"{filename:<20} {len(ssim_values):<8} {coil_count:<8} "
                       f"{np.mean(ssim_array):<8.4f} {np.std(ssim_array):<8.4f} "
                       f"{np.min(ssim_array):<8.4f} {np.max(ssim_array):<8.4f}\n")
    
    print(f"Saved per-file summary to: {combined_output}")
    
    return acc4_output, acc8_output, combined_output


if __name__ == '__main__':
    """
    Enhanced leaderboard evaluation with detailed statistics.
    Provides min, max, mean, std for overall and category-specific SSIM values.
    """
    parser = argparse.ArgumentParser(description=
                                     'FastMRI challenge Leaderboard Image Evaluation with Statistics',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0)
    parser.add_argument('-lp', '--path_leaderboard_data', type=Path, default='/Data/leaderboard/')
    parser.add_argument('-yp', '--path_your_data', type=Path, default='result/test_Unet/reconstructions_leaderboard/')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')
    parser.add_argument('-o', '--output_file', type=str, default=None, 
                       help='Optional file to save statistics (default: no file output)')
    parser.add_argument('--detailed', action='store_true', default=False,
                       help='Print individual slice SSIM values')
    parser.add_argument('--save-per-file', action='store_true', default=False,
                       help='Save per-file SSIM values to JSON and summary files')
    parser.add_argument('--per-file-dir', type=str, default=None,
                       help='Directory to save per-file SSIM results (default: same as output file or current dir)')
    
    args = parser.parse_args()

    assert(len(os.listdir(args.path_leaderboard_data)) == 2)
    
    # Process both acc4 and acc8
    all_stats = {}
    
    # acc4
    print("Processing acc4 data...")
    args.leaderboard_data_path = args.path_leaderboard_data / "acc4" / 'image'
    args.your_data_path = args.path_your_data / "acc4"
    args.acceleration = 4
    
    ssim_values_acc4, file_ssim_acc4, file_metadata_acc4 = forward_with_stats(args)
    
    # acc8
    print("Processing acc8 data...")
    args.leaderboard_data_path = args.path_leaderboard_data / "acc8" / 'image'
    args.your_data_path = args.path_your_data / "acc8"
    args.acceleration = 8
    
    ssim_values_acc8, file_ssim_acc8, file_metadata_acc8 = forward_with_stats(args)
    
    # Combine results
    combined_ssim_values = {
        'overall': ssim_values_acc4['overall'] + ssim_values_acc8['overall'],
        'acc4_brain': ssim_values_acc4['acc4_brain'],
        'acc4_knee': ssim_values_acc4['acc4_knee'],
        'acc8_brain': ssim_values_acc8['acc8_brain'],
        'acc8_knee': ssim_values_acc8['acc8_knee']
    }
    
    # Compute statistics for each category
    stats_dict = {}
    for category, values in combined_ssim_values.items():
        stats_dict[category] = compute_statistics(values)
    
    # Print results
    print_statistics_table(stats_dict)
    
    # Print detailed results if requested
    if args.detailed:
        print("\nDETAILED SLICE-BY-SLICE RESULTS:")
        print("-"*50)
        for category, values in combined_ssim_values.items():
            if values:  # Only print if category has values
                print(f"\n{category.replace('_', ' ').title()} ({len(values)} slices):")
                for i, val in enumerate(values):
                    print(f"  Slice {i+1:3d}: {val:.4f}")
    
    # Save to file if requested
    if args.output_file:
        save_statistics_to_file(stats_dict, args.output_file)
        print(f"\nStatistics saved to: {args.output_file}")
    
    # Save per-file SSIM values if requested
    if args.save_per_file:
        # Determine output directory
        if args.per_file_dir:
            output_dir = args.per_file_dir
        elif args.output_file:
            output_dir = os.path.dirname(args.output_file)
        else:
            output_dir = "."
        
        save_per_file_ssim(file_ssim_acc4, file_ssim_acc8, file_metadata_acc4, file_metadata_acc8, output_dir)
    
    # Print original format for compatibility
    print("\n" + "="*30)
    print("ORIGINAL FORMAT RESULTS:")
    print("="*30)
    overall_mean = stats_dict['overall']['mean']
    acc4_mean = (stats_dict['acc4_brain']['mean'] * stats_dict['acc4_brain']['count'] + 
                 stats_dict['acc4_knee']['mean'] * stats_dict['acc4_knee']['count']) / \
                (stats_dict['acc4_brain']['count'] + stats_dict['acc4_knee']['count'])
    acc8_mean = (stats_dict['acc8_brain']['mean'] * stats_dict['acc8_brain']['count'] + 
                 stats_dict['acc8_knee']['mean'] * stats_dict['acc8_knee']['count']) / \
                (stats_dict['acc8_brain']['count'] + stats_dict['acc8_knee']['count'])
    
    print("Leaderboard SSIM : {:.4f}".format(overall_mean))
    print("="*10 + " Details " + "="*10)
    print("Leaderboard SSIM (acc4): {:.4f}".format(acc4_mean))
    print("Leaderboard SSIM (acc8): {:.4f}".format(acc8_mean))