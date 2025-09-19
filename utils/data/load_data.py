import h5py
import random
from utils.data.transforms import DataTransform
from utils.data.mask_augment import create_mask_augmentor
from utils.data.mr_augment import create_mr_augmentor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


def get_anatomy_label(filename):
    """
    Extract anatomy label from filename.
    
    Args:
        filename: Path or string containing filename
        
    Returns:
        int: 0 for brain, 1 for knee
    """
    filename_str = str(filename).lower()
    if 'brain' in filename_str:
        return 0
    elif 'knee' in filename_str:
        return 1
    else:
        # Default to brain if neither is found
        return 0

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, mode=0, grappa=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.mode = mode
        self.image_examples = []
        self.kspace_examples = []
        self.grappa = grappa

        print(f"Loading Data... Mode: {self.mode} (baseline: 0, brain-only: 1, knee-only: 2)")

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                if self._should_include_file(fname):
                    num_slices = self._get_metadata(fname)

                    self.image_examples += [
                        (fname, slice_ind) for slice_ind in range(num_slices)
                    ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            if self._should_include_file(fname):
                num_slices = self._get_metadata(fname)

                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    def _should_include_file(self, fname):
        """
        Determine if a file should be included based on the mode.
        
        Args:
            fname: Path object for the file
            
        Returns:
            bool: True if file should be included, False otherwise
        """
        if self.mode == 0:
            # Baseline mode - include all files
            return True
        elif self.mode == 1:
            # Only brain data
            return "brain" in str(fname).lower()
        elif self.mode == 2:
            # Only knee data
            return "knee" in str(fname).lower()
        else:
            # Default to baseline for unknown modes
            return True

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        kspace_fname, dataslice = self.kspace_examples[i]
        
        if not self.forward:
            image_fname, _ = self.image_examples[i]
            if image_fname.name != kspace_fname.name:
                raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        
        grappa = -1
        if self.forward:
            target = -1
            attrs = -1
            if (self.grappa):
                # Need to find corresponding image file for forward pass
                # Replace 'kspace' with 'image' in the path
                image_fname = Path(str(kspace_fname).replace('/kspace/', '/image/'))
                try:
                    with h5py.File(image_fname, "r") as hf:
                        if 'image_grappa' in hf.keys():
                            grappa = hf['image_grappa'][dataslice]
                        else:
                            grappa = -1
                except (KeyError, FileNotFoundError) as e:
                    print(f"Error loading data from {image_fname}: {e}")
                    if isinstance(e, FileNotFoundError):
                        print(f"Image file not found: {image_fname}")
                    else:
                        print(f"Available keys: {list(hf.keys())}")
                        print(f"Available attrs: {list(hf.attrs.keys())}")
                    grappa = -1
        else: 
            try:
                with h5py.File(image_fname, "r") as hf:
                    #print(f"Image file {image_fname} keys: {list(hf.keys())}")
                    #print(f"Image file {image_fname} attrs: {list(hf.attrs.keys())}")
                    target = hf[self.target_key][dataslice]
                    attrs = dict(hf.attrs)
                    if (self.grappa):
                        grappa = hf['image_grappa'][dataslice]
            except KeyError as e:
                print(f"Error loading data from {image_fname}: {e}")
                print(f"Available keys: {list(hf.keys())}")
                print(f"Available attrs: {list(hf.attrs.keys())}")
                raise
            
        # Generate anatomy label from filename
        anatomy_label = get_anatomy_label(kspace_fname.name)
        
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice, anatomy_label, grappa)

def create_combined_slice_data(train_path, val_path, transform, input_key, target_key, forward=False, mode=0, grappa=False):
    """
    Create a combined dataset from train and val directories.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        transform: Data transform to apply
        input_key: Key for input data
        target_key: Key for target data
        forward: Whether this is for forward pass
        mode: Data filtering mode (0=baseline, 1=brain only, 2=knee only)
        
    Returns:
        Combined SliceData dataset
    """
    # Create individual datasets
    train_data = SliceData(train_path, transform, input_key, target_key, forward, mode, grappa)
    val_data = SliceData(val_path, transform, input_key, target_key, forward, mode, grappa)
    
    # Combine the examples
    combined_data = SliceData.__new__(SliceData)  # Create without calling __init__
    combined_data.transform = transform
    combined_data.input_key = input_key
    combined_data.target_key = target_key
    combined_data.forward = forward
    combined_data.grappa = grappa
    
    # Combine examples from both datasets
    combined_data.image_examples = train_data.image_examples + val_data.image_examples
    combined_data.kspace_examples = train_data.kspace_examples + val_data.kspace_examples
    
    return combined_data

def create_data_loaders(data_path, args, shuffle=False, isforward=False, grappa=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    
    # Create mask augmentor if augmentation is enabled and we have the right arguments
    # For forward/inference, only create if explicitly requested
    mask_augmentor = None
    mr_augmentor = None
    
    if hasattr(args, 'mask_mode'):
        if not isforward or args.mask_mode == 'augment':
            mask_augmentor = create_mask_augmentor(args)
    
    if hasattr(args, 'mr_aug_on'):
        if not isforward or args.mr_aug_on:
            mr_augmentor = create_mr_augmentor(args)
    
    # Get mode from args if available, default to 0 (baseline)
    mode = getattr(args, 'data_anatomy_mode', 0)
    
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_, mask_augmentor, mr_augmentor),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        mode=mode,
        grappa=grappa
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader

def create_combined_data_loader(train_path, val_path, args, shuffle=False, isforward=False, grappa=False):
    """
    Create a data loader that combines train and val datasets.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data  
        args: Command line arguments
        shuffle: Whether to shuffle the data
        isforward: Whether this is for forward pass
        
    Returns:
        DataLoader with combined train+val data
    """
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    
    # Create mask augmentor if augmentation is enabled and we have the right arguments
    # For forward/inference, only create if explicitly requested
    mask_augmentor = None
    mr_augmentor = None
    
    if hasattr(args, 'mask_mode'):
        if not isforward or args.mask_mode == 'augment':
            mask_augmentor = create_mask_augmentor(args)
    
    if hasattr(args, 'mr_aug_on'):
        if not isforward or args.mr_aug_on:
            mr_augmentor = create_mr_augmentor(args)
    
    transform = DataTransform(isforward, max_key_, mask_augmentor, mr_augmentor)
    
    # Get mode from args if available, default to 0 (baseline)
    mode = getattr(args, 'data_anatomy_mode', 0)
    
    # Create combined dataset
    combined_data = create_combined_slice_data(
        train_path, val_path, transform, args.input_key, target_key_, isforward, mode, grappa
    )

    data_loader = DataLoader(
        dataset=combined_data,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    
    return data_loader

def create_val_args(args):
    """
    Create validation-specific arguments from training arguments.
    
    Args:
        args: Original training arguments
        
    Returns:
        Modified args object with validation-specific settings
    """
    import copy
    val_args = copy.deepcopy(args)
    
    # Replace training mask augmentation settings with validation settings if they exist
    if hasattr(args, 'val_mask_mode'):
        val_args.mask_mode = args.val_mask_mode
    if hasattr(args, 'val_mask_types_prob'):
        val_args.mask_types_prob = args.val_mask_types_prob
    if hasattr(args, 'val_mask_acc_probs'):
        val_args.mask_acc_probs = args.val_mask_acc_probs
    if hasattr(args, 'val_mask_aug_schedule'):
        val_args.mask_aug_schedule = args.val_mask_aug_schedule
    if hasattr(args, 'val_mask_aug_start_prob'):
        val_args.mask_aug_start_prob = args.val_mask_aug_start_prob
    if hasattr(args, 'val_mask_aug_end_prob'):
        val_args.mask_aug_end_prob = args.val_mask_aug_end_prob
    if hasattr(args, 'val_mask_aug_delay'):
        val_args.mask_aug_delay = args.val_mask_aug_delay
    
    # Replace training MR augmentation settings with validation settings if they exist
    if hasattr(args, 'val_mr_aug_on'):
        val_args.mr_aug_on = args.val_mr_aug_on
    if hasattr(args, 'val_mr_aug_debug'):
        val_args.mr_aug_debug = args.val_mr_aug_debug
    if hasattr(args, 'val_mr_aug_schedule'):
        val_args.mr_aug_schedule = args.val_mr_aug_schedule
    if hasattr(args, 'val_mr_aug_delay'):
        val_args.mr_aug_delay = args.val_mr_aug_delay
    if hasattr(args, 'val_mr_aug_strength'):
        val_args.mr_aug_strength = args.val_mr_aug_strength
    if hasattr(args, 'val_mr_aug_exp_decay'):
        val_args.mr_aug_exp_decay = args.val_mr_aug_exp_decay
        
    # Technique weights
    if hasattr(args, 'val_mr_aug_weight_fliph'):
        val_args.mr_aug_weight_fliph = args.val_mr_aug_weight_fliph
    if hasattr(args, 'val_mr_aug_weight_flipv'):
        val_args.mr_aug_weight_flipv = args.val_mr_aug_weight_flipv
    if hasattr(args, 'val_mr_aug_weight_rot90'):
        val_args.mr_aug_weight_rot90 = args.val_mr_aug_weight_rot90
    if hasattr(args, 'val_mr_aug_weight_rotation'):
        val_args.mr_aug_weight_rotation = args.val_mr_aug_weight_rotation
    if hasattr(args, 'val_mr_aug_weight_translation'):
        val_args.mr_aug_weight_translation = args.val_mr_aug_weight_translation
    if hasattr(args, 'val_mr_aug_weight_scaling'):
        val_args.mr_aug_weight_scaling = args.val_mr_aug_weight_scaling
    if hasattr(args, 'val_mr_aug_weight_shearing'):
        val_args.mr_aug_weight_shearing = args.val_mr_aug_weight_shearing
        
    # Transformation limits
    if hasattr(args, 'val_mr_aug_max_rotation'):
        val_args.mr_aug_max_rotation = args.val_mr_aug_max_rotation
    if hasattr(args, 'val_mr_aug_max_scaling'):
        val_args.mr_aug_max_scaling = args.val_mr_aug_max_scaling
    if hasattr(args, 'val_mr_aug_max_translation_x'):
        val_args.mr_aug_max_translation_x = args.val_mr_aug_max_translation_x
    if hasattr(args, 'val_mr_aug_max_translation_y'):
        val_args.mr_aug_max_translation_y = args.val_mr_aug_max_translation_y
    if hasattr(args, 'val_mr_aug_max_shearing_x'):
        val_args.mr_aug_max_shearing_x = args.val_mr_aug_max_shearing_x
    if hasattr(args, 'val_mr_aug_max_shearing_y'):
        val_args.mr_aug_max_shearing_y = args.val_mr_aug_max_shearing_y
        
    # Interpolation settings
    if hasattr(args, 'val_mr_aug_interpolation_order'):
        val_args.mr_aug_interpolation_order = args.val_mr_aug_interpolation_order
    if hasattr(args, 'val_mr_aug_upsample'):
        val_args.mr_aug_upsample = args.val_mr_aug_upsample
    if hasattr(args, 'val_mr_aug_upsample_factor'):
        val_args.mr_aug_upsample_factor = args.val_mr_aug_upsample_factor
    if hasattr(args, 'val_mr_aug_upsample_order'):
        val_args.mr_aug_upsample_order = args.val_mr_aug_upsample_order
    if hasattr(args, 'val_mr_aug_max_train_resolution'):
        val_args.mr_aug_max_train_resolution = args.val_mr_aug_max_train_resolution
    
    return val_args
