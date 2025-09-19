"""
Mask augmentation module for FastMRI Challenge.

This module provides functionality to augment training masks by replacing
the fixed Equispaced masks with various mask types and accelerations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.fastmri.data.subsample import create_mask_for_mask_type


class MaskAugmentor:
    """
    Handles mask augmentation for training data.
    
    Supports 5 mask types with configurable probabilities:
    - random
    - equispaced  
    - equispaced_fraction
    - magic
    - magic_fraction
    """
    
    MASK_TYPES = ['random', 'equispaced', 'equispaced_fraction', 'magic', 'magic_fraction']
    
    def __init__(self, mask_types_prob: List[float], mask_acc_probs: str):
        """
        Initialize the mask augmentor.
        
        Args:
            mask_types_prob: List of 5 probabilities for mask types
            mask_acc_probs: String with acceleration:probability pairs 
                            (e.g., "4:0.4 6:0.1 8:0.5")
        """
        self.base_mask_types_prob = np.array(mask_types_prob)
        self.base_mask_types_prob = self.base_mask_types_prob / self.base_mask_types_prob.sum()  # Normalize
        
        # Current scheduling probability (will be updated each epoch)
        self.schedule_prob = 1.0
        
        # Parse acceleration probabilities
        self.accelerations, self.acc_probs = self._parse_acc_probs(mask_acc_probs)
        
        self.rng = np.random.RandomState()
    
    def _parse_acc_probs(self, mask_acc_probs: str) -> Tuple[List[int], List[float]]:
        """
        Parse acceleration probabilities string.
        
        Args:
            mask_acc_probs: String like "4:0.4 6:0.1 8:0.5"
            
        Returns:
            Tuple of (accelerations, probabilities)
        """
        pairs = mask_acc_probs.strip().split()
        accelerations = []
        probs = []
        
        for pair in pairs:
            acc_str, prob_str = pair.split(':')
            accelerations.append(int(acc_str))
            probs.append(float(prob_str))
        
        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return accelerations, probs.tolist()
    
    def _get_center_fraction(self, acceleration: int) -> float:
        """
        Get default center fraction for given acceleration.
        
        Args:
            acceleration: Acceleration factor
            
        Returns:
            Center fraction
        """
        # Use reasonable defaults based on FastMRI paper
        if acceleration <= 4:
            return 0.08
        elif acceleration <= 8:
            return 0.04
        else:
            return 0.02
    
    def sample_mask_type(self) -> str:
        """
        Sample a mask type based on configured probabilities scaled by schedule_prob.
        
        Returns:
            Selected mask type
        """
        # Scale mask type probabilities by current scheduling probability
        # When schedule_prob = 0, use only equispaced (index 1)
        # When schedule_prob = 1, use full base probabilities
        current_prob = self.base_mask_types_prob.copy()
        
        # Scale non-equispaced probabilities by schedule_prob
        scaled_prob = current_prob * self.schedule_prob
        
        # Add remaining probability to equispaced (index 1) to maintain normalization
        remaining_prob = 1.0 - scaled_prob.sum()
        scaled_prob[1] += remaining_prob  # equispaced gets the remaining probability
        
        # Ensure probabilities sum to 1
        scaled_prob = scaled_prob / scaled_prob.sum()
        
        return self.rng.choice(self.MASK_TYPES, p=scaled_prob)
    
    def sample_acceleration(self) -> int:
        """
        Sample an acceleration based on configured probabilities.
        
        Returns:
            Selected acceleration
        """
        return self.rng.choice(self.accelerations, p=self.acc_probs)
    
    def generate_mask(self, kspace_shape: Tuple[int, ...], seed: Optional[int] = None) -> Tuple[np.ndarray, str, int]:
        """
        Generate a new mask for the given k-space shape.
        
        Args:
            kspace_shape: Shape of the k-space data (slice, coil, height, width)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (mask, mask_type, acceleration) where:
            - mask: Generated mask as numpy array with shape (width,)
            - mask_type: The mask type used
            - acceleration: The acceleration used
        """
        # Set seed for reproducible sampling
        if seed is not None:
            self.set_seed(seed)
        
        # Sample mask type and acceleration
        mask_type = self.sample_mask_type()
        acceleration = self.sample_acceleration()
        
        # Get center fraction for this specific acceleration
        center_fraction = self._get_center_fraction(acceleration)
        
        # Create mask function for this specific mask type and acceleration
        mask_func = create_mask_for_mask_type(
            mask_type, [center_fraction], [acceleration]
        )
        
        # For our data format (slice, coil, height, width), we need to create a shape
        # that the FastMRI library expects. The library uses shape[-2] for masking,
        # but our mask should be along the width dimension (shape[-1]).
        # So we need to swap the last two dimensions for the library call.
        
        # Original shape: (slice, coil, height, width) = (slice, coil, 640, 372)
        # FastMRI expects: (slice, coil, width, height) = (slice, coil, 372, 640)
        fastmri_shape = kspace_shape[:-2] + (kspace_shape[-1], kspace_shape[-2])
        
        # Generate mask using official FastMRI API
        mask_result = mask_func(fastmri_shape, seed=seed)
        
        # Handle both old and new API
        if isinstance(mask_result, tuple):
            mask, num_low_frequencies = mask_result
        else:
            mask = mask_result
        
        # Convert to numpy and reshape to 1D
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()
        mask = mask.reshape(-1)
        
        return mask.astype(np.float32), mask_type, acceleration
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng.seed(seed)
    
    def update_schedule_prob(self, schedule_prob: float):
        """
        Update the scheduling probability for mask augmentation.
        
        Args:
            schedule_prob: Current scheduling probability (0.0-1.0)
        """
        self.schedule_prob = max(0.0, min(1.0, schedule_prob))


def calculate_mask_aug_schedule_prob(epoch: int, args) -> float:
    """
    Calculate the current mask augmentation scheduling probability based on epoch and schedule type.
    
    Args:
        epoch: Current epoch (0-indexed)
        args: Parsed command line arguments
        
    Returns:
        Current scheduling probability (0.0-1.0)
    """
    if epoch < args.mask_aug_delay:
        return 0.0  # No augmentation during delay period
    
    # Calculate progress from delay epoch to end
    total_epochs = args.num_epochs
    progress_epochs = total_epochs - args.mask_aug_delay
    current_progress_epoch = epoch - args.mask_aug_delay
    
    if progress_epochs <= 0:
        return args.mask_aug_end_prob
    
    # Calculate progress ratio (0.0 to 1.0)
    progress = min(1.0, current_progress_epoch / progress_epochs)
    
    if args.mask_aug_schedule == 'constant':
        return args.mask_aug_end_prob
    elif args.mask_aug_schedule == 'linear':
        return args.mask_aug_start_prob + (args.mask_aug_end_prob - args.mask_aug_start_prob) * progress
    elif args.mask_aug_schedule == 'exp':
        # Exponential scheduling: starts slow, accelerates
        exp_progress = 1.0 - np.exp(-5.0 * progress)  # -5 gives good curve shape
        return args.mask_aug_start_prob + (args.mask_aug_end_prob - args.mask_aug_start_prob) * exp_progress
    else:
        return args.mask_aug_end_prob


def create_mask_augmentor(args) -> Optional[MaskAugmentor]:
    """
    Create a mask augmentor from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        MaskAugmentor instance if augmentation is enabled, None otherwise
    """
    if args.mask_mode == 'original':
        return None
    
    return MaskAugmentor(
        mask_types_prob=args.mask_types_prob,
        mask_acc_probs=args.mask_acc_probs
    )