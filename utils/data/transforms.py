import numpy as np
import torch
from typing import Optional

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, mask_augmentor=None, mr_augmentor=None):
        self.isforward = isforward
        self.max_key = max_key
        self.mask_augmentor = mask_augmentor
        self.mr_augmentor = mr_augmentor
    def __call__(self, mask, input, target, attrs, fname, slice, anatomy_label, grappa):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        # Grappa
        if not isinstance(grappa, int) or grappa != -1:
            grappa = to_tensor(grappa)

        # Convert original k-space to tensor (fully-sampled)
        original_kspace = to_tensor(input)
        
        # Apply MRAugment if enabled (BEFORE masking - operates on fully-sampled k-space)
        if self.mr_augmentor is not None and not self.isforward:
            # Convert to real/imag format for MRAugment
            kspace_realimag = torch.stack((original_kspace.real, original_kspace.imag), dim=-1)
            
            debug = getattr(self.mr_augmentor.args, 'mr_aug_debug', False) if hasattr(self.mr_augmentor, 'args') else False
            if debug:
                print(f"Before MRAugment - kspace shape: {kspace_realimag.shape}, target shape: {target.shape}")
            
            # Apply MRAugment to fully-sampled k-space
            # Use original target shape for consistency (384, 384)
            aug_kspace_realimag, target = self.mr_augmentor(kspace_realimag, target.shape)
            
            if debug:
                print(f"After MRAugment - aug_kspace shape: {aug_kspace_realimag.shape}, target shape: {target.shape}")
            
            # Convert back to complex format for masking
            augmented_kspace = torch.complex(aug_kspace_realimag[..., 0], aug_kspace_realimag[..., 1])
        else:
            augmented_kspace = original_kspace
        
        # Apply mask augmentation if enabled
        if self.mask_augmentor is not None and not self.isforward:
            # Generate augmented mask based on augmented k-space shape
            augmented_mask, _, _ = self.mask_augmentor.generate_mask(augmented_kspace.shape)
            
            # Use augmented mask instead of original
            mask = augmented_mask
        
        # Apply mask to k-space data (AFTER MRAugment)
        kspace = augmented_kspace * mask
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        anatomy_label = torch.tensor(anatomy_label, dtype=torch.long)
        return mask, kspace, target, maximum, fname, slice, anatomy_label, grappa,
