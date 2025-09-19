"""
MRAugment implementation for FastMRI Challenge
Ported from official MRAugment repository: https://github.com/z-fabian/MRAugment

Applies physics-aware data augmentation to MRI k-space data with 7 augmentation techniques:
- Horizontal flip, Vertical flip, 90° rotation (pixel-preserving)
- Arbitrary rotation, Translation, Scaling, Shearing (interpolating)

Supports probability scheduling for gradual augmentation strength increase.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from math import exp

from utils.fastmri import fft2c, ifft2c


def complex_channel_first(x):
    """Convert (C, H, W, 2) to (2, C, H, W) for torchvision compatibility"""
    assert x.shape[-1] == 2
    if len(x.shape) == 3:
        # Single-coil (H, W, 2) -> (2, H, W)
        x = x.permute(2, 0, 1)
    else:
        # Multi-coil (C, H, W, 2) -> (2, C, H, W)
        assert len(x.shape) == 4
        x = x.permute(3, 0, 1, 2)
    return x


def complex_channel_last(x):
    """Convert (2, C, H, W) back to (C, H, W, 2)"""
    assert x.shape[0] == 2
    if len(x.shape) == 3:
        # Single-coil (2, H, W) -> (H, W, 2)
        x = x.permute(1, 2, 0)
    else:
        # Multi-coil (2, C, H, W) -> (C, H, W, 2)
        assert len(x.shape) == 4
        x = x.permute(1, 2, 3, 0)
    return x


def crop_if_needed(im, max_shape):
    """Center crop image if larger than max_shape"""
    assert len(max_shape) == 2
    if im.shape[-2] >= max_shape[0]:
        h_diff = im.shape[-2] - max_shape[0]
        h_crop_before = h_diff // 2
        h_interval = max_shape[0]
    else:
        h_crop_before = 0
        h_interval = im.shape[-2]

    if im.shape[-1] >= max_shape[1]:
        w_diff = im.shape[-1] - max_shape[1]
        w_crop_before = w_diff // 2
        w_interval = max_shape[1]
    else:
        w_crop_before = 0
        w_interval = im.shape[-1]

    return im[..., h_crop_before:h_crop_before+h_interval, w_crop_before:w_crop_before+w_interval]


def complex_crop_if_needed(im, max_shape):
    """Center crop complex image if larger than max_shape (for real/imag format)"""
    assert len(max_shape) == 2
    if im.shape[-3] >= max_shape[0]:
        h_diff = im.shape[-3] - max_shape[0]
        h_crop_before = h_diff // 2
        h_interval = max_shape[0]
    else:
        h_crop_before = 0
        h_interval = im.shape[-3]

    if im.shape[-2] >= max_shape[1]:
        w_diff = im.shape[-2] - max_shape[1]
        w_crop_before = w_diff // 2
        w_interval = max_shape[1]
    else:
        w_crop_before = 0
        w_interval = im.shape[-2]

    return im[..., h_crop_before:h_crop_before+h_interval, w_crop_before:w_crop_before+w_interval, :]


class MRAugmentationPipeline:
    """
    Core augmentation pipeline that applies geometric transformations to MRI data.
    Handles both pixel-preserving and interpolating transformations.
    """
    
    def __init__(self, args):
        self.args = args
        self.weight_dict = {
            'translation': args.mr_aug_weight_translation,
            'rotation': args.mr_aug_weight_rotation,
            'scaling': args.mr_aug_weight_scaling,
            'shearing': args.mr_aug_weight_shearing,
            'rot90': args.mr_aug_weight_rot90,
            'fliph': args.mr_aug_weight_fliph,
            'flipv': args.mr_aug_weight_flipv,
            'brightness': getattr(args, 'mr_aug_weight_brightness', 0.5),
            'contrast': getattr(args, 'mr_aug_weight_contrast', 0.5),
            'gibbs': getattr(args, 'mr_aug_weight_gibbs', 0.6),
            'noise': getattr(args, 'mr_aug_weight_noise', 1.0),
            'biasfield': getattr(args, 'mr_aug_weight_biasfield', 1.0),
            'streaks': getattr(args, 'mr_aug_weight_streaks', 0.0)
        }
        self.upsample_augment = args.mr_aug_upsample
        self.upsample_factor = args.mr_aug_upsample_factor
        self.upsample_order = args.mr_aug_upsample_order
        self.transform_order = args.mr_aug_interpolation_order
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im, max_output_size=None):
        """Apply augmentations to image tensor"""
        # Trailing dims must be image height and width (for torchvision)
        im = complex_channel_first(im)
        
        # ---------------------------  
        # pixel preserving transforms
        # ---------------------------  
        # Horizontal flip
        if self.random_apply('fliph'):
            im = TF.hflip(im)

        # Vertical flip 
        if self.random_apply('flipv'):
            im = TF.vflip(im)

        # Rotation by multiples of 90 deg 
        if self.random_apply('rot90'):
            k = self.rng.randint(1, 4)  
            im = torch.rot90(im, k, dims=[-2, -1])

        # Translation by integer number of pixels
        if self.random_apply('translation'):
            h, w = im.shape[-2:]
            t_x = self.rng.uniform(-self.args.mr_aug_max_translation_x, self.args.mr_aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.args.mr_aug_max_translation_y, self.args.mr_aug_max_translation_y)
            t_y = int(t_y * w)
            
            pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.crop(im, top, left, h, w)

        # ------------------------       
        # interpolating transforms
        # ------------------------  
        interp = False 

        # Rotation
        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.args.mr_aug_max_rotation, self.args.mr_aug_max_rotation)
        else:
            rot = 0.

        # Shearing
        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.args.mr_aug_max_shearing_x, self.args.mr_aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.args.mr_aug_max_shearing_y, self.args.mr_aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        # Scaling
        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1-self.args.mr_aug_max_scaling, 1 + self.args.mr_aug_max_scaling)
        else:
            scale = 1.

        # Upsample if needed
        upsample = interp and self.upsample_augment
        if upsample:
            upsampled_shape = [im.shape[-2] * self.upsample_factor, im.shape[-1] * self.upsample_factor]
            original_shape = im.shape[-2:]
            interpolation = TF.InterpolationMode.BICUBIC if self.upsample_order == 3 else TF.InterpolationMode.BILINEAR
            im = TF.resize(im, size=upsampled_shape, interpolation=interpolation)

        # Apply interpolating transformations 
        if interp:
            h, w = im.shape[-2:]
            pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.affine(im,
                           angle=rot,
                           scale=scale,
                           shear=(shear_x, shear_y),
                           translate=[0, 0],
                           interpolation=TF.InterpolationMode.BILINEAR
                          )
            im = TF.center_crop(im, (h, w))
        
        # Downsampling
        if upsample:
            im = TF.resize(im, size=original_shape, interpolation=interpolation)
        
        # Final cropping if augmented image is too large
        if max_output_size is not None:
            im = crop_if_needed(im, max_output_size)
        
        # Apply color augmentations to magnitude images
        im = self.apply_color_augmentations(im)
        
        # Apply bias field augmentation to simulate coil sensitivity variations
        # Need to convert back to channel-last format for bias field processing
        im_channel_last = complex_channel_last(im)
        if self.random_apply('biasfield'):
            biasfield_order = getattr(self.args, 'mr_aug_biasfield_order', 3)
            biasfield_min = getattr(self.args, 'mr_aug_biasfield_min', 0.9)
            biasfield_max = getattr(self.args, 'mr_aug_biasfield_max', 1.1)
            
            im_channel_last = self.apply_bias_field(
                im_channel_last,
                polynomial_order=biasfield_order,
                intensity_range=(biasfield_min, biasfield_max)
            )
            
        # Convert back to channel-first for final conversion
        im = complex_channel_first(im_channel_last)
            
        # Reset original channel ordering
        im = complex_channel_last(im)
        
        return im
    
    def augment_from_kspace(self, kspace, target_size, max_train_size=None):
        """
        Apply augmentations starting from k-space data
        
        Args:
            kspace: torch.Tensor of shape (coil, height, width, 2) with real/imaginary
            target_size: tuple (height, width) for output target
            max_train_size: optional max resolution for training
            
        Returns:
            augmented_kspace: torch.Tensor of shape (coil, height, width, 2)
            augmented_target: torch.Tensor of shape (height, width)
        """
        debug = getattr(self.args, 'mr_aug_debug', False)
        
        # Apply Gibbs ringing simulation in k-space domain
        if self.random_apply('gibbs'):
            truncation_min = getattr(self.args, 'mr_aug_gibbs_truncation_min', 0.6)
            truncation_max = getattr(self.args, 'mr_aug_gibbs_truncation_max', 0.9)
            gibbs_type = getattr(self.args, 'mr_aug_gibbs_type', 'circular')
            gibbs_anisotropy = getattr(self.args, 'mr_aug_gibbs_anisotropy', 0.3)
            
            kspace = self.apply_gibbs_ringing(
                kspace, 
                truncation_factor_range=(truncation_min, truncation_max),
                truncation_type=gibbs_type,
                anisotropy=gibbs_anisotropy
            )
            if debug:
                print(f"Applied Gibbs ringing with truncation range: [{truncation_min:.2f}, {truncation_max:.2f}]")
        
        # Apply k-space noise injection to simulate coil thermal noise
        if self.random_apply('noise'):
            noise_type = getattr(self.args, 'mr_aug_noise_type', 'rician')
            noise_std_min = getattr(self.args, 'mr_aug_noise_std_min', 0.005)
            noise_std_max = getattr(self.args, 'mr_aug_noise_std_max', 0.02)
            
            kspace = self.apply_kspace_noise(
                kspace,
                noise_type=noise_type,
                std_range=(noise_std_min, noise_std_max)
            )
            if debug:
                print(f"Applied k-space noise: type={noise_type}, std_range=[{noise_std_min:.4f}, {noise_std_max:.4f}]")
        
        # Apply k-space streak artifacts to simulate motion or flow artifacts
        if self.random_apply('streaks'):
            streak_density_min = getattr(self.args, 'mr_aug_streak_density_min', 0.01)
            streak_density_max = getattr(self.args, 'mr_aug_streak_density_max', 0.05)
            
            kspace = self.apply_kspace_streaks(
                kspace,
                density_range=(streak_density_min, streak_density_max)
            )
            if debug:
                print(f"Applied k-space streaks with density range: [{streak_density_min:.3f}, {streak_density_max:.3f}]")
        
        # Transform to image domain (FastMRI ifft2c expects real/imag format)
        im = ifft2c(kspace)
        
        # Apply augmentations to image
        im = self.augment_image(im, max_output_size=max_train_size)
        
        # Generate target from augmented image
        target = self.im_to_target(im, target_size)
        
        # Transform back to k-space (FastMRI fft2c expects real/imag format)
        augmented_kspace = fft2c(im)
        
        return augmented_kspace, target
    
    def im_to_target(self, im, target_size):
        """Generate target image from complex image data, crop/pad to exact target_size"""
        # For real/imag format: im.shape is (coil, height, width, 2)
        
        if len(im.shape) == 3: 
            # Single-coil (height, width, 2)
            from utils.fastmri import complex_abs
            magnitude_im = complex_abs(im)
        else:
            # Multi-coil (coil, height, width, 2)
            assert len(im.shape) == 4
            from utils.fastmri import rss_complex
            magnitude_im = rss_complex(im)
        
        debug = getattr(self.args, 'mr_aug_debug', False) if hasattr(self, 'args') else False
        
        # Get current and target dimensions
        current_h, current_w = magnitude_im.shape[-2], magnitude_im.shape[-1]
        target_h, target_w = target_size[0], target_size[1]
        
        # Step 1: Center crop if current > target
        if current_h > target_h or current_w > target_w:
            crop_h = min(current_h, target_h)
            crop_w = min(current_w, target_w)
            
            from utils.fastmri.data.transforms import center_crop
            magnitude_im = center_crop(magnitude_im, [crop_h, crop_w])
        
        # Step 2: Zero-pad if current < target  
        current_h, current_w = magnitude_im.shape[-2], magnitude_im.shape[-1]
        if current_h < target_h or current_w < target_w:
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            
            # Calculate symmetric padding
            pad_h_before = pad_h // 2
            pad_h_after = pad_h - pad_h_before
            pad_w_before = pad_w // 2
            pad_w_after = pad_w - pad_w_before
            
            # Apply padding (left, right, top, bottom)
            import torch.nn.functional as F
            magnitude_im = F.pad(magnitude_im, (pad_w_before, pad_w_after, pad_h_before, pad_h_after), 'constant', 0)
        
        return magnitude_im
            
    def random_apply(self, transform_name):
        """Check if transform should be applied based on probability"""
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else: 
            return False
        
    def set_augmentation_strength(self, p):
        """Set current augmentation strength [0, 1]"""
        self.augmentation_strength = p
    

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """Calculate necessary padding for affine transformation"""
        h, w = im.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.], 
            [h/2, w/2, 1.], 
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1) 
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return int(py.item()), int(px.item())

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        """Calculate padding and crop parameters for translation"""
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1) # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1) # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1) # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1) # pad right
            left = pad[2]
        return pad, top, left

    def apply_color_augmentations(self, im):
        """
        Apply brightness and contrast augmentations to complex image data.
        
        Args:
            im: Complex image tensor in channel-first format (2, C, H, W) or (2, H, W)
            
        Returns:
            Augmented complex image tensor
        """
        # Convert complex to magnitude for color augmentation
        if len(im.shape) == 3:
            # Single-coil (2, H, W) 
            magnitude = torch.sqrt(im[0]**2 + im[1]**2)
            phase = torch.atan2(im[1], im[0])
        else:
            # Multi-coil (2, C, H, W)
            magnitude = torch.sqrt(im[0]**2 + im[1]**2)
            phase = torch.atan2(im[1], im[0])
        
        # Apply brightness augmentation
        if self.random_apply('brightness'):
            brightness_factor = 1.0 + self.rng.uniform(
                -getattr(self.args, 'mr_aug_max_brightness', 0.2),
                getattr(self.args, 'mr_aug_max_brightness', 0.2)
            )
            magnitude = magnitude * brightness_factor
        
        # Apply contrast augmentation (gamma correction style)
        if self.random_apply('contrast'):
            gamma = 1.0 + self.rng.uniform(
                -getattr(self.args, 'mr_aug_max_contrast', 0.2),
                getattr(self.args, 'mr_aug_max_contrast', 0.2)
            )
            # Normalize to 0-1 range for gamma correction
            mag_min = magnitude.min()
            mag_max = magnitude.max()
            mag_range = mag_max - mag_min
            if mag_range > 0:
                normalized_mag = (magnitude - mag_min) / mag_range
                normalized_mag = torch.pow(normalized_mag, gamma)
                magnitude = normalized_mag * mag_range + mag_min
        
        # Convert back to complex representation
        if len(im.shape) == 3:
            # Single-coil (2, H, W)
            im[0] = magnitude * torch.cos(phase)
            im[1] = magnitude * torch.sin(phase)
        else:
            # Multi-coil (2, C, H, W)
            im[0] = magnitude * torch.cos(phase)
            im[1] = magnitude * torch.sin(phase)
        
        return im

    def create_truncation_mask(self, spatial_shape, truncation_factor, truncation_type='circular', anisotropy=0.0):
        """
        Create k-space truncation mask for Gibbs ringing simulation.
        
        Args:
            spatial_shape: Spatial dimensions tuple (H, W)
            truncation_factor: Fraction of k-space to keep (0.0-1.0)
            truncation_type: Type of truncation ('circular', 'rectangular', 'elliptical')
            anisotropy: Directional bias (0.0=isotropic, 1.0=strong phase-encode truncation)
            
        Returns:
            mask: Binary mask for k-space truncation (H, W)
        """
        H, W = spatial_shape
        center_h, center_w = H // 2, W // 2
        
        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(H, device=torch.device('cpu')),
            torch.arange(W, device=torch.device('cpu')),
            indexing='ij'
        )
        y = y.float() - center_h
        x = x.float() - center_w
        
        if truncation_type == 'circular':
            # Circular truncation with optional anisotropy
            radius_h = (H * truncation_factor) / 2
            radius_w = (W * truncation_factor) / 2
            
            # Apply anisotropy (reduce radius in phase-encode direction)
            if anisotropy > 0:
                radius_h *= (1 - anisotropy * 0.5)  # Reduce phase-encode radius
                
            distance = (y / radius_h) ** 2 + (x / radius_w) ** 2
            mask = (distance <= 1.0).float()
            
        elif truncation_type == 'rectangular':
            # Rectangular truncation
            half_h = int((H * truncation_factor) / 2)
            half_w = int((W * truncation_factor) / 2)
            
            # Apply anisotropy by reducing height
            if anisotropy > 0:
                half_h = int(half_h * (1 - anisotropy * 0.5))
                
            mask = torch.zeros((H, W))
            mask[center_h - half_h:center_h + half_h + 1, 
                 center_w - half_w:center_w + half_w + 1] = 1.0
                 
        elif truncation_type == 'elliptical':
            # Elliptical truncation
            radius_h = (H * truncation_factor) / 2
            radius_w = (W * truncation_factor) / 2
            
            # Apply anisotropy
            if anisotropy > 0:
                radius_h *= (1 - anisotropy * 0.3)
                
            # Elliptical equation
            distance = (y / radius_h) ** 2 + (x / radius_w) ** 2
            mask = (distance <= 1.0).float()
            
        else:
            raise ValueError(f"Unknown truncation type: {truncation_type}")
        
        return mask

    def apply_gibbs_ringing(self, kspace, truncation_factor_range=(0.6, 0.9), 
                           truncation_type='circular', anisotropy=0.3):
        """
        Apply Gibbs ringing artifacts by k-space truncation.
        
        Args:
            kspace: Complex k-space data (coil, height, width, 2)
            truncation_factor_range: Range of truncation factors (min, max)
            truncation_type: Type of truncation pattern
            anisotropy: Directional truncation bias
            
        Returns:
            truncated_kspace: k-space with truncation artifacts
        """
        # Validate input shape
        if len(kspace.shape) != 4 or kspace.shape[-1] != 2:
            raise ValueError(f"Expected kspace shape (coil, height, width, 2), got {kspace.shape}")
        
        debug = getattr(self.args, 'mr_aug_debug', False) if hasattr(self, 'args') else False
        if debug:
            print(f"Gibbs ringing - input kspace shape: {kspace.shape}")
        # Randomly sample truncation factor
        truncation_factor = self.rng.uniform(
            truncation_factor_range[0], 
            truncation_factor_range[1]
        )
        
        # Create truncation mask
        # Pass only spatial dimensions (height, width) to create_truncation_mask
        spatial_shape = kspace.shape[-3:-1]  # (height, width) from (coil, height, width, 2)
        mask = self.create_truncation_mask(
            spatial_shape,
            truncation_factor,
            truncation_type,
            anisotropy
        )
        
        # Move mask to same device as kspace
        if kspace.is_cuda:
            mask = mask.cuda()
            
        # Apply mask to all coils
        # kspace shape: (coil, height, width, 2)
        # mask shape: (height, width) -> broadcast to (1, height, width, 2)
        # Fix: Expand mask to match the real/imaginary dimension
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, height, width, 1)
        mask = mask.expand(-1, -1, -1, 2)       # (1, height, width, 2)
        
        if debug:
            print(f"Gibbs ringing - mask shape after reshape: {mask.shape}")
            print(f"Broadcasting: kspace {kspace.shape} * mask {mask.shape}")
        
        # Apply truncation
        truncated_kspace = kspace * mask
        
        if debug:
            print(f"Gibbs ringing - output shape: {truncated_kspace.shape}")
        
        return truncated_kspace

    def apply_kspace_noise(self, kspace, noise_type='rician', std_range=(0.005, 0.02)):
        """
        Apply noise injection directly to k-space data to simulate coil thermal noise.
        
        Args:
            kspace: Complex k-space data (coil, height, width, 2)
            noise_type: Type of noise distribution ('gaussian' or 'rician')
            std_range: Range of noise standard deviation (min, max)
            
        Returns:
            noisy_kspace: k-space with noise injection
        """
        # Validate input shape
        if len(kspace.shape) != 4 or kspace.shape[-1] != 2:
            raise ValueError(f"Expected kspace shape (coil, height, width, 2), got {kspace.shape}")
        
        debug = getattr(self.args, 'mr_aug_debug', False) if hasattr(self, 'args') else False
        if debug:
            print(f"K-space noise - input shape: {kspace.shape}")
        
        # Randomly sample noise standard deviation
        noise_std = self.rng.uniform(std_range[0], std_range[1])
        
        # Convert to complex tensor for easier manipulation
        kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        
        # Estimate data magnitude for SNR-aware noise scaling
        data_magnitude = torch.abs(kspace_complex).mean()
        scaled_noise_std = noise_std * data_magnitude
        
        if debug:
            print(f"K-space noise - type: {noise_type}, std: {noise_std:.6f}, scaled_std: {scaled_noise_std:.6f}")
        
        if noise_type == 'gaussian':
            # Add complex Gaussian noise
            # Generate independent noise for real and imaginary parts
            noise_real = torch.normal(0, scaled_noise_std, kspace_complex.shape, 
                                    device=kspace_complex.device, dtype=kspace_complex.real.dtype)
            noise_imag = torch.normal(0, scaled_noise_std, kspace_complex.shape,
                                    device=kspace_complex.device, dtype=kspace_complex.real.dtype)
            noise_complex = torch.complex(noise_real, noise_imag)
            
            # Add noise to k-space
            noisy_kspace_complex = kspace_complex + noise_complex
            
        elif noise_type == 'rician':
            # Apply Rician noise by adding Gaussian noise then taking magnitude
            # This simulates the magnitude-only nature of MRI signal detection
            
            # Add Gaussian noise to both real and imaginary parts
            noise_real = torch.normal(0, scaled_noise_std, kspace_complex.shape,
                                    device=kspace_complex.device, dtype=kspace_complex.real.dtype)
            noise_imag = torch.normal(0, scaled_noise_std, kspace_complex.shape,
                                    device=kspace_complex.device, dtype=kspace_complex.real.dtype)
            
            # Add noise to get intermediate noisy complex signal
            noisy_real = kspace_complex.real + noise_real
            noisy_imag = kspace_complex.imag + noise_imag
            
            # Apply Rician distribution: magnitude follows Rician, phase is preserved
            original_phase = torch.angle(kspace_complex)
            
            # New magnitude follows Rician distribution (approximated by noisy magnitude)
            rician_magnitude = torch.sqrt(noisy_real**2 + noisy_imag**2)
            
            # Reconstruct complex signal with Rician magnitude and preserved phase
            noisy_kspace_complex = rician_magnitude * torch.exp(1j * original_phase)
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Convert back to real/imaginary format
        noisy_kspace = torch.stack([noisy_kspace_complex.real, noisy_kspace_complex.imag], dim=-1)
        
        if debug:
            original_power = torch.mean(torch.abs(kspace_complex)**2)
            noisy_power = torch.mean(torch.abs(noisy_kspace_complex)**2)
            snr_db = 10 * torch.log10(original_power / (noisy_power - original_power + 1e-10))
            print(f"K-space noise - estimated SNR: {snr_db:.2f} dB")
            print(f"K-space noise - output shape: {noisy_kspace.shape}")
        
        return noisy_kspace

    def apply_kspace_streaks(self, kspace, density_range=(0.01, 0.05)):
        """
        Apply realistic motion streak artifacts by simulating phase encoding line corruption.
        Creates sparse corrupted k-space lines that manifest as motion streaks in the image.
        
        Args:
            kspace: Complex k-space data (coil, height, width, 2)
            density_range: Range of corrupted line fraction (min, max)
            
        Returns:
            streaked_kspace: k-space with motion streak artifacts
        """
        # Validate input shape
        if len(kspace.shape) != 4 or kspace.shape[-1] != 2:
            raise ValueError(f"Expected kspace shape (coil, height, width, 2), got {kspace.shape}")
        
        debug = getattr(self.args, 'mr_aug_debug', False) if hasattr(self, 'args') else False
        if debug:
            print(f"K-space streaks - input shape: {kspace.shape}")
        
        # Randomly sample streak density
        streak_density = self.rng.uniform(density_range[0], density_range[1])
        
        # Convert to complex tensor for easier manipulation
        kspace_complex = torch.complex(kspace[..., 0], kspace[..., 1])
        
        # Get k-space dimensions
        num_coils, height, width = kspace_complex.shape
        
        # Calculate number of corrupted lines (much more reasonable)
        num_corrupted_lines = int(streak_density * height)  # Only corrupt phase encoding lines
        
        if num_corrupted_lines > 0:
            # Choose which direction to corrupt (usually phase encoding = height dimension)
            if self.rng.random() < 0.8:  # 80% chance for horizontal streaks (phase encoding corruption)
                # Corrupt horizontal lines (entire rows)
                corrupted_lines = self.rng.choice(height, size=num_corrupted_lines, replace=False)
                
                for line_idx in corrupted_lines:
                    # Motion corruption strategies:
                    corruption_type = self.rng.choice(['duplicate', 'interpolate', 'zero', 'phase_shift'])
                    
                    if corruption_type == 'duplicate':
                        # Duplicate a nearby line (motion = same anatomy acquired twice)
                        source_line = max(0, min(height-1, line_idx + self.rng.randint(-3, 4)))
                        kspace_complex[:, line_idx, :] = kspace_complex[:, source_line, :]
                        
                    elif corruption_type == 'interpolate':
                        # Linear interpolation between neighboring lines
                        prev_line = max(0, line_idx - 1)
                        next_line = min(height - 1, line_idx + 1)
                        kspace_complex[:, line_idx, :] = 0.5 * (kspace_complex[:, prev_line, :] + 
                                                               kspace_complex[:, next_line, :])
                        
                    elif corruption_type == 'zero':
                        # Missing line (severe motion artifact)
                        kspace_complex[:, line_idx, :] = 0.0
                        
                    elif corruption_type == 'phase_shift':
                        # Phase shift due to motion (more realistic)
                        phase_shift = self.rng.uniform(-np.pi/4, np.pi/4)  # ±45 degrees max
                        phase_factor = torch.tensor(np.exp(1j * phase_shift), 
                                                   device=kspace_complex.device, 
                                                   dtype=kspace_complex.dtype)
                        kspace_complex[:, line_idx, :] *= phase_factor
            
            else:  # 20% chance for vertical streaks (readout corruption - less common)
                # Corrupt vertical lines (entire columns) 
                corrupted_lines = self.rng.choice(width, size=min(num_corrupted_lines, width//10), replace=False)
                
                for line_idx in corrupted_lines:
                    # Similar corruption but for columns
                    corruption_type = self.rng.choice(['duplicate', 'interpolate', 'phase_shift'])
                    
                    if corruption_type == 'duplicate':
                        source_line = max(0, min(width-1, line_idx + self.rng.randint(-2, 3)))
                        kspace_complex[:, :, line_idx] = kspace_complex[:, :, source_line]
                        
                    elif corruption_type == 'interpolate':
                        prev_line = max(0, line_idx - 1) 
                        next_line = min(width - 1, line_idx + 1)
                        kspace_complex[:, :, line_idx] = 0.5 * (kspace_complex[:, :, prev_line] + 
                                                               kspace_complex[:, :, next_line])
                        
                    elif corruption_type == 'phase_shift':
                        phase_shift = self.rng.uniform(-np.pi/6, np.pi/6)  # Smaller shift for readout
                        phase_factor = torch.tensor(np.exp(1j * phase_shift),
                                                   device=kspace_complex.device,
                                                   dtype=kspace_complex.dtype)
                        kspace_complex[:, :, line_idx] *= phase_factor
        
        # Convert back to real/imaginary format
        streaked_kspace = torch.stack([kspace_complex.real, kspace_complex.imag], dim=-1)
        
        if debug:
            print(f"K-space streaks - density: {streak_density:.4f}, corrupted_lines: {num_corrupted_lines}")
            print(f"K-space streaks - output shape: {streaked_kspace.shape}")
        
        return streaked_kspace

    def apply_bias_field(self, im, polynomial_order=3, intensity_range=(0.9, 1.1)):
        """
        Apply coil sensitivity bias field to simulate smooth shading variations in MRI.
        
        Args:
            im: Complex image data (coil, height, width, 2) or (height, width, 2)
            polynomial_order: Order of polynomial basis functions (higher = smoother)
            intensity_range: Range of bias field multiplier (min, max)
            
        Returns:
            biased_im: Image with bias field applied
        """
        debug = getattr(self.args, 'mr_aug_debug', False) if hasattr(self, 'args') else False
        
        # Handle both single-coil and multi-coil data
        if len(im.shape) == 3:
            # Single-coil (height, width, 2)
            height, width = im.shape[0], im.shape[1]
            is_single_coil = True
        elif len(im.shape) == 4:
            # Multi-coil (coil, height, width, 2)
            height, width = im.shape[1], im.shape[2]
            is_single_coil = False
        else:
            raise ValueError(f"Expected image shape (coil, H, W, 2) or (H, W, 2), got {im.shape}")
        
        if debug:
            print(f"Bias field - input shape: {im.shape}, single_coil: {is_single_coil}")
        
        # Generate normalized coordinate grids [-1, 1]
        y_coords = torch.linspace(-1, 1, height, device=im.device, dtype=im.dtype)
        x_coords = torch.linspace(-1, 1, width, device=im.device, dtype=im.dtype)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Generate polynomial basis functions up to specified order
        bias_field = torch.zeros_like(Y)
        
        # Generate random coefficients for polynomial terms
        num_terms = 0
        for i in range(polynomial_order + 1):
            for j in range(polynomial_order + 1 - i):
                if i + j <= polynomial_order:
                    # Random coefficient for this polynomial term
                    coeff = self.rng.uniform(-1.0, 1.0)
                    
                    # Add polynomial term: coeff * x^i * y^j
                    if i == 0 and j == 0:
                        # Constant term - use larger weight for overall scaling
                        bias_field += coeff * 0.5
                    else:
                        bias_field += coeff * (X ** i) * (Y ** j)
                    
                    num_terms += 1
        
        # Normalize bias field and scale to desired intensity range
        bias_field = bias_field / num_terms  # Normalize by number of terms
        
        # Map to intensity range [min, max]
        min_intensity, max_intensity = intensity_range
        bias_range = max_intensity - min_intensity
        
        # Apply sigmoid-like normalization to keep values reasonable
        bias_field = torch.tanh(bias_field)  # Maps to [-1, 1]
        
        # Add sinusoidal stripe component for structured banding artifacts
        stripe_prob = getattr(self.args, 'mr_aug_bias_stripe_prob', 0.5)
        if self.rng.uniform() < stripe_prob:
            # Randomly choose stripe parameters
            stripe_freq_min = getattr(self.args, 'mr_aug_bias_stripe_freq_min', 5.0)
            stripe_freq_max = getattr(self.args, 'mr_aug_bias_stripe_freq_max', 40.0)
            stripe_amp_min = getattr(self.args, 'mr_aug_bias_stripe_amp_min', 0.005)
            stripe_amp_max = getattr(self.args, 'mr_aug_bias_stripe_amp_max', 0.04)
            
            stripe_freq = self.rng.uniform(stripe_freq_min, stripe_freq_max)  # pixels per cycle
            stripe_phase = self.rng.uniform(0, 2 * np.pi)
            stripe_amp = self.rng.uniform(stripe_amp_min, stripe_amp_max)
            
            # Choose stripe direction (horizontal or vertical)
            if self.rng.uniform() < 0.5:
                # Horizontal stripes (varying along Y axis)
                stripe_pattern = stripe_amp * torch.sin(2 * np.pi * Y * height / stripe_freq + stripe_phase)
            else:
                # Vertical stripes (varying along X axis)
                stripe_pattern = stripe_amp * torch.sin(2 * np.pi * X * width / stripe_freq + stripe_phase)
            
            # Add stripe pattern to bias field before final normalization
            bias_field += stripe_pattern
            
            if debug:
                print(f"Bias field stripes - freq: {stripe_freq:.1f} pixels/cycle, amp: {stripe_amp:.4f}")
        
        # Final mapping to intensity range [min, max]
        bias_field = min_intensity + bias_range * (bias_field + 1) / 2  # Maps to [min, max]
        
        if debug:
            print(f"Bias field - range: [{bias_field.min():.4f}, {bias_field.max():.4f}]")
            print(f"Bias field - mean: {bias_field.mean():.4f}")
        
        # Convert complex image to magnitude and phase for processing
        im_complex = torch.complex(im[..., 0], im[..., 1])
        magnitude = torch.abs(im_complex)
        phase = torch.angle(im_complex)
        
        # Apply bias field multiplicatively to magnitude only
        if is_single_coil:
            # Single-coil: apply directly
            biased_magnitude = magnitude * bias_field
        else:
            # Multi-coil: expand bias field to match coil dimension
            bias_field_expanded = bias_field.unsqueeze(0).expand(im.shape[0], -1, -1)
            biased_magnitude = magnitude * bias_field_expanded
        
        # Reconstruct complex image with biased magnitude and preserved phase
        biased_im_complex = biased_magnitude * torch.exp(1j * phase)
        
        # Convert back to real/imaginary format
        biased_im = torch.stack([biased_im_complex.real, biased_im_complex.imag], dim=-1)
        
        if debug:
            magnitude_change = (biased_magnitude / (magnitude + 1e-10)).mean()
            print(f"Bias field - average magnitude change factor: {magnitude_change:.4f}")
            print(f"Bias field - output shape: {biased_im.shape}")
        
        return biased_im


class MRAugmentor:
    """
    High-level MRAugment wrapper with probability scheduling.
    Integrates with the training pipeline to apply augmentations based on current epoch.
    """
        
    def __init__(self, args, current_epoch_fn):
        """
        Args:
            args: Namespace with MRAugment configuration arguments
            current_epoch_fn: Function that returns current epoch for scheduling
        """
        self.current_epoch_fn = current_epoch_fn
        self.args = args
        self.aug_on = args.mr_aug_on
        if self.aug_on:
            self.augmentation_pipeline = MRAugmentationPipeline(args)
        self.max_train_resolution = getattr(args, 'mr_aug_max_train_resolution', None)
        self.last_printed_epoch = -1
        
    def __call__(self, kspace, target_size):
        """
        Apply MRAugment to k-space and generate corresponding target
        
        Args:
            kspace: torch.Tensor of shape (coil, height, width, 2) with real/imaginary
            target_size: tuple (height, width) for target shape
            
        Returns:
            augmented_kspace: torch.Tensor of shape (coil, height, width, 2)
            augmented_target: torch.Tensor of shape (height, width)
        """
        # Set augmentation probability
        if self.aug_on:
            p = self.schedule_p()
            current_epoch = self.current_epoch_fn()
            # Print only once per epoch
            if current_epoch != self.last_printed_epoch:
                # Show active augmentations
                active_augs = [name for name, weight in self.augmentation_pipeline.weight_dict.items() 
                              if weight > 0]
                p_start = getattr(self.args, 'mr_aug_start_prob', 0.0)
                p_end = getattr(self.args, 'mr_aug_end_prob', 0.5)
                print(f"[MR-Aug] Epoch {current_epoch}: probability = {p:.4f} (range: {p_start:.2f} → {p_end:.2f})")
                print(f"[MR-Aug] Active augmentations: {active_augs}")
                self.last_printed_epoch = current_epoch
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0
        
        # Apply augmentation if needed
        if self.aug_on and p > 0.0:
            kspace, target = self.augmentation_pipeline.augment_from_kspace(
                kspace,
                target_size=target_size,
                max_train_size=self.max_train_resolution
            )
        else:
            # If no augmentation, apply conditional cropping if needed (like official MRAugment)
            if self.max_train_resolution is not None:
                if kspace.shape[-3] > self.max_train_resolution[0] or kspace.shape[-2] > self.max_train_resolution[1]:
                    im = ifft2c(kspace)
                    im = complex_crop_if_needed(im, self.max_train_resolution)
                    kspace = fft2c(im)
            
            # Generate target from current kspace
            im = ifft2c(kspace)
            target = self.augmentation_pipeline.im_to_target(im, target_size)
                    
        return kspace, target
        
    def schedule_p(self):
        """Calculate current augmentation probability based on schedule (similar to mask augmentation)"""
        D = self.args.mr_aug_delay
        T = self.args.num_epochs
        t = self.current_epoch_fn()
        p_start = getattr(self.args, 'mr_aug_start_prob', 0.0)
        p_end = getattr(self.args, 'mr_aug_end_prob', 0.5)

        if t < D:
            return p_start
        
        # Calculate progress from delay epoch to end
        progress_epochs = T - D
        current_progress_epoch = t - D
        
        if progress_epochs <= 0:
            return p_end
        
        # Calculate progress ratio (0.0 to 1.0)
        progress = min(1.0, current_progress_epoch / progress_epochs)
        
        if self.args.mr_aug_schedule == 'constant':
            return p_end
        elif self.args.mr_aug_schedule == 'ramp':
            return p_start + (p_end - p_start) * progress
        elif self.args.mr_aug_schedule == 'exp':
            # Exponential scheduling: starts slow, accelerates
            exp_progress = 1.0 - exp(-5.0 * progress)  # -5 gives good curve shape
            return p_start + (p_end - p_start) * exp_progress
        else:
            return p_end

    def update_epoch(self, _epoch):
        """Update current epoch for scheduling (called from training loop)"""
        pass  # epoch is retrieved via current_epoch_fn


def create_mr_augmentor(args):
    """
    Factory function to create MRAugmentor if MRAugment is enabled
    
    Args:
        args: Namespace with MRAugment configuration
        
    Returns:
        MRAugmentor instance if enabled, None otherwise
    """
    if hasattr(args, 'mr_aug_on') and args.mr_aug_on:
        # Create function to get current epoch - will be set by training loop
        current_epoch_fn = lambda: getattr(args, '_current_epoch', 0)
        return MRAugmentor(args, current_epoch_fn)
    else:
        return None