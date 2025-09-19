import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, Union
import cv2
from scipy.ndimage import map_coordinates
import math

class MRIAugmentation:
    """
    Comprehensive MRI data augmentation for PA-NAFNet training on knee data.
    Designed for 8GB GPU limitation with batch_size=1 and variable image shapes.
    
    Strategies:
    1. Bias-field shading augmentation (models coil sensitivity variations)
    2. Crop-resize (forces scale robustness; PA learns global + local integration)
    3. Elastic deformation (light) (simulates motion/anatomical variability)
    """
    
    def __init__(self, 
                 prob_bias_field: float = 0.6,
                 prob_crop_resize: float = 0.7,
                 prob_elastic: float = 0.5,
                 target_ssim: float = 0.98,
                 memory_efficient: bool = True):
        """
        Initialize MRI augmentation pipeline.
        
        Args:
            prob_bias_field: Probability of applying bias field shading
            prob_crop_resize: Probability of applying crop-resize
            prob_elastic: Probability of applying elastic deformation
            target_ssim: Target SSIM (0.98+) - affects augmentation intensity
            memory_efficient: Enable memory optimization for 8GB GPU
        """
        self.prob_bias_field = prob_bias_field
        self.prob_crop_resize = prob_crop_resize
        self.prob_elastic = prob_elastic
        self.target_ssim = target_ssim
        self.memory_efficient = memory_efficient
        
        # Intensity scaling based on target SSIM - higher SSIM requires gentler augmentation
        # For SSIM 0.96, we can use more aggressive augmentation than 0.98
        self.intensity_scale = max(0.4, 1.0 - (target_ssim - 0.94) * 5)
        
    def bias_field_shading(self, random_prob: float, image: torch.Tensor, 
                          strength_range: Tuple[float, float] = (0.6, 1.4)) -> torch.Tensor:
        """
        Apply bias field shading to simulate coil sensitivity variations.
        
        This is crucial for MRI as it models the non-uniform sensitivity of receiver coils,
        which is a major source of variation in real MRI data.
        
        Args:
            image: Input tensor of shape (B, C, H, W), (C, H, W) or (H, W)
            strength_range: Range for bias field strength
            
        Returns:
            Augmented image with bias field applied
        """
        if random_prob > self.prob_bias_field:
            return image
            
        original_shape = image.shape
        if len(original_shape) == 4:
            # (B, C, H, W) - squeeze batch dimension
            image = image.squeeze(0)
        elif len(original_shape) == 2:
            # (H, W) - add channel dimension
            image = image.unsqueeze(0)
            
        C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        # Create smooth bias field using low-frequency components
        # Scale strength based on target SSIM
        strength = random.uniform(*strength_range)
        strength = 1.0 + (strength - 1.0) * self.intensity_scale
        
        # Generate smooth bias field with different patterns
        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        
        # Combine multiple smooth patterns for realistic bias field
        patterns = []
        
        # Radial pattern (common in MRI)
        radial = torch.sqrt(X**2 + Y**2)
        patterns.append(torch.exp(-radial * random.uniform(0.5, 2.0)))
        
        # Polynomial patterns
        poly_x = random.uniform(0.1, 0.8) * self.intensity_scale
        poly_y = random.uniform(0.1, 0.8) * self.intensity_scale
        patterns.append(1.0 + poly_x * X + poly_y * Y)
        
        # Sinusoidal pattern for coil sensitivity simulation
        freq_x = random.uniform(0.5, 2.0) * self.intensity_scale
        freq_y = random.uniform(0.5, 2.0) * self.intensity_scale
        patterns.append(1.0 + 0.3 * torch.sin(freq_x * X * math.pi) * torch.sin(freq_y * Y * math.pi))
        
        # Combine patterns
        bias_field = patterns[0]
        for pattern in patterns[1:]:
            weight = random.uniform(0.2, 0.8) * self.intensity_scale
            bias_field = bias_field * (1 - weight) + pattern * weight
            
        # Normalize and apply strength
        bias_field = bias_field / bias_field.mean()
        bias_field = (bias_field - 1.0) * (strength - 1.0) + 1.0
        
        # Apply bias field to all channels
        augmented = image * bias_field.unsqueeze(0)
        
        # Memory cleanup for 8GB limitation
        if self.memory_efficient:
            del bias_field, patterns, X, Y
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        if len(original_shape) == 4:
            return augmented.unsqueeze(0)
        elif len(original_shape) == 2:
            return augmented.squeeze(0)
        else:
            return augmented
    
    def crop_resize_augmentation(self, random_prob: float, image: torch.Tensor, 
                               crop_scale_range: Tuple[float, float] = (0.65, 0.9)) -> torch.Tensor:
        """
        Apply crop-resize augmentation to force scale robustness.
        
        This helps PA-NAFNet learn to integrate both global context and local details,
        crucial for achieving high SSIM scores.
        
        Args:
            image: Input tensor of shape (B, C, H, W), (C, H, W) or (H, W)
            crop_scale_range: Range for crop scale factor
            
        Returns:
            Crop-resized image maintaining original dimensions
        """
        if random_prob > self.prob_crop_resize:
            return image
            
        original_shape = image.shape
        if len(original_shape) == 4:
            # (B, C, H, W) - squeeze batch dimension
            image = image.squeeze(0)
        elif len(original_shape) == 2:
            # (H, W) - add channel dimension
            image = image.unsqueeze(0)
            
        C, H, W = image.shape
        
        # Adaptive crop scale based on target SSIM - gentler for higher SSIM targets
        scale_min, scale_max = crop_scale_range
        scale_range = scale_max - scale_min
        adjusted_range = scale_range * self.intensity_scale
        scale = random.uniform(scale_max - adjusted_range, scale_max)
        
        # Calculate crop dimensions
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        
        # Random crop position
        start_h = random.randint(0, H - crop_h)
        start_w = random.randint(0, W - crop_w)
        
        # Crop
        cropped = image[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
        
        # Resize back to original size using high-quality interpolation
        resized = F.interpolate(cropped.unsqueeze(0), 
                              size=(H, W), 
                              mode='bicubic', 
                              align_corners=True,
                              antialias=True).squeeze(0)
        
        # Memory cleanup
        if self.memory_efficient:
            del cropped
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        if len(original_shape) == 4:
            return resized.unsqueeze(0)
        elif len(original_shape) == 2:
            return resized.squeeze(0)
        else:
            return resized
    
    def elastic_deformation(self, random_prob: float, image: torch.Tensor,
                          alpha_range: Tuple[float, float] = (0.8, 3.0),
                          sigma_range: Tuple[float, float] = (0.06, 0.15)) -> torch.Tensor:
        """
        Apply light elastic deformation to simulate motion and anatomical variability.
        
        This encourages PA to align structures and be robust to small anatomical variations.
        Light deformation is crucial to maintain high SSIM while adding useful variability.
        
        Args:
            image: Input tensor of shape (B, C, H, W), (C, H, W) or (H, W)
            alpha_range: Range for deformation strength
            sigma_range: Range for Gaussian kernel sigma (as fraction of image size)
            
        Returns:
            Elastically deformed image
        """
        if random_prob > self.prob_elastic:
            return image
        
        original_shape = image.shape
        if len(original_shape) == 4:
            # (B, C, H, W) - squeeze batch dimension
            image = image.squeeze(0)
        elif len(original_shape) == 2:
            # (H, W) - add channel dimension
            image = image.unsqueeze(0)
            
        C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        # Scale deformation parameters based on target SSIM and image size
        alpha = random.uniform(*alpha_range) * self.intensity_scale * min(H, W) / 384
        sigma = random.uniform(*sigma_range) * min(H, W)
        
        # Generate random displacement fields
        dx = np.random.randn(H, W).astype(np.float32) * alpha
        dy = np.random.randn(H, W).astype(np.float32) * alpha
        
        # Apply Gaussian smoothing to create smooth deformation
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply deformation to each channel
        deformed_channels = []
        for c in range(C):
            img_np = image[c].cpu().numpy()
            deformed = map_coordinates(img_np, indices, order=1, mode='reflect')
            deformed = deformed.reshape(H, W)
            deformed_channels.append(torch.from_numpy(deformed).to(device=device, dtype=dtype))
        
        deformed_image = torch.stack(deformed_channels, dim=0)
        
        # Memory cleanup
        if self.memory_efficient:
            del dx, dy, indices, deformed_channels
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        if len(original_shape) == 4:
            return deformed_image.unsqueeze(0)
        elif len(original_shape) == 2:
            return deformed_image.squeeze(0)
        else:
            return deformed_image
    
    def __call__(
            self, 
            image1: torch.Tensor, 
            image2: torch.Tensor, 
            image3: torch.Tensor, 
            target: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply comprehensive augmentation pipeline.
        
        Args:
            image: Input image tensor
            target: Optional target image (ground truth)
            
        Returns:
            Augmented image(s)
        """
        # Ensure input is on correct device and dtype
        if not isinstance(image1, torch.Tensor):
            image1 = torch.tensor(image1, dtype=torch.float32)
        if not isinstance(image2, torch.Tensor):
            image2 = torch.tensor(image2, dtype=torch.float32)
        if not isinstance(image3, torch.Tensor):
            image3 = torch.tensor(image3, dtype=torch.float32)
        
            
        # Store original device and move to GPU if available and memory permits
        original_device = image1.device
        if torch.cuda.is_available() and self.memory_efficient:
            try:
                image1 = image1.cuda()
                image2 = image2.cuda()
                image3 = image3.cuda()
                if target is not None:
                    target = target.cuda()
            except RuntimeError:
                # Fall back to CPU if GPU memory insufficient
                pass
        
        # Apply augmentations in sequence
        # Order matters: spatial -> intensity

        # Sample Random
        random_prob_elastic = random.random()
        random_prob_crop = random.random()
        random_prob_bias = random.random()
        
        # 1. Elastic deformation (light structural variation)
        augmented1 = self.elastic_deformation(random_prob_elastic, image1)
        augmented2 = self.elastic_deformation(random_prob_elastic, image2)
        augmented3 = self.elastic_deformation(random_prob_elastic, image3)
        
        # 2. Crop-resize (scale robustness)
        augmented1 = self.crop_resize_augmentation(random_prob_crop, augmented1)
        augmented2 = self.crop_resize_augmentation(random_prob_crop, augmented2)
        augmented3 = self.crop_resize_augmentation(random_prob_crop, augmented3)
        
        # 3. Bias field shading (intensity variation modeling coil sensitivity)
        augmented1 = self.bias_field_shading(random_prob_bias, augmented1)
        augmented2 = self.bias_field_shading(random_prob_bias, augmented2)
        augmented3 = self.bias_field_shading(random_prob_bias, augmented3)

        # Move back to original device
        augmented1 = augmented1.to(original_device)
        augmented2 = augmented2.to(original_device)
        augmented3 = augmented3.to(original_device)
        
        # Apply same spatial transforms to target if provided
        if target is not None:
            target_aug = self.elastic_deformation(random_prob_elastic, target)
            target_aug = self.crop_resize_augmentation(random_prob_crop, target_aug)
            # Note: Don't apply bias field to target (ground truth should remain clean)
            target_aug = target_aug.to(original_device)

        return augmented1, augmented2, augmented3, target_aug

class ScheduledAugmentation:
    """
    Augmentation scheduler optimized for SSIM 0.96 target on knee data.
    Uses more aggressive augmentation early, then gradual refinement.
    """
    
    def __init__(self, base_augmentation: MRIAugmentation, 
                 total_epochs: int,
                 warmup_epochs: int = 15,
                 peak_epoch: int = None,
                 final_intensity: float = 0.6,
                 # New MR-augment style parameters
                 schedule_type: str = 'legacy',
                 delay_epochs: int = 0,
                 start_prob: float = 0.0,
                 end_prob: float = 0.5,
                 prob_bias_field: float = 0.5,
                 prob_crop_resize: float = 0.5,
                 prob_elastic: float = 0.5,
                 ):
        """
        Initialize augmentation scheduler with support for both legacy and MR-augment style scheduling.
        
        Args:
            base_augmentation: Base augmentation instance
            total_epochs: Total training epochs
            warmup_epochs: Epochs for augmentation warmup (legacy mode only)
            peak_epoch: Epoch at which augmentation peaks (legacy mode only)
            final_intensity: Final intensity for fine-tuning (legacy mode only)
            schedule_type: 'legacy', 'constant', 'ramp', or 'exp' (new MR-augment style)
            delay_epochs: Number of epochs before augmentation starts (MR-augment style)
            start_prob: Starting augmentation probability (MR-augment style)
            end_prob: Ending augmentation probability (MR-augment style)
        """
        self.base_aug = base_augmentation
        self.total_epochs = total_epochs
        
        # Legacy parameters (for backward compatibility)
        self.warmup_epochs = warmup_epochs
        self.peak_epoch = peak_epoch or total_epochs // 2
        self.final_intensity = final_intensity
        self.original_intensity = base_augmentation.intensity_scale
        
        # New MR-augment style parameters
        self.schedule_type = schedule_type
        self.delay_epochs = delay_epochs
        self.start_prob = start_prob
        self.end_prob = end_prob
        
        # Store original probabilities for scaling
        self.original_prob_bias = prob_bias_field
        self.original_prob_crop = prob_crop_resize
        self.original_prob_elastic = prob_elastic
        
    def schedule_p(self, epoch: int) -> float:
        """
        Calculate current augmentation probability based on schedule (matches MR augment style).
        
        Args:
            epoch: Current epoch
            
        Returns:
            Augmentation probability multiplier [0.0 to 1.0]
        """
        D = self.delay_epochs
        T = self.total_epochs
        t = epoch
        p_start = self.start_prob
        p_end = self.end_prob

        if t < D:
            return p_start
        
        # Calculate progress from delay epoch to end
        progress_epochs = T - D
        current_progress_epoch = t - D
        
        if progress_epochs <= 0:
            return p_end
        
        # Calculate progress ratio (0.0 to 1.0)
        progress = min(1.0, current_progress_epoch / progress_epochs)
        
        if self.schedule_type == 'constant':
            return p_end
        elif self.schedule_type == 'ramp':
            return p_start + (p_end - p_start) * progress
        elif self.schedule_type == 'exp':
            # Exponential scheduling: starts slow, accelerates (matches MR augment)
            from math import exp
            exp_progress = 1.0 - exp(-5.0 * progress)  # -5 gives good curve shape
            return p_start + (p_end - p_start) * exp_progress
        else:
            # Legacy mode - will use old get_augmentation logic
            return 1.0  # Full intensity for legacy compatibility
    
    def get_augmentation(self, epoch: int) -> MRIAugmentation:
        """
        Get augmentation with scheduled intensity using unified scheduling approach.
        Supports both legacy 3-phase scheduling and new MR-augment style scheduling.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Augmentation instance with adjusted intensity and probabilities
        """
        if self.schedule_type == 'legacy':
            # Use original complex 3-phase scheduling for backward compatibility
            return self._legacy_get_augmentation(epoch)
        else:
            # Use new MR-augment style scheduling
            return self._mr_style_get_augmentation(epoch)
    
    def _legacy_get_augmentation(self, epoch: int) -> MRIAugmentation:
        """Legacy 3-phase scheduling for backward compatibility."""
        if epoch < self.warmup_epochs:
            # Gradual warmup - start more gently for stability
            intensity = self.original_intensity * (epoch / self.warmup_epochs) * 0.7
        elif epoch < self.peak_epoch:
            # Build to full intensity - 0.96 target allows more aggressive augmentation
            progress = (epoch - self.warmup_epochs) / (self.peak_epoch - self.warmup_epochs)
            intensity = self.original_intensity * (0.7 + 0.5 * progress)
        else:
            # Gradual decay but maintain higher final intensity for 0.96 target
            progress = (epoch - self.peak_epoch) / (self.total_epochs - self.peak_epoch)
            intensity = self.original_intensity * (1.2 - (1.2 - self.final_intensity) * progress)
        
        # Update intensity with bounds suitable for 0.96 target
        self.base_aug.intensity_scale = max(0.3, min(1.5, intensity))
        
        # Adaptive probability scheduling for 0.96 target
        if epoch < self.warmup_epochs:
            # Conservative probabilities during warmup
            self.base_aug.prob_bias_field = 0.2
            self.base_aug.prob_crop_resize = 0.2
            self.base_aug.prob_elastic = 0.1
        elif epoch < self.peak_epoch:
            # Full probabilities during main training
            self.base_aug.prob_bias_field = 0.5
            self.base_aug.prob_crop_resize = 0.5
            self.base_aug.prob_elastic = 0.3
        else:
            # Moderate probabilities for fine-tuning
            self.base_aug.prob_bias_field = 0.4
            self.base_aug.prob_crop_resize = 0.2
            self.base_aug.prob_elastic = 0.1
            
        return self.base_aug
    
    def _mr_style_get_augmentation(self, epoch: int) -> MRIAugmentation:
        """New MR-augment style scheduling - simple and flexible."""
        # Get current probability multiplier from schedule_p
        prob_multiplier = self.schedule_p(epoch)
        
        # Scale original probabilities uniformly
        self.base_aug.prob_bias_field = self.original_prob_bias * prob_multiplier
        self.base_aug.prob_crop_resize = self.original_prob_crop * prob_multiplier
        self.base_aug.prob_elastic = self.original_prob_elastic * prob_multiplier
        
        # Scale intensity_scale proportionally to probability
        # Keep intensity in reasonable bounds while allowing scheduling control
        intensity_multiplier = 0.5 + 0.5 * prob_multiplier  # Maps [0,1] -> [0.5, 1.0]
        self.base_aug.intensity_scale = self.original_intensity * intensity_multiplier
        
        return self.base_aug