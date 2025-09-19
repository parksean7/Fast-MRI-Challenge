import numpy as np
import torch
from torch import nn
from utils import fastmri
from utils.common.utils import center_crop
from typing import Tuple


class accelerationRouter(nn.Module):
    def __init__(self, threshold: float = 4.0):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, mask: torch.Tensor) -> int:
        # Count non-zero values
        non_zero_count = torch.count_nonzero(mask).item()
        total_count = mask.shape[-2]  # k_width

        # Calculate acceleration rate
        if non_zero_count > 0:
            acc_rate = total_count / non_zero_count
        else:
            acc_rate = float('inf')  # Handle case where all values are zero

        # Return classification based on threshold
        return 0 if acc_rate < self.threshold else 1

class classifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves the spatial dimensions
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves again
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves again
        )

        # Add an adaptive pooling layer to handle variable input sizes
        # and produce a fixed-size output for the linear layer.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 classes: 0 for brain, 1 for knee
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.
        Args:
            x: Input tensor of shape (B, 1, H, W), where B is the batch size.
        Returns:
            Output tensor of shape (B, 2) with raw logits for each class.
        """
        # Ensure input is 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc_layers(x)
        return x

class anatomyClassifier_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = classifierCNN()

    def forward(self, kspace: torch.Tensor, train_mode=False):
        """
        Classify anatomy type by looking at the aliased image generated from undersampled kspace
        Uses internal pretrained CNN classifier
        
        Args:
            kspace: Input tensor of shape (coils, height, width, 2) or (batch, coils, height, width, 2)
            
        Returns:
            (train_mode == True) Output tensor of shape (batch, 2) (raw logits)
            (train_mode == False) Output integer: 0 for brain, 1 for knee
            
        """
        # Generate the aliased image from k-space
        image = fastmri.ifft2c(kspace)
        image = fastmri.rss(fastmri.complex_abs(image), dim=1)
        image = center_crop(image, 384, 384)
        image = image.unsqueeze(1)  # Add channel dimension after center crop

        # Normalize
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = torch.zeros_like(image)

        # Get the model's prediction
        output = self.model(image)
        if train_mode:
            return output
        else:
            _, predicted = torch.max(output.data, 1)
            return predicted.item()

class anatomyClassifier_Intensity(nn.Module):
    """
    Simple and robust anatomy classifier using intensity statistics.
    Based on analysis showing perfect separation between brain and knee MRI data
    using standard deviation of image intensities
    """
    def __init__(self, threshold: float = 0.0000395):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, kspace: torch.Tensor, debug_mode=False):
        """
        Classify anatomy type based on intensity statistics.
        
        Args:
            data: Input tensor of shape (coils, height, width) or (batch, coils, height, width)
            
        Returns:
            int: 0 for brain, 1 for knee
            (debug_mode) float: intensity
        """  
        # Generate the aliased image from k-space
        image = fastmri.ifft2c(kspace)
        image = fastmri.rss(fastmri.complex_abs(image), dim=1)
        image = center_crop(image, 384, 384)
        image = image.unsqueeze(1)  # Add channel dimension after center crop
        
        # Convert to numpy for statistics calculation
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().numpy()
        else:
            image_np = image
        
        # Calculate standard deviation of entire volume
        std_intensity = np.std(image_np)
        
        # Classify based on threshold
        # Brain has higher intensity variation (std > threshold)
        # Knee has lower intensity variation (std <= threshold)
        pred = 0 if std_intensity > self.threshold else 1
        if debug_mode:
            return pred, std_intensity
        else:
            return pred

class anatomyClassifier_Shape(nn.Module):
    """
    Simple anatomy classifier using kspacce shape
    """
    def __init__(self, brain_h:int=768, knee_h:int=640):
        super().__init__()
        self.brain_height = brain_h
        self.knee_height = knee_h
    
    def forward(self, kspace: torch.Tensor) -> int:
        """
        Classify anatomy type based on kspace shape
        
        Args:
            data: Input tensor of shape (slices, height, width, 2) or (batch, slices, height, width, 2)
            
        Returns:
            int: 0 for brain, 1 for knee, -1 for uncertain
        """
        if kspace.shape[-3] == self.brain_height:
            return 0
        elif kspace.shape[-3] == self.knee_height:
            return 1
        else:
            return -1

class anatomyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier_cnn = anatomyClassifier_CNN()
        self.classifier_shape = anatomyClassifier_Shape()

    def forward(self, kspace: torch.Tensor) -> int:
        """
        Classify anatomy type
        
        Args:
            data: Input tensor of shape (slices, height, width, 2) or (batch, slices, height, width, 2)
            
        Returns:
            int: 0 for brain, 1 for knee
        """
        # If anatomyClassifier_Shape makes prediction, trust
        pred_shape = self.classifier_shape(kspace)
        if (pred_shape != -1):
            return pred_shape
        
        # But if it doesn't predict, look at CNN classifier
        pred_cnn = self.classifier_cnn(kspace, train_mode=False)
        return pred_cnn
