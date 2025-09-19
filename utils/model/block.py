import torch
from torch import nn
from pathlib import Path
import math
from typing import List, Tuple

from utils.model.moe_router import anatomyRouter
from utils import fastmri
from utils.fastmri.data import transforms
from utils.fastmri.data.transforms import to_tensor
from utils.common.utils import center_crop
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)

class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        
        return output

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x

class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # get low frequency line locations and mask them out
        squeezed_mask = mask[:, 0, 0, :, 0]
        cent = squeezed_mask.shape[1] // 2
        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = transforms.batched_mask_center(masked_kspace, pad, pad + num_low_freqs)

        # convert to image space
        x = fastmri.ifft2c(x)
        x, b = self.chans_to_batch_dim(x)

        # estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)

        return x

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.
    """

    def __init__(self, model: nn.Module, dc_mode: str = 'soft'):
        super().__init__()
        self.model = model
        self.dc_mode = dc_mode
        if self.dc_mode == 'soft':
            self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        if self.dc_mode == 'soft':
            zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
            soft_dc = torch.where(mask.bool(), current_kspace - ref_kspace, zero) * self.dc_weight
            return current_kspace - soft_dc - model_term
        elif self.dc_mode == 'hard':
            return torch.where(mask.bool(), ref_kspace, model_term)
        else:
            raise ValueError(f"Unknown dc_mode: {self.dc_mode}")

class AdaptiveSAGUnet(nn.Module):
    """
    Self-Adaptive Gradient U-Net based on PromptMR-plus implementation.
    
    This U-Net supports adaptive input with buffer components for SAG.
    When adaptive_input is enabled, it takes 10 input channels:
    - 2 channels: current image (real, imag)
    - 8 channels: buffer [A^H*A*x_i, latent, x0, A^H*A*x_i-x0] x 2 (real, imag)
    
    Output channels are also 10 when adaptive input is enabled:
    - 2 channels: correction term (real, imag)
    - 8 channels: updated buffer components
    """
    
    def __init__(
        self,
        chans: int,
        num_pools: int,
        adaptive_input: bool = True,
        n_buffer: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            adaptive_input: Enable adaptive input with buffer components.
            n_buffer: Number of buffer components (default: 4 for SAG).
            in_chans: Base number of input channels (2 for complex).
            out_chans: Base number of output channels (2 for complex).
            drop_prob: Dropout probability.
        """
        super().__init__()
        
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer if adaptive_input else 0
        self.base_in_chans = in_chans
        self.base_out_chans = out_chans
        
        # Adjust channel counts for adaptive input
        if adaptive_input:
            actual_in_chans = in_chans * (1 + n_buffer)  # 2 * (1 + 4) = 10
            actual_out_chans = out_chans * (1 + n_buffer)  # 2 * (1 + 4) = 10
        else:
            actual_in_chans = in_chans
            actual_out_chans = out_chans
            
        self.unet = Unet(
            in_chans=actual_in_chans,
            out_chans=actual_out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm - handle both standard (2,4,6...) and adaptive input (10) channels
        b, c, h, w = x.shape
        
        # Flatten for computing statistics
        x_flat = x.view(b, c * h * w)
        mean = x_flat.mean(dim=1).view(b, 1, 1, 1)
        std = x_flat.std(dim=1).view(b, 1, 1, 1)
        
        # Avoid division by zero
        std = torch.clamp(std, min=1e-8)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor, buffer: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Current image [B, H, W, 2]
            buffer: Buffer components [B, 4, H, W, 2] when adaptive_input=True
            
        Returns:
            correction: Model correction term [B, H, W, 2]
            updated_latent: Updated latent component [B, H, W, 2] (or None if not adaptive)
        """
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")
        
        cc = x.shape[1]  # Number of base channels (should be 1 for single-coil)
        
        if self.adaptive_input and buffer is not None:
            # Concatenate current image with buffer: [current_img, ffx, latent, img_zf, ffx-img_zf]
            x_input = torch.cat([x, buffer], dim=1)  # [B, 5, H, W, 2]
        else:
            x_input = x

        # get shapes for unet and normalize
        x_input = self.complex_to_chan_dim(x_input)  # [B, 10, H, W] when adaptive
        x_input, mean, std = self.norm(x_input)
        x_input, pad_sizes = self.pad(x_input)

        x_output = self.unet(x_input)

        # get shapes back and unnormalize
        x_output = self.unpad(x_output, *pad_sizes)
        x_output = self.unnorm(x_output, mean, std)
        x_output = self.chan_complex_to_last_dim(x_output)  # [B, 5, H, W, 2] when adaptive

        if self.adaptive_input and buffer is not None:
            # Split output: [correction, _, latent, _]
            correction, _, updated_latent, _ = torch.split(
                x_output, [cc, cc, cc, x_output.shape[1] - 3*cc], dim=1
            )
        else:
            correction = x_output
            updated_latent = None

        return correction, updated_latent

class SAGVarNetBlock(nn.Module):
    """
    Model block for SAG-enabled variational network.
    
    This block implements the Self-Adaptive Gradient mechanism by:
    1. Computing buffer components: [A^H*A*x_i, latent, x0, A^H*A*x_i-x0]  
    2. Feeding current image + buffer to adaptive U-Net
    3. Applying SAG correction with data consistency
    4. Updating latent state for next cascade
    """

    def __init__(self, model: AdaptiveSAGUnet):
        """
        Args:
            model: AdaptiveSAGUnet for SAG correction computation.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def compute_sag_buffer(self, current_img: torch.Tensor, latent: torch.Tensor, 
                          img_zf: torch.Tensor, mask: torch.Tensor, 
                          sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Compute the SAG buffer components following PromptMR-plus implementation.
        
        Args:
            current_img: Current image estimate [B, 1, H, W, 2]
            latent: Current latent state [B, 1, H, W, 2]  
            img_zf: Zero-filled reconstruction [B, 1, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1] 
            sens_maps: Sensitivity maps [B, N_coils, H, W, 2]
            
        Returns:
            buffer: Buffer components [B, 4, H, W, 2]
                   [A^H*A*x_i, latent, x0, A^H*A*x_i-x0]
        """
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_img)
        
        # Compute A^H*A*x_i (forward-backward operation)
        current_kspace = self.sens_expand(current_img, sens_maps)
        ffx = self.sens_reduce(torch.where(mask.bool(), current_kspace, zero), sens_maps)
        
        # Create buffer: [A^H*A*x_i, latent, x0, A^H*A*x_i-x0]
        buffer = torch.cat([
            ffx,            # A^H*A*x_i
            latent,         # latent state s_i
            img_zf,         # x0 (zero-filled)
            ffx - img_zf    # A^H*A*x_i - x0
        ], dim=1)  # [B, 4, H, W, 2]
        
        return buffer, ffx

    def forward(
        self,
        current_img: torch.Tensor,
        img_zf: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            current_img: Current image estimate [B, 1, H, W, 2]
            img_zf: Zero-filled reconstruction [B, 1, H, W, 2] 
            latent: Current latent state [B, 1, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1]
            sens_maps: Sensitivity maps [B, N_coils, H, W, 2]
            
        Returns:
            img_pred: Updated image estimate [B, 1, H, W, 2]
            updated_latent: Updated latent state [B, 1, H, W, 2]
        """
        # Compute SAG buffer components
        buffer, ffx = self.compute_sag_buffer(current_img, latent, img_zf, mask, sens_maps)
        
        # Apply SAG model to get correction and updated latent
        correction, updated_latent = self.model(current_img, buffer)
        
        # Data consistency term
        soft_dc = (ffx - img_zf) * self.dc_weight
        
        # SAG update rule: x_{t+1} = x_t - DC - SAG_correction
        img_pred = current_img - soft_dc - correction
        
        return img_pred, updated_latent
