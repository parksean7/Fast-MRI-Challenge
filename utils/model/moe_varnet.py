import torch
from torch import nn
from utils.model.block import SensitivityModel, VarNetBlock, NormUnet, AdaptiveSAGUnet, SAGVarNetBlock
from utils.model.moe_router import anatomyRouter
from utils import fastmri
from utils.common.utils import center_crop


class VarNet(nn.Module):
    def __init__(
            self,
            sens_chans: int = 4,
            sens_pools: int = 4,
            num_cascades: int = 2,
            chans: int = 15,
            pools: int = 4,
            dc_mode:str = 'soft',
    ):
        super().__init__()
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools), dc_mode = dc_mode) for _ in range(num_cascades)]
        )
    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, recon: bool = True):
        # clone
        kspace_pred = masked_kspace.clone()

        # sensitivity map
        sens_maps = self.sens_net(masked_kspace, mask)
        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        
        # output
        if recon:
            result = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
            result = center_crop(result, 384, 384)
            return result
        else:
            return kspace_pred
    
    def sens_maps(self, masked_kspace: torch.Tensor, mask: torch.Tensor):
        return self.sens_net(masked_kspace, mask)

class SAGVarNet(nn.Module):
    """
    Self-Adaptive Gradient Variational Network based on PromptMR-plus.
    
    This model combines data consistency with SAG-based regularization using
    adaptive input buffers and latent state propagation across cascades.
    
    Target configuration: 6 cascades, 15 channels, 4 pools for SSIM > 0.98
    """

    def __init__(
        self,
        num_cascades: int = 6,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 15,
        pools: int = 4,
        n_buffer: int = 4,
        use_checkpoint: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (default: 6 for SAG).
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools: Number of downsampling and upsampling layers for sensitivity map U-Net.
            chans: Number of channels for cascade U-Net (default: 15 for SAG).
            pools: Number of downsampling and upsampling layers for cascade U-Net (default: 4 for SAG).
            n_buffer: Number of buffer components for SAG (default: 4).
            use_checkpoint: Use gradient checkpointing for memory efficiency.
        """
        super().__init__()

        self.num_cascades = num_cascades
        self.use_checkpoint = use_checkpoint
        
        # Use existing sensitivity map estimation
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        
        # Create SAG cascades
        self.cascades = nn.ModuleList([
            SAGVarNetBlock(
                AdaptiveSAGUnet(
                    chans=chans,
                    num_pools=pools,
                    adaptive_input=True,
                    n_buffer=n_buffer,
                    drop_prob=0.0,
                )
            ) for _ in range(num_cascades)
        ])

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_kspace: Input k-space data [B, N_coils, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1] or [B, N_coils, H, W, 1]
            
        Returns:
            result: Reconstructed image [B, H, W] (RSS of final image)
        """
        # Step 1: Estimate sensitivity maps (unchanged from VarNet)
        sens_maps = self.sens_net(masked_kspace, mask)
        
        # Step 2: Initialize with zero-filled reconstruction
        img_zf = fastmri.complex_mul(
            fastmri.ifft2c(masked_kspace), 
            fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)
        
        # Step 3: Initialize current image and latent state
        current_img = img_zf.clone()
        latent = img_zf.clone()
        
        # Step 4: SAG cascade iterations
        for cascade in self.cascades:
            if self.use_checkpoint:
                current_img, latent = torch.utils.checkpoint.checkpoint(
                    cascade, current_img, img_zf, latent, mask, sens_maps, 
                    use_reentrant=False
                )
            else:
                current_img, latent = cascade(current_img, img_zf, latent, mask, sens_maps)
        
        # Step 5: Final reconstruction (RSS like original VarNet)
        # current_img is [B, 1, H, W, 2], squeeze to get [B, H, W, 2]
        squeezed_img = current_img.squeeze(1)
        
        # Convert complex to real magnitude: [B, H, W, 2] -> [B, H, W]
        magnitude_img = fastmri.complex_abs(squeezed_img)
        
        # Always apply center crop to match original VarNet behavior
        # This ensures consistent 384x384 output size matching training expectations
        result = center_crop(magnitude_img, 384, 384)
        
        return result

class MoE_SAGVarNet(nn.Module):
    """
    Mixture-of-Expert (MoE) Architecture with anatomy-specific networks.

    Consisted of
        - anatomy router: classifies brain / knee and routes to appropriate expert
        - brain expert: SAG VarNet
        - knee expert: SAG VarNet
    """
    
    def __init__(
        self,
        brain_num_cascades: int = 8,
        brain_sens_chans: int = 4,
        brain_sens_pools: int = 4,
        brain_chans: int = 20,
        brain_pools: int = 4,
        brain_n_buffer: int = 4,
        brain_use_checkpoint: bool = False,
        knee_num_cascades: int = 8,
        knee_sens_chans: int = 4,
        knee_sens_pools: int = 4,
        knee_chans: int = 20,
        knee_pools: int = 4,
        knee_n_buffer: int = 4,
        knee_use_checkpoint: bool = False,
    ):
        super().__init__()

        self.anatomy_router = anatomyRouter()
        self.brain_expert = SAGVarNet(
            num_cascades=brain_num_cascades,
            sens_chans=brain_sens_chans,
            sens_pools=brain_sens_pools,
            chans=brain_chans,
            pools=brain_pools,
            n_buffer=brain_n_buffer,
            use_checkpoint=brain_use_checkpoint,
        )
        self.knee_expert = SAGVarNet(
            num_cascades=knee_num_cascades,
            sens_chans=knee_sens_chans,
            sens_pools=knee_sens_pools,
            chans=knee_chans,
            pools=knee_pools,
            n_buffer=knee_n_buffer,
            use_checkpoint=knee_use_checkpoint,
        )
    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_kspace: Input k-space data [B, N_coils, H, W, 2]
            mask: Sampling mask [B, 1, H, W, 1] or [B, N_coils, H, W, 1]
            
        Returns:
            result: Reconstructed image [B, H, W] (RSS of final image)
        """
        # Step 1: Classify Brain / Knee
        anatomy = self.anatomy_router(masked_kspace)

        # Step 2: Route to appropriate Expert
        if anatomy == 0: # brain
            result = self.brain_expert(masked_kspace, mask)
        else:
            result = self.knee_expert(masked_kspace, mask)
        
        return result
