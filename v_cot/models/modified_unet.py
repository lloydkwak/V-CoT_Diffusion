import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Union, Optional, Dict
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D, ConditionalResidualBlock1D

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dim_head: int = 64):
        """
        Standard Cross-Attention layer to attend to spatial goal features.
        """
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D_q] (Action trajectory features)
        context: [B, N, D_c] (Spatial goal features)
        """
        b, t, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        # Core Attention Mechanism
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = einops.rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)

class ModifiedConditionalUnet1D(ConditionalUnet1D):
    def __init__(self, 
                 input_dim: int, 
                 global_cond_dim: int, 
                 subgoal_feat_dim: int = 512,
                 use_cross_attn: bool = True,
                 **kwargs):
        """
        Modified 1D U-Net with optional Cross-Attention for Visual Subgoals.
        """
        super().__init__(input_dim=input_dim, global_cond_dim=global_cond_dim, **kwargs)
        
        self.use_cross_attn = use_cross_attn
        
        if self.use_cross_attn:
            # Inject Cross-Attention at the bottleneck (Mid-block)
            # Mid-dim is the highest dimension in the U-Net hierarchy
            mid_dim = kwargs.get('down_dims', [256, 512, 1024])[-1]
            self.subgoal_attn = CrossAttention(query_dim=mid_dim, context_dim=subgoal_feat_dim)

    def forward(self, 
                sample: torch.Tensor, 
                timestep: Union[torch.Tensor, float, int], 
                global_cond: Optional[torch.Tensor] = None, 
                subgoal_spatial_feat: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        sample: [B, T, input_dim] (Noisy action sequence)
        subgoal_spatial_feat: [B, N, C] (Extracted from VisionEncoder)
        """
        # 1. Time & Global Conditioning
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # 2. Encoding (Down-sampling)
        x = einops.rearrange(sample, 'b t d -> b d t')
        h_list = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h_list.append(x)
            x = downsample(x)

        # 3. Bottleneck (Mid-modules)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # --- KEY MODIFICATION: Cross-Attention with Subgoal ---
        if self.use_cross_attn and subgoal_spatial_feat is not None:
            x = einops.rearrange(x, 'b c t -> b t c')
            x = x + self.subgoal_attn(x, subgoal_spatial_feat)
            x = einops.rearrange(x, 'b t c -> b c t')
        # -----------------------------------------------------

        # 4. Decoding (Up-sampling)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h_list.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b d t -> b t d')
        return x
