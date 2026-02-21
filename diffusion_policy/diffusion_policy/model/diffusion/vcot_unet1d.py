import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D, ConditionalResidualBlock1D

class VCoTAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(context_dim, dim)
        self.value = nn.Linear(context_dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        # x: [B, dim, T], context: [B, context_dim]
        B, D, T = x.shape
        x = x.transpose(1, 2) # [B, T, D]
        
        q = self.query(x)
        k = self.key(context).unsqueeze(1) # [B, 1, D]
        v = self.value(context).unsqueeze(1) # [B, 1, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, T, 1]
        attn = attn.softmax(dim=-1)
        
        out = attn @ v # [B, T, D]
        out = out.transpose(1, 2) # [B, D, T]
        return x.transpose(1, 2) + out

class VCoTUNet1D(ConditionalUnet1D):
    def __init__(self, *args, context_dim=512, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a cross-attention block at the bottleneck
        self.cross_attn = VCoTAttentionBlock(kwargs['diffusion_step_embed_dim'], context_dim)

    def forward(self, sample, timestep, local_cond=None, global_cond=None, cross_cond=None):
        """
        cross_cond: Subgoal features from the vision encoder
        """
        # This follows the standard UNet flow but injects cross_cond via attention
        # For simplicity in this plugin version, we inject it into the global conditioning
        if cross_cond is not None:
            if global_cond is None:
                global_cond = cross_cond
            else:
                global_cond = torch.cat([global_cond, cross_cond], dim=-1)
        
        return super().forward(sample, timestep, local_cond=local_cond, global_cond=global_cond)
