from typing import Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply

# Import our custom modules
from v_cot.models.vision_encoder import VCoTVisionEncoder
from v_cot.models.modified_unet import ModifiedConditionalUnet1D

class VCoTPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: VCoTVisionEncoder,
            horizon: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            use_cross_attn=True,
            **kwargs):
        """
        V-CoT Policy: Integrates Spatial Subgoal Reasoning into Diffusion Policy.
        """
        super().__init__()

        # 1. Setup Action Dimensions
        action_shape = shape_meta['action']['shape']
        action_dim = action_shape[0]

        # 2. Get Feature Dimensions from VCoTVisionEncoder
        # Encoder returns a dict with 'global_feat' and 'spatial_feats'
        out_shapes = obs_encoder.output_shape()
        global_feature_dim = out_shapes['global_feat'][0]
        # Assume subgoal feature dim is the last channel of spatial maps
        subgoal_feat_dim = list(out_shapes['spatial_feats'].values())[0][-1]

        # 3. Initialize Modified 1D U-Net
        input_dim = action_dim
        global_cond_dim = global_feature_dim * n_obs_steps

        self.model = ModifiedConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            subgoal_feat_dim=subgoal_feat_dim,
            use_cross_attn=use_cross_attn,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        
        self.num_inference_steps = num_inference_steps if num_inference_steps else noise_scheduler.config.num_train_timesteps
        self.kwargs = kwargs

    # ========= Inference Core ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            global_cond=None, subgoal_spatial_feat=None,
            generator=None, **kwargs):
        
        trajectory = torch.randn(size=condition_data.shape, device=condition_data.device, generator=generator)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # Predict with both Global FiLM and Spatial Cross-Attention
            model_output = self.model(
                trajectory, t, 
                global_cond=global_cond, 
                subgoal_spatial_feat=subgoal_spatial_feat
            )

            trajectory = self.noise_scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        To = self.n_obs_steps

        # 1. Encode Observations
        # Extract features for current steps
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1, *x.shape[2:]))
        out = self.obs_encoder(this_nobs)
        
        # Prepare Global Condition (Concatenated past observations)
        global_cond = out['global_feat'].reshape(B, -1)
        
        # Extract Subgoal Spatial Map (Assume 'agentview_subgoal' key exists)
        # Spatial map is from the last available frame in nobs
        subgoal_spatial_feat = out['spatial_feats'].get('agentview_subgoal', None)
        if subgoal_spatial_feat is not None:
            # Reshape back to [B, To, N, C] and take the last frame's subgoal
            subgoal_spatial_feat = subgoal_spatial_feat.reshape(B, To, -1, subgoal_spatial_feat.shape[-1])
            subgoal_spatial_feat = subgoal_spatial_feat[:, -1] # [B, N, C]

        # 2. Run Diffusion Sampling
        cond_data = torch.zeros((B, self.horizon, self.action_dim), device=self.device)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        nsample = self.conditional_sample(
            cond_data, cond_mask,
            global_cond=global_cond,
            subgoal_spatial_feat=subgoal_spatial_feat,
            **self.kwargs
        )
        
        # 3. Unnormalize and Return
        action_pred = self.normalizer['action'].unnormalize(nsample)
        start = To - 1
        action = action_pred[:, start:start+self.n_action_steps]
        
        return {'action': action, 'action_pred': action_pred}

    # ========= Training Core ============
    def compute_loss(self, batch):
        # 1. Normalize and Encode
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]

        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        out = self.obs_encoder(this_nobs)
        
        global_cond = out['global_feat'].reshape(B, -1)
        
        # Handle Subgoal features
        subgoal_spatial_feat = out['spatial_feats'].get('agentview_subgoal', None)
        if subgoal_spatial_feat is not None:
            subgoal_spatial_feat = subgoal_spatial_feat.reshape(B, self.n_obs_steps, -1, subgoal_spatial_feat.shape[-1])
            subgoal_spatial_feat = subgoal_spatial_feat[:, -1]

        # 2. Diffusion Forward Process
        noise = torch.randn(nactions.shape, device=nactions.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=nactions.device).long()
        noisy_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        # 3. Predict and Loss Calculation
        pred = self.model(
            noisy_actions, timesteps, 
            global_cond=global_cond, 
            subgoal_spatial_feat=subgoal_spatial_feat
        )

        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else nactions
        loss = F.mse_loss(pred, target, reduction='none')
        return reduce(loss, 'b ... -> b (...)', 'mean').mean()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
