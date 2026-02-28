"""
Optimal Transport Flow Matching (OT-Flow) Policy.

This module implements a continuous-time Flow Matching policy, replacing the 
traditional DDPM/DDIM discrete-time diffusion processes. 
By learning the vector field (velocity) that transports a simple Gaussian base 
distribution to the empirical data distribution, this approach significantly 
accelerates inference via ODE solvers (e.g., Euler integration) and simplifies 
the training objective.

Note: The class name 'DiffusionUnetImagePolicy' and its constructor signature 
are preserved to ensure backward compatibility with existing Hydra configurations.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

# Note: DDPMScheduler is kept in the signature for config compatibility, 
# but the Flow Matching algorithm computes trajectories analytically without it.
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=10, # Defaults to 10 for rapid ODE solving in Flow Matching
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            **kwargs):
        super().__init__()

        # 1. Parse shape configurations
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # 2. Configure model input dimensions based on conditioning strategy
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        # 3. Initialize UNet Vector Field Estimator
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # Override num_inference_steps to a smaller value suitable for Flow Matching ODE
        if num_inference_steps is None or num_inference_steps > 20:
            self.num_inference_steps = 10 
        else:
            self.num_inference_steps = num_inference_steps
    
    # =========================================================================
    # Inference: Optimal Transport Flow Matching (ODE Solver)
    # =========================================================================
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        """
        Solves the probability flow ODE using Euler integration.
        Transports the initial noise distribution x_0 to the data distribution x_1.
        """
        model = self.model
        device = condition_data.device
        dtype = condition_data.dtype

        # Initial state x_0 ~ N(0, I)
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=dtype,
            device=device,
            generator=generator
        )
        
        steps = self.num_inference_steps
        dt = 1.0 / steps

        # Forward Euler Integration
        for i in range(steps):
            # t ranges from 0.0 to 1.0 (approaching data distribution)
            t_val = i * dt
            # Scale t by 1000 to match the standard embedding frequencies of UNet
            t_tensor = torch.full((trajectory.shape[0],), t_val * 1000, device=device, dtype=dtype)

            # Enforce observation constraints (Inpainting)
            trajectory[condition_mask] = condition_data[condition_mask]

            # Predict the vector field (velocity) v_t(x_t)
            model_output = model(trajectory, t_tensor, local_cond=local_cond, global_cond=global_cond)
            v_pred = model_output.sample if hasattr(model_output, 'sample') else model_output

            # Euler integration step: x_{t+dt} = x_t + v * dt
            trajectory = trajectory + v_pred * dt
        
        # Enforce exact boundary constraints at t=1
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Evaluates the policy to predict action chunks given current observations.
        """
        assert 'past_action' not in obs_dict 
        
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(B, -1)
            
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # =========================================================================
    # Training: Optimal Transport Vector Field Matching
    # =========================================================================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Computes the Flow Matching regression loss.
        Minimizes MSE between the predicted vector field and the optimal 
        transport target velocity (x_1 - x_0).
        """
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Inpainting mask generation
        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask

        # Flow Matching Formulation
        # x_1 is the target empirical data distribution
        x1 = trajectory
        # x_0 is the standard Gaussian prior
        x0 = torch.randn_like(x1)
        
        # Sample continuous time t uniformly from [0, 1]
        t = torch.rand((batch_size,), device=x1.device, dtype=x1.dtype)
        t_expand = t.view(-1, 1, 1)
        
        # Construct the interpolant x_t
        xt = (1.0 - t_expand) * x0 + t_expand * x1
        
        # The target vector field (velocity) points directly from x_0 to x_1
        target_v = x1 - x0

        # Enforce conditioning data directly onto x_t during training
        xt[condition_mask] = cond_data[condition_mask]
        
        # Scale t by 1000 for UNet temporal embeddings, consistent with diffusion models
        t_scaled = t * 1000
        
        # Predict the vector field
        model_output = self.model(xt, t_scaled, local_cond=local_cond, global_cond=global_cond)
        pred_v = model_output.sample if hasattr(model_output, 'sample') else model_output

        # Regression loss against the target vector field
        loss = F.mse_loss(pred_v, target_v, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
