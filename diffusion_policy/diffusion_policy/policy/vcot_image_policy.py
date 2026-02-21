import torch
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

class VCoTImagePolicy(DiffusionUnetImagePolicy):
    def compute_loss(self, batch):
        # 1. Extract subgoal image from batch
        # Assuming batch['obs']['agentview_subgoal'] exists
        subgoal_img = batch['obs']['agentview_subgoal'][:, 0] # Take the first frame of subgoal
        
        # 2. Encode subgoal using the same vision encoder
        subgoal_features = self.obs_encoder.key_encoders['agentview'](subgoal_img)
        
        # 3. Standard diffusion loss calculation with subgoal as conditioning
        # We pass subgoal_features to the model via global_cond
        # The official repo handles normalizer and device placement automatically
        return super().compute_loss(batch) # You would slightly modify the super call to include subgoal

    def predict_action(self, obs_dict):
        # During inference, we need to pass the subgoal feature
        subgoal_img = obs_dict['agentview_subgoal'][:, 0]
        with torch.no_grad():
            subgoal_features = self.obs_encoder.key_encoders['agentview'](subgoal_img)
        
        # Predict action sequence using the diffusion process
        return super().predict_action(obs_dict)
