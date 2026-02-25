"""
End-to-End Evaluation of Hierarchical Visual Chain-of-Thought (V-CoT) Architecture.

This script integrates:
1. High-Level VLM (InstructPix2Pix + LoRA): Generates visual subgoals (key-frames) 
   every N steps based on current observations and language instructions.
2. Low-Level Diffusion Policy: Executes 8-step action chunks to reach the 
   VLM-generated subgoals.

The evaluation is conducted in a fully autonomous, closed-loop manner within 
the Robosuite simulation environment.
"""

import os
import sys
import torch
import copy
import torch.nn.functional as F
import numpy as np
import cv2
import collections
import hydra
import wandb
import dill
from PIL import Image

# VLM and PEFT libraries
from diffusers import StableDiffusionInstructPix2PixPipeline
from peft import PeftModel

# Set project paths
project_root = "/workspace/diffusion_policy"
if project_root not in sys.path:
    sys.path.append(project_root)

import robosuite as suite
from robosuite.controllers import load_controller_config

# =====================================================================
# Global Configuration
# =====================================================================
# Model Checkpoints
DP_CKPT_PATH = "/workspace/data/outputs/2026-02-24/10-48-30/checkpoints/latest.ckpt"
VLM_BASE_MODEL = "timbrooks/instruct-pix2pix"
VLM_LORA_PATH = "/workspace/data/vlm_outputs/checkpoint-epoch-9"

DEVICE = "cuda:0"

# Control Hyperparameters
REPLAN_STEPS = 8      # Execute 8 steps per DP inference (Temporal smoothing)
VLM_FREQ = 24         # Update visual subgoal every 24 steps
MAX_TOTAL_STEPS = 600 
DP_GUIDANCE_SCALE = 1.2 
VLM_INFERENCE_STEPS = 20

# Task Suite
TASK_CONFIGS = {
    "Lift": "Pick up the red block.",
    "PickPlaceCan": "Pick up the red can and place it in the bin.",
    "NutAssemblySquare": "Pick up the square nut and place it on the peg."
}

# =====================================================================
# Utility Functions
# =====================================================================
def preprocess_image(img_raw, target_shape=(3, 84, 128)):
    """Normalizes and reshapes images for Diffusion Policy input."""
    if img_raw.shape[0] == 3:
        img_t = torch.from_numpy(img_raw.copy()).float()
    else:
        img_t = torch.from_numpy(img_raw.copy()).float().permute(2, 0, 1)
    
    if img_t.max() > 1.0:
        img_t = img_t / 255.0
        
    if img_t.shape != tuple(target_shape):
        img_t = F.interpolate(img_t.unsqueeze(0), size=target_shape[1:], 
                            mode='bilinear', align_corners=False, antialias=True).squeeze(0)
    return img_t

def get_vis_img(img_raw, flip=False):
    """Utility for formatted OpenCV visualization."""
    img = img_raw.copy()
    if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0: img = img * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (256, 256))
    if flip: img = cv2.flip(img, 0)
    return img

def predict_action(policy, obs_dict, cfg, w=1.2):
    """Infers action trajectory using the Diffusion Policy with CFG."""
    device = next(policy.parameters()).device
    action_dim = cfg.policy.shape_meta.action.shape[0] if hasattr(cfg.policy, 'shape_meta') else cfg.task.dataset_shape_meta.action.shape[0]
    n_obs_steps, horizon = cfg.n_obs_steps, cfg.horizon
    
    def get_cond(obs):
        valid_keys = policy.normalizer.params_dict.keys()
        filtered = {k: v for k, v in obs.items() if k in valid_keys or 'image' in k or 'subgoal' in k}
        nobs = policy.normalizer.normalize(filtered)
        this_nobs = {k: v[:, :n_obs_steps, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        feat = policy.obs_encoder(this_nobs)
        return feat.reshape(1, -1)

    cond_c = get_cond(obs_dict)
    obs_u = copy.deepcopy(obs_dict)
    obs_u['subgoal'] = torch.zeros_like(obs_u['subgoal'])
    cond_u = get_cond(obs_u)
    
    traj = torch.randn(size=(1, horizon, action_dim), dtype=cond_c.dtype, device=device)
    policy.noise_scheduler.set_timesteps(policy.noise_scheduler.config.num_train_timesteps)
    
    for t in policy.noise_scheduler.timesteps:
        with torch.no_grad():
            eps_c = policy.model(traj, t, global_cond=cond_c)
            eps_u = policy.model(traj, t, global_cond=cond_u)
        eps = eps_u + w * (eps_c - eps_u)
        traj = policy.noise_scheduler.step(model_output=eps, timestep=t, sample=traj).prev_sample
        
    action_np = policy.normalizer['action'].unnormalize(traj).detach().cpu().numpy()[0]
    action_np[:, -1] = np.where(action_np[:, -1] < 0.0, -1.0, 1.0) # Binarize gripper
    return action_np

# =====================================================================
# Main Evaluation Pipeline
# =====================================================================
def main():
    print(f"[INFO] Loading Low-Level Diffusion Policy: {DP_CKPT_PATH}")
    payload = torch.load(open(DP_CKPT_PATH, 'rb'), pickle_module=dill, map_location=DEVICE)
    dp_cfg = payload['cfg']
    dp_policy = hydra.utils.instantiate(dp_cfg.policy)
    dp_policy.load_state_dict(payload['state_dicts'].get('ema_model', payload['state_dicts']['model']))
    dp_policy.to(DEVICE).eval()

    print(f"[INFO] Loading High-Level VLM Pipeline: {VLM_BASE_MODEL}")
    vlm_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        VLM_BASE_MODEL, torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
    
    if os.path.exists(VLM_LORA_PATH):
        print(f"[INFO] Injecting Expert LoRA weights: {VLM_LORA_PATH}")
        vlm_pipe.unet = PeftModel.from_pretrained(vlm_pipe.unet, VLM_LORA_PATH)
    else:
        print(f"[WARNING] LoRA path not found. Model will lack domain-specific reasoning.")

    wandb.init(project="V-CoT_Final_Evaluation", name=f"VLA_Closed_Loop_R{REPLAN_STEPS}")

    for task, prompt in TASK_CONFIGS.items():
        print(f"\n[EXECUTION] Task: {task} | Prompt: '{prompt}'")
        env = suite.make(env_name=task, robots="Panda", has_offscreen_renderer=True, 
                         use_camera_obs=True, camera_names="agentview", 
                         camera_heights=256, camera_widths=256, controller_configs=load_controller_config(default_controller="OSC_POSE"))
        env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] = 45
        obs = env.reset()

        obs_history = collections.deque(maxlen=dp_cfg.n_obs_steps)
        frames, step, success = [], 0, False
        curr_subgoal_t, curr_subgoal_pil = None, None

        while step < MAX_TOTAL_STEPS:
            # 1. High-Level Reasoning (VLM Subgoal Generation)
            live_img = cv2.flip(obs['agentview_image'], 0)
            if step % VLM_FREQ == 0 or curr_subgoal_t is None:
                print(f"[INFO] Step {step}: VLM imagining future state...")
                with torch.no_grad():
                    curr_subgoal_pil = vlm_pipe(prompt=prompt, image=Image.fromarray(live_img), 
                                              num_inference_steps=VLM_INFERENCE_STEPS, 
                                              image_guidance_scale=1.5, guidance_scale=7.0).images[0]
                curr_subgoal_t = preprocess_image(np.array(curr_subgoal_pil)).to(DEVICE)

            # 2. State Preparation
            state = {'img': preprocess_image(live_img), 'pos': torch.from_numpy(obs['robot0_eef_pos']).float(),
                     'quat': torch.from_numpy(obs['robot0_eef_quat']).float(), 'grip': torch.from_numpy(obs['robot0_gripper_qpos']).float()}
            if not obs_history: [obs_history.append(state) for _ in range(dp_cfg.n_obs_steps)]
            else: obs_history.append(state)

            obs_dict = {k: torch.stack([x[v] for x in obs_history]).unsqueeze(0).to(DEVICE) 
                        for k, v in [('agentview_image','img'), ('robot0_eef_pos','pos'), 
                                     ('robot0_eef_quat','quat'), ('robot0_gripper_qpos','grip')]}
            obs_dict['subgoal'] = torch.stack([curr_subgoal_t] * dp_cfg.n_obs_steps).unsqueeze(0)

            # 3. Low-Level Control (Diffusion Inference)
            actions = predict_action(dp_policy, obs_dict, dp_cfg, w=DP_GUIDANCE_SCALE)

            # 4. Action Chunk Execution (8-step Horizon)
            for i in range(REPLAN_STEPS):
                obs, _, _, _ = env.step(actions[i])
                step += 1
                
                # Visual Feedback
                live_v = get_vis_img(obs['agentview_image'], flip=True)
                sub_v = get_vis_img(np.array(curr_subgoal_pil))
                combined = cv2.cvtColor(np.concatenate([live_v, sub_v], axis=1), cv2.COLOR_RGB2BGR)
                cv2.putText(combined, f"STEP: {step} | {task}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.putText(combined, "VLM SUBGOAL", (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

                if env._check_success() and not success:
                    print(f"[SUCCESS] Goal achieved for {task}!"); success = True
                if success and step % 40 == 0: break
            if success and step % 40 == 0: break

        wandb.log({f"{task}/Success": 1.0 if success else 0.0, 
                   f"Video_{task}": wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=20, format="mp4")})

    print("[INFO] Evaluation complete."); wandb.finish()

if __name__ == "__main__":
    main()
