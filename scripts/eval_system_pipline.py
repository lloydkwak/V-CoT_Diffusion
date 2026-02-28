"""
Integrated Hierarchical VLA Pipeline: VLM (Subgoal) + Flow Matching (Action).

Architecture:
    1. High-Level (VLM): InstructPix2Pix + LoRA generates visual subgoals 
       every N steps to provide long-term semantic guidance.
    2. Low-Level (Flow Matching): Solves the probability flow ODE via Euler 
       integration to execute precise action chunks conditioned on the subgoal.

This version is optimized for Flow Matching policies, providing smoother 
trajectories and faster inference compared to standard diffusion.
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
# Updated to the new Flow Matching checkpoint path
DP_CKPT_PATH = "/workspace/data/outputs/flow_matching_multitask/checkpoints/latest.ckpt"
VLM_BASE_MODEL = "timbrooks/instruct-pix2pix"
VLM_LORA_PATH = "/workspace/data/vlm_outputs/checkpoint-epoch-9"

DEVICE = "cuda:0"

# Control Hyperparameters
REPLAN_STEPS = 8      # Execution horizon (Action Chunking)
VLM_FREQ = 24         # VLM updates subgoal every 24 steps (3 chunks)
MAX_TOTAL_STEPS = 600 
GUIDANCE_SCALE = 1.2  # CFG scale for Flow Matching subgoal conditioning
ODE_STEPS = 10        # Number of Euler steps for FM inference

# VLM Rendering Quality
VLM_INFERENCE_STEPS = 50
IMAGE_GUIDANCE = 1.8
TEXT_GUIDANCE = 5.0

TASK_CONFIGS = {
    "Lift": "Pick up the red block.",
    "PickPlaceCan": "Pick up the red can and place it in the bin.",
    "NutAssemblySquare": "Pick up the square nut and place it on the peg."
}

# =====================================================================
# Utility Functions
# =====================================================================
def preprocess_image(img_raw, target_shape=(3, 84, 128)):
    if img_raw.shape[0] == 3:
        img_t = torch.from_numpy(img_raw.copy()).float()
    else:
        img_t = torch.from_numpy(img_raw.copy()).float().permute(2, 0, 1)
    
    if img_t.max() > 1.0: img_t = img_t / 255.0
        
    if img_t.shape != tuple(target_shape):
        img_t = F.interpolate(img_t.unsqueeze(0), size=target_shape[1:], 
                            mode='bilinear', align_corners=False, antialias=True).squeeze(0)
    return img_t

def get_vis_img(img_raw, flip=False):
    img = img_raw.copy()
    if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0: img = img * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (256, 256))
    if flip: img = cv2.flip(img, 0)
    return img

# =====================================================================
# Flow Matching Inference (Euler ODE Solver)
# =====================================================================
def predict_action_fm(policy, obs_dict, cfg, w=1.2):
    """
    Predicts actions by integrating the learned vector field.
    Flow Matching transports x_0 (noise) to x_1 (action) via dx = v(x, t) dt.
    """
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
    
    # Start from Gaussian Noise (t=0)
    x = torch.randn(size=(1, horizon, action_dim), dtype=cond_c.dtype, device=device)
    dt = 1.0 / ODE_STEPS
    
    for i in range(ODE_STEPS):
        t_val = i * dt
        t_tensor = torch.full((1,), t_val * 1000, device=device, dtype=cond_c.dtype)
        
        with torch.no_grad():
            v_c = policy.model(x, t_tensor, global_cond=cond_c)
            v_u = policy.model(x, t_tensor, global_cond=cond_u)
            if hasattr(v_c, 'sample'): v_c, v_u = v_c.sample, v_u.sample
            
        v_final = v_u + w * (v_c - v_u)
        # Euler update step
        x = x + v_final * dt
        
    action_pred = policy.normalizer['action'].unnormalize(x)
    action_np = action_pred.detach().cpu().numpy()[0]
    action_np[:, -1] = np.where(action_np[:, -1] < 0.0, -1.0, 1.0)
    return action_np

# =====================================================================
# Main Execution Pipeline
# =====================================================================
def main():
    print(f"[*] Loading High-Level VLM Brain...")
    vlm_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        VLM_BASE_MODEL, torch_dtype=torch.float16, safety_checker=None).to(DEVICE)
    if os.path.exists(VLM_LORA_PATH):
        vlm_pipe.unet = PeftModel.from_pretrained(vlm_pipe.unet, VLM_LORA_PATH)

    print(f"[*] Loading Low-Level Flow Matching Cerebellum: {DP_CKPT_PATH}")
    payload = torch.load(open(DP_CKPT_PATH, 'rb'), pickle_module=dill, map_location=DEVICE)
    dp_cfg = payload['cfg']
    dp_policy = hydra.utils.instantiate(dp_cfg.policy)
    dp_policy.load_state_dict(payload['state_dicts'].get('ema_model', payload['state_dicts']['model']))
    dp_policy.to(DEVICE).eval()

    wandb.init(project="V-CoT_Final_System", name=f"FlowMatching_VLA_Steps{ODE_STEPS}")

    for task, prompt in TASK_CONFIGS.items():
        print(f"\n[RUNNING] Task: {task}")
        env = suite.make(env_name=task, robots="Panda", has_offscreen_renderer=True, 
                         use_camera_obs=True, camera_names="agentview", 
                         camera_heights=256, camera_widths=256, controller_configs=load_controller_config(default_controller="OSC_POSE"))
        env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] = 45
        obs = env.reset()

        obs_history = collections.deque(maxlen=dp_cfg.n_obs_steps)
        frames, step, success, curr_subgoal_t, curr_subgoal_pil = [], 0, False, None, None

        while step < MAX_TOTAL_STEPS:
            # 1. Hierarchical Reasoning (VLM Subgoal)
            live_img = cv2.flip(obs['agentview_image'], 0)
            if step % VLM_FREQ == 0 or curr_subgoal_t is None:
                print(f"[VLM] Step {step}: Generating Visual CoT Subgoal...")
                with torch.no_grad():
                    curr_subgoal_pil = vlm_pipe(prompt=prompt, image=Image.fromarray(live_img), 
                                              num_inference_steps=VLM_INFERENCE_STEPS, 
                                              image_guidance_scale=IMAGE_GUIDANCE, guidance_scale=TEXT_GUIDANCE).images[0]
                curr_subgoal_t = preprocess_image(np.array(curr_subgoal_pil)).to(DEVICE)

            # 2. Observation Queueing
            state = {'img': preprocess_image(live_img), 'pos': torch.from_numpy(obs['robot0_eef_pos']).float(),
                     'quat': torch.from_numpy(obs['robot0_eef_quat']).float(), 'grip': torch.from_numpy(obs['robot0_gripper_qpos']).float()}
            if not obs_history: [obs_history.append(state) for _ in range(dp_cfg.n_obs_steps)]
            else: obs_history.append(state)

            obs_dict = {k: torch.stack([x[v] for x in obs_history]).unsqueeze(0).to(DEVICE) 
                        for k, v in [('agentview_image','img'), ('robot0_eef_pos','pos'), 
                                     ('robot0_eef_quat','quat'), ('robot0_gripper_qpos','grip')]}
            obs_dict['subgoal'] = torch.stack([curr_subgoal_t] * dp_cfg.n_obs_steps).unsqueeze(0)

            # 3. Flow Matching Action Generation
            actions = predict_action_fm(dp_policy, obs_dict, dp_cfg, w=GUIDANCE_SCALE)

            # 4. Action Execution
            for i in range(REPLAN_STEPS):
                obs, _, _, _ = env.step(actions[i])
                step += 1
                
                # Render Loop
                vis_live = get_vis_img(obs['agentview_image'], flip=True)
                vis_sub = get_vis_img(np.array(curr_subgoal_pil))
                combined = cv2.cvtColor(np.concatenate([vis_live, vis_sub], axis=1), cv2.COLOR_RGB2BGR)
                cv2.putText(combined, f"FM-VLA | Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

                if env._check_success() and not success:
                    print(f"[*] Task Success: {task}!"); success = True
                if success and step % 40 == 0: break
            if (success and step % 40 == 0) or step >= MAX_TOTAL_STEPS: break
        
        env.close()
        wandb.log({f"{task}/Success": 1.0 if success else 0.0, 
                   f"video_{task}": wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=20, format="mp4")})

    print("[*] All tasks complete."); wandb.finish()

if __name__ == "__main__":
    main()
