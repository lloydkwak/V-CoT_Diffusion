"""
Independent Evaluation Script for Low-Level Flow Matching Policy.

This script benchmarks the performance of the Subgoal-Conditioned Flow Matching 
policy. Unlike traditional diffusion-based evaluation, this script utilizes 
Euler integration (ODE solving) to transport noise to the action distribution.

Evaluation Paradigm: 
    - Teacher Forcing: Ground-truth frames from expert demonstrations are 
      provided as subgoals to isolate the controller's performance.

Metrics:
    - Task Success Rate: Binary success based on environment semantics.
    - Trajectory Tracking Error: L2 distance to expert EEF trajectories.
"""

import os
import sys
import torch
import copy
import torch.nn.functional as F
import numpy as np
import h5py
import cv2
import collections
import hydra
import wandb
import dill

# Add project root to sys.path for diffusion_policy modules
project_root = "/workspace/diffusion_policy"
if project_root not in sys.path:
    sys.path.append(project_root)

import robosuite as suite
from robosuite.controllers import load_controller_config

# =====================================================================
# Global Configuration & Model Paths
# =====================================================================
# Updated to point to the new Flow Matching multitask checkpoint
CKPT_PATH = "/workspace/data/outputs/flow_matching_multitask/checkpoints/latest.ckpt"
DEVICE = "cuda:0"

# Inference Hyperparameters
LOOKAHEAD_STEPS = 30  # Subgoal horizon (t + N) for Teacher Forcing
REPLAN_STEPS = 8      # Increased to 8 steps for smoother Flow Matching control
MAX_TOTAL_STEPS = 600 # Max episode length
GUIDANCE_SCALE = 1.2  # Classifier-Free Guidance (CFG) scale
ODE_STEPS = 10        # Number of Euler integration steps (Flow Matching standard)

TASK_CONFIGS = {
    "Lift": "/workspace/data/lift/ph/image.hdf5",
    "PickPlaceCan": "/workspace/data/play_data/play_pickplacecan.hdf5",
    "NutAssemblySquare": "/workspace/data/play_data/play_nutassemblysquare.hdf5"
}

# =====================================================================
# Neural Architecture Utilities
# =====================================================================
def preprocess_image(img_raw, target_shape=(3, 84, 128)):
    """Normalizes and reshapes raw images into PyTorch tensors."""
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
    """Formats raw images for visual logging (OpenCV/WandB)."""
    img = img_raw.copy()
    if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0: img = img * 255.0
    img = img.astype(np.uint8)
    img = cv2.resize(img, (256, 256))
    if flip: img = cv2.flip(img, 0)
    return img

# =====================================================================
# Flow Matching Inference Engine (Euler ODE Solver)
# =====================================================================
def predict_action_cfg(policy, obs_dict, cfg, w=1.2):
    """
    Predicts action trajectory using Optimal Transport Flow Matching.
    Solves dx/dt = v_t(x, cond) via Euler integration with CFG.
    """
    device = next(policy.parameters()).device
    try:
        action_dim = cfg.policy.shape_meta.action.shape[0]
    except AttributeError:
        action_dim = cfg.task.dataset_shape_meta.action.shape[0]
        
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon
    
    def get_global_cond(obs):
        valid_keys = policy.normalizer.params_dict.keys()
        filtered = {k: v for k, v in obs.items() if k in valid_keys or 'image' in k or 'subgoal' in k}
        nobs = policy.normalizer.normalize(filtered)
        this_nobs = {k: v[:, :n_obs_steps, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        feat = policy.obs_encoder(this_nobs)
        return feat.reshape(1, -1)

    # 1. Compute Conditional and Unconditional embeddings
    cond_c = get_global_cond(obs_dict)
    obs_u = copy.deepcopy(obs_dict)
    obs_u['subgoal'] = torch.zeros_like(obs_u['subgoal'])
    cond_u = get_global_cond(obs_u)
    
    # 2. Initialize Trajectory from Prior Noise (x_0 ~ N(0, I))
    x = torch.randn(size=(1, horizon, action_dim), dtype=cond_c.dtype, device=device)
    
    # 3. Solve Probability Flow ODE via Euler Method
    dt = 1.0 / ODE_STEPS
    for i in range(ODE_STEPS):
        t_val = i * dt
        # Scale t by 1000 to match UNet's sinusoidal embedding frequency
        t_tensor = torch.full((1,), t_val * 1000, device=device, dtype=cond_c.dtype)
        
        with torch.no_grad():
            # Predict velocity fields (v) for both conditions
            v_c = policy.model(x, t_tensor, global_cond=cond_c)
            v_u = policy.model(x, t_tensor, global_cond=cond_u)
            
            if hasattr(v_c, 'sample'): v_c, v_u = v_c.sample, v_u.sample
            
        # Apply Classifier-Free Guidance on the velocity field
        v_final = v_u + w * (v_c - v_u)
        
        # Euler Step: x_{t+dt} = x_t + v * dt
        x = x + v_final * dt
        
    # 4. Denormalize and Post-process
    action_pred = policy.normalizer['action'].unnormalize(x)
    action_np = action_pred.detach().cpu().numpy()[0]
    
    # Binarize Gripper Action
    action_np[:, -1] = np.where(action_np[:, -1] < 0.0, -1.0, 1.0)
    
    return action_np

# =====================================================================
# Main Execution Logic
# =====================================================================
def run_eval():
    print(f"[*] Loading Flow Matching Policy: {CKPT_PATH}")
    payload = torch.load(open(CKPT_PATH, 'rb'), pickle_module=dill, map_location=DEVICE)
    cfg = payload['cfg']
    
    policy = hydra.utils.instantiate(cfg.policy)
    weights = payload['state_dicts'].get('ema_model', payload['state_dicts']['model'])
    policy.load_state_dict(weights)
    policy.to(DEVICE).eval()

    wandb.init(project="V-CoT_FlowMatching_Eval", name=f"FM_TeacherForcing_Steps{ODE_STEPS}")

    for task_name, dataset_path in TASK_CONFIGS.items():
        if not os.path.exists(dataset_path): continue

        print(f"\n[EVALUATING] Task: {task_name}")
        with h5py.File(dataset_path, 'r') as f:
            demo_id = list(f['data'].keys())[0] # Evaluate on first demo for consistency
            demo_imgs = f[f'data/{demo_id}/obs/agentview_image'][:]
            demo_eefs = f[f'data/{demo_id}/obs/robot0_eef_pos'][:]
            demo_states = f[f'data/{demo_id}/states'][:] if 'states' in f[f'data/{demo_id}'] else None

        env = suite.make(env_name=task_name, robots="Panda", has_offscreen_renderer=True, 
                         use_camera_obs=True, camera_names="agentview", 
                         camera_heights=256, camera_widths=256, controller_configs=load_controller_config(default_controller="OSC_POSE"))
        env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] = 45
        env.reset()
        if demo_states is not None:
            env.sim.set_state_from_flattened(demo_states[0])
            env.sim.forward()
        
        obs = env._get_observations()
        obs_deque = collections.deque(maxlen=cfg.n_obs_steps)
        frames, trajectory_errors, step, success = [], [], 0, False

        while step < MAX_TOTAL_STEPS:
            # 1. Teacher Forcing Subgoal Selection
            target_idx = min(step + LOOKAHEAD_STEPS, len(demo_imgs) - 1)
            sub_t = preprocess_image(demo_imgs[target_idx]).to(DEVICE)
            
            # 2. Observation Formatting
            live_img = cv2.flip(obs['agentview_image'], 0)
            state = {'img': preprocess_image(live_img), 'pos': torch.from_numpy(obs['robot0_eef_pos']).float(),
                     'quat': torch.from_numpy(obs['robot0_eef_quat']).float(), 'grip': torch.from_numpy(obs['robot0_gripper_qpos']).float()}
            if not obs_deque: [obs_deque.append(state) for _ in range(cfg.n_obs_steps)]
            else: obs_deque.append(state)

            obs_dict = {k: torch.stack([x[v] for x in obs_deque]).unsqueeze(0).to(DEVICE) 
                        for k, v in [('agentview_image','img'), ('robot0_eef_pos','pos'), 
                                     ('robot0_eef_quat','quat'), ('robot0_gripper_qpos','grip')]}
            obs_dict['subgoal'] = torch.stack([sub_t] * cfg.n_obs_steps).unsqueeze(0)

            # 3. Vector Field Prediction (Flow Matching)
            actions = predict_action_cfg(policy, obs_dict, cfg, w=GUIDANCE_SCALE)

            # 4. Action Execution
            for i in range(REPLAN_STEPS):
                obs, _, _, _ = env.step(actions[i])
                traj_err = np.linalg.norm(obs['robot0_eef_pos'] - demo_eefs[min(step, len(demo_eefs)-1)])
                trajectory_errors.append(traj_err)
                step += 1

                # Visualization
                vis_live = get_vis_img(obs['agentview_image'], flip=True)
                vis_sub = get_vis_img(demo_imgs[target_idx])
                combined = cv2.cvtColor(np.concatenate([vis_live, vis_sub], axis=1), cv2.COLOR_RGB2BGR)
                cv2.putText(combined, f"FM Step: {step} | Err: {traj_err:.3f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

                if env._check_success() and not success:
                    print(f"[*] {task_name} Success!"); success = True
                if success and step % 40 == 0: break
            if (success and step % 40 == 0) or step > len(demo_imgs) + 50: break
        
        env.close()
        wandb.log({f"{task_name}/Success": 1.0 if success else 0.0, f"{task_name}/Avg_Err": np.mean(trajectory_errors),
                   f"video_{task_name}": wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=20, format="mp4")})

    wandb.finish()

if __name__ == "__main__":
    run_eval()
