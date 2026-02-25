"""
Evaluation Script for Low-Level Visuomotor Policy in V-CoT Architecture.

This script evaluates the performance of the Subgoal-Conditioned Diffusion Policy
independently from the High-Level Vision-Language Model (VLM). It employs a 
'Teacher Forcing' evaluation paradigm where ground-truth future observations 
from expert demonstrations are continuously provided as subgoals.

Key Metrics:
    1. Task Success Rate: Binary indicator of task completion.
    2. Average Trajectory Tracking Error: The L2 distance between the model's 
       end-effector trajectory and the expert's ground-truth trajectory.
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

# Add project root to sys.path
project_root = "/workspace/diffusion_policy"
if project_root not in sys.path:
    sys.path.append(project_root)

import robosuite as suite
from robosuite.controllers import load_controller_config

# =====================================================================
# Hyperparameters & Configuration
# =====================================================================
CKPT_PATH = "/workspace/data/outputs/2026-02-24/10-48-30/checkpoints/latest.ckpt"
DEVICE = "cuda:0"

# Evaluation hyperparameters
LOOKAHEAD_STEPS = 30  # Horizon (t + N) for the visual subgoal via Teacher Forcing
REPLAN_STEPS = 4      # Execution horizon before replanning (Action Chunking)
MAX_TOTAL_STEPS = 600 # Maximum allowable steps per episode
GUIDANCE_SCALE = 1.2  # Classifier-Free Guidance (CFG) scale for subgoal conditioning

TASK_CONFIGS = {
    "Lift": "/workspace/data/lift/ph/image.hdf5",
    "PickPlaceCan": "/workspace/data/play_data/play_pickplacecan.hdf5",
    "NutAssemblySquare": "/workspace/data/play_data/play_nutassemblysquare.hdf5"
}

# =====================================================================
# Utility Functions
# =====================================================================
def preprocess_image(img_raw, target_shape=(3, 84, 128)):
    """Preprocesses raw image arrays into normalized PyTorch tensors."""
    if img_raw.shape[0] == 3: 
        img_t = torch.from_numpy(img_raw.copy()).float()
    else: 
        img_t = torch.from_numpy(img_raw.copy()).float().permute(2, 0, 1)
    
    if img_t.max() > 1.0: 
        img_t = img_t / 255.0
        
    if img_t.shape != tuple(target_shape):
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=target_shape[1:], 
            mode='bilinear', align_corners=False, antialias=True
        ).squeeze(0)
    return img_t

def get_vis_img(img_raw, flip=False):
    """Formats raw images for visual rendering and logging."""
    img = img_raw.copy()
    if img.shape[0] == 3: 
        img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0: 
        img = img * 255.0
    
    img = img.astype(np.uint8)
    img = cv2.resize(img, (256, 256))
    if flip: 
        img = cv2.flip(img, 0)
    return img

def predict_action_cfg(policy, obs_dict, cfg, w=1.2):
    """
    Predicts the optimal action sequence using Classifier-Free Guidance (CFG).
    
    Args:
        policy: The conditioned diffusion model.
        obs_dict: Dictionary containing current observations and the target subgoal.
        cfg: Model configuration.
        w: Guidance scale.
        
    Returns:
        np.ndarray: Predicted action chunk.
    """
    device = next(policy.parameters()).device
    
    try:
        action_dim = cfg.policy.shape_meta.action.shape[0]
    except AttributeError:
        action_dim = cfg.task.dataset_shape_meta.action.shape[0]
        
    n_obs_steps = cfg.n_obs_steps
    horizon = cfg.horizon
    
    def get_global_cond(obs):
        # Filter observation keys to ensure backward compatibility with older checkpoints
        valid_keys = policy.normalizer.params_dict.keys()
        filtered_obs = {k: v for k, v in obs.items() if k in valid_keys or 'image' in k or 'subgoal' in k}
        
        nobs = policy.normalizer.normalize(filtered_obs)
        # Reshape to merge batch and temporal dimensions for the vision encoder (Batch * Time, C, H, W)
        this_nobs = {k: v[:, :n_obs_steps, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        feat = policy.obs_encoder(this_nobs)
        
        if len(feat.shape) > 2:
            feat = feat.reshape(feat.shape[0], -1)
        # Reshape back to (Batch, Feature_Dim) to align with Diffusion UNet global conditioning
        return feat.reshape(1, -1)

    # Compute conditional and unconditional feature representations
    global_cond_cond = get_global_cond(obs_dict)
    
    obs_dict_uncond = copy.deepcopy(obs_dict)
    obs_dict_uncond['subgoal'] = torch.zeros_like(obs_dict_uncond['subgoal'])
    global_cond_uncond = get_global_cond(obs_dict_uncond)
    
    # Initialize Gaussian noise trajectory
    trajectory = torch.randn(size=(1, horizon, action_dim), dtype=global_cond_cond.dtype, device=device)
    
    num_steps = policy.noise_scheduler.config.num_train_timesteps
    policy.noise_scheduler.set_timesteps(num_steps)
    
    # Reverse diffusion process (Denoising loop) with CFG
    for t in policy.noise_scheduler.timesteps:
        with torch.no_grad():
            noise_cond = policy.model(trajectory, t, global_cond=global_cond_cond)
            noise_uncond = policy.model(trajectory, t, global_cond=global_cond_uncond)
            
        noise_pred = noise_uncond + w * (noise_cond - noise_uncond)
        trajectory = policy.noise_scheduler.step(model_output=noise_pred, timestep=t, sample=trajectory).prev_sample
        
    action_pred = policy.normalizer['action'].unnormalize(trajectory)
    action_np = action_pred.detach().cpu().numpy()[0]
    
    # Post-processing Heuristic: Gripper Action Binarization
    # Diffusion models tend to predict smoothed, continuous values for discrete actions.
    # We binarize the gripper action (typically the last dimension) to prevent hovering and ensure firm grasps.
    gripper_action = action_np[:, -1]
    action_np[:, -1] = np.where(gripper_action < 0.0, -1.0, 1.0)
    
    return action_np

# =====================================================================
# Main Evaluation Loop
# =====================================================================
def run_eval():
    print(f"[*] Initializing evaluation pipeline. Loading checkpoint: {CKPT_PATH}")
    payload = torch.load(open(CKPT_PATH, 'rb'), pickle_module=dill, map_location=DEVICE)
    cfg = payload['cfg']
    
    policy = hydra.utils.instantiate(cfg.policy)
    policy_weights = payload['state_dicts']['ema_model'] if 'ema_model' in payload['state_dicts'] else payload['state_dicts']['model']
    policy.load_state_dict(policy_weights)
    policy.to(DEVICE).eval()

    wandb.init(project="diffusion_policy_multitask", name=f"eval_teacher_forcing_w{GUIDANCE_SCALE}")

    for task_name, dataset_path in TASK_CONFIGS.items():
        if not os.path.exists(dataset_path): 
            continue

        print(f"\n==========================================")
        print(f"[*] Task: {task_name} | Subgoal-Conditioned Teacher Forcing Evaluation")
        print(f"==========================================")

        # Load expert demonstration dataset
        with h5py.File(dataset_path, 'r') as f:
            demo_keys = list(f['data'].keys())
            
            # Identify the longest demonstration to serve as the standard reference trajectory
            best_demo_id = demo_keys[0]
            max_len = 0
            for d_id in demo_keys:
                curr_len = len(f[f'data/{d_id}/obs/agentview_image'])
                if curr_len > max_len:
                    max_len = curr_len
                    best_demo_id = d_id
            
            print(f"[*] Expert Reference Selected: {best_demo_id} (Duration: {max_len} steps)")
            
            demo_imgs = f[f'data/{best_demo_id}/obs/agentview_image'][:]
            demo_eefs = f[f'data/{best_demo_id}/obs/robot0_eef_pos'][:]
            total_demo_steps = len(demo_imgs)
            
            has_states = 'states' in f[f'data/{best_demo_id}']
            if has_states: 
                demo_states = f[f'data/{best_demo_id}/states'][:]

        # Initialize Robosuite environment
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make(
            env_name=task_name, robots="Panda", has_renderer=False,
            has_offscreen_renderer=True, use_camera_obs=True,
            camera_names="agentview", camera_heights=256, camera_widths=256,
            reward_shaping=False,
            controller_configs=controller_config,
        )
        # Adjust camera FOV for consistent observation rendering
        env.sim.model.cam_fovy[env.sim.model.camera_name2id("agentview")] = 45

        env.reset()
        if has_states:
            env.sim.set_state_from_flattened(demo_states[0])
            env.sim.forward()
        obs = env._get_observations()

        obs_deque = collections.deque(maxlen=cfg.n_obs_steps)
        frames = []
        trajectory_errors = []  # Log for Trajectory Tracking Error metric

        step = 0
        task_success = False
        finishing_steps = 40 

        while step < MAX_TOTAL_STEPS:
            # Implement Teacher Forcing paradigm: 
            # Subgoal is rigorously defined as the expert's image observation at t + LOOKAHEAD_STEPS.
            target_idx = min(step + LOOKAHEAD_STEPS, total_demo_steps - 1)
            raw_subgoal = demo_imgs[target_idx]
            
            sub_t = preprocess_image(raw_subgoal).to(DEVICE)
            subgoal_input = torch.stack([sub_t] * cfg.n_obs_steps).unsqueeze(0)

            live_img_upright = cv2.flip(obs['agentview_image'], 0)
            curr_img_t = preprocess_image(live_img_upright)
            
            curr_pos_t = torch.from_numpy(obs['robot0_eef_pos']).float()
            curr_quat_t = torch.from_numpy(obs['robot0_eef_quat']).float()
            curr_grip_t = torch.from_numpy(obs['robot0_gripper_qpos']).float()

            state_dict = {'img': curr_img_t, 'pos': curr_pos_t, 'quat': curr_quat_t, 'grip': curr_grip_t}

            # Populate observation history queue
            if len(obs_deque) == 0:
                for _ in range(cfg.n_obs_steps): 
                    obs_deque.append(state_dict)
            else:
                obs_deque.append(state_dict)

            obs_dict = {
                'agentview_image': torch.stack([x['img'] for x in obs_deque]).unsqueeze(0).to(DEVICE),
                'robot0_eef_pos': torch.stack([x['pos'] for x in obs_deque]).unsqueeze(0).to(DEVICE),
                'robot0_eef_quat': torch.stack([x['quat'] for x in obs_deque]).unsqueeze(0).to(DEVICE),
                'robot0_gripper_qpos': torch.stack([x['grip'] for x in obs_deque]).unsqueeze(0).to(DEVICE),
                'subgoal': subgoal_input
            }

            # Generate action via policy inference
            action_seq = predict_action_cfg(policy, obs_dict, cfg, w=GUIDANCE_SCALE)

            # Execute Receding Horizon Control (Action Chunking)
            for i in range(REPLAN_STEPS):
                obs, reward, done, _ = env.step(action_seq[i])
                
                # Evaluation Metric: Compute L2 Distance between true EEF and expert EEF at current timestep
                expert_idx = min(step, total_demo_steps - 1)
                expert_eef = demo_eefs[expert_idx]
                traj_error = np.linalg.norm(obs['robot0_eef_pos'] - expert_eef)
                trajectory_errors.append(traj_error)

                step += 1

                # Render evaluation visualization
                live_view = get_vis_img(obs['agentview_image'], flip=True)
                target_view = get_vis_img(raw_subgoal, flip=False)
                combined = np.concatenate([live_view, target_view], axis=1)
                
                status_text = "SUCCESS!" if task_success else f"Traj Err: {traj_error:.3f}m"
                
                combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                cv2.putText(combined, f"LIVE | Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(combined, f"TARGET {target_idx} | {status_text}", (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

                # Verify task completion utilizing underlying environment semantics
                if env._check_success() and not task_success:
                    print(f"[*] Task {task_name} executed successfully. Initiating finishing sequence...")
                    task_success = True

                # Allow policy to stabilize post-success before terminating episode
                if task_success:
                    finishing_steps -= 1
                    if finishing_steps <= 0:
                        done = True
                        break

            if done: 
                break
                
            # Timeout constraint: Terminate if model fails to complete task well beyond expert demonstration length
            if step > total_demo_steps + 50 and not task_success:
                print(f"[!] Evaluation Terminated: Failed to achieve goal within reasonable temporal margin.")
                break

        # Aggregate and log evaluation metrics
        avg_traj_error = np.mean(trajectory_errors)
        print(f"\n[Evaluation Summary: {task_name}]")
        print(f"  - Task Success Rate: {'1.0 (Yes)' if task_success else '0.0 (No)'}")
        print(f"  - Avg Trajectory Tracking Error: {avg_traj_error:.4f} m\n")
        
        wandb.log({
            f"{task_name}/Task_Success": 1.0 if task_success else 0.0,
            f"{task_name}/Avg_Trajectory_Error": avg_traj_error,
            f"video_{task_name}": wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=20, format="mp4")
        })

    print(f"[*] Evaluation completed. Syncing remaining logs to WandB...")
    wandb.finish()

if __name__ == "__main__":
    run_eval()
