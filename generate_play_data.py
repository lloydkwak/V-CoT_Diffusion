import os
import h5py
import numpy as np
import robosuite as suite
import cv2
from tqdm import tqdm
from robosuite.controllers import load_controller_config

# --- Configuration ---
TASKS = ['Lift', 'PickPlaceCan', 'NutAssemblySquare'] 
ROBOT = "Panda"
NUM_EPISODES_PER_TASK = 500
STEPS_PER_EPISODE = 100
NOISE_LEVEL = 0.15  
IMAGE_SIZE = (128, 84) 
OUTPUT_DIR = "data/play_data"

def get_object_pos(obs, task_name):
    """
    Finds the object position dynamically based on the task.
    """
    # Heuristic: Priority keys for specific tasks
    priority_keys = {
        "Lift": "cube_pos",
        "PickPlaceCan": "Can_pos",
        "NutAssemblySquare": "SquareNut_pos"
    }
    
    # 1. Try priority key first
    if task_name in priority_keys and priority_keys[task_name] in obs:
        return obs[priority_keys[task_name]]
    
    # 2. Fallback: Search for any key ending in '_pos' excluding robot parts
    for k, v in obs.items():
        if k.endswith("_pos") and "robot" not in k and "eye" not in k:
            return v
            
    # 3. Last resort: Return a zero vector if nothing found
    return np.zeros(3)

def save_episode_to_h5(group, obs_list, action_list):
    obs_group = group.create_group("obs")
    imgs = []
    for o in obs_list:
        img = cv2.resize(o['agentview_image'], IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        img = np.flipud(img) 
        imgs.append(img.transpose(2, 0, 1))
    
    obs_group.create_dataset("agentview_image", data=np.stack(imgs), compression="gzip")
    obs_group.create_dataset("robot0_eef_pos", data=np.stack([o['robot0_eef_pos'] for o in obs_list]), compression="gzip")
    obs_group.create_dataset("robot0_eef_quat", data=np.stack([o['robot0_eef_quat'] for o in obs_list]), compression="gzip")
    obs_group.create_dataset("robot0_gripper_qpos", data=np.stack([o['robot0_gripper_qpos'] for o in obs_list]), compression="gzip")
    group.create_dataset("actions", data=np.array(action_list), compression="gzip")

def collect_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    controller_config = load_controller_config(default_controller="OSC_POSE")

    for task_name in TASKS:
        h5_path = os.path.join(OUTPUT_DIR, f"play_{task_name.lower()}.hdf5")
        
        if os.path.exists(h5_path):
            print(f"[*] Task {task_name} already exists. Skipping...")
            continue

        print(f"\n[*] Starting Play Data collection for: {task_name}")
        with h5py.File(h5_path, 'w') as f:
            data_group = f.create_group("data")
            env = suite.make(
                task_name, robots=ROBOT, has_renderer=False, has_offscreen_renderer=True,
                use_camera_obs=True, camera_names=["agentview"], 
                camera_heights=256, camera_widths=256,
                controller_configs=controller_config
            )

            for ep_idx in tqdm(range(NUM_EPISODES_PER_TASK), desc=f"Collecting {task_name}"):
                obs = env.reset()
                ep_obs, ep_actions = [], []

                for _ in range(STEPS_PER_EPISODE):
                    # Dynamically find object position
                    target_pos = get_object_pos(obs, task_name)
                    direction = target_pos - obs['robot0_eef_pos']
                    dist = np.linalg.norm(direction)
                    if dist > 0: direction /= dist
                    
                    action = np.zeros(7)
                    action[:3] = direction * 0.4 + np.random.normal(0, NOISE_LEVEL, 3)
                    action[3:6] = np.random.normal(0, NOISE_LEVEL, 3)
                    action[6] = 1.0 if (dist < 0.15 and np.random.random() > 0.5) else -1.0
                    
                    ep_obs.append(obs)
                    ep_actions.append(action)
                    obs, _, _, _ = env.step(action)

                demo_group = data_group.create_group(f"demo_{ep_idx}")
                demo_group.attrs["num_samples"] = len(ep_actions)
                save_episode_to_h5(demo_group, ep_obs, ep_actions)
            env.close()

if __name__ == "__main__":
    collect_data()
