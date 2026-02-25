import os
import h5py
import torch
import numpy as np
import torchvision.transforms as T
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class VCoTMultitaskDataset(BaseImageDataset):
    """
    Multitask Dataset for Vision Chain-of-Thought (V-CoT) in Diffusion Policies.
    
    This dataset aggregates multiple offline demonstration datasets (HDF5 format) and 
    structures them for conditional trajectory generation. It features:
    1. Temporal windowing for observation history and action prediction horizons.
    2. Extraction of full proprioceptive states (Position, Quaternion, Gripper).
    3. Stochastic subgoal dropout for Classifier-Free Guidance (CFG).
    4. Asymmetric visual augmentation to prevent overfitting to low-level textures.
    """

    def __init__(self, 
                 file_paths, 
                 n_obs_steps=2, 
                 horizon=16, 
                 max_subgoal_dist=50, 
                 subgoal_drop_prob=0.2, 
                 **kwargs):
        """
        Args:
            file_paths (list): List of absolute paths to HDF5 dataset files.
            n_obs_steps (int): Number of consecutive observation frames to use as context.
            horizon (int): Length of the action trajectory to predict.
            max_subgoal_dist (int): Maximum temporal distance (in steps) to sample a future subgoal.
            subgoal_drop_prob (float): Probability of zeroing out the subgoal for CFG training.
        """
        super().__init__()
        
        self.file_paths = file_paths
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.max_subgoal_dist = max_subgoal_dist
        self.subgoal_drop_prob = subgoal_drop_prob 
        
        # Visual augmentation pipeline for current observations
        self.augmentor = T.Compose([
            T.RandomCrop((80, 120)), 
            T.Resize((84, 128)),     
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        ])

        self.indices = []
        print(f"[Dataset] Initializing temporal indexing across {len(file_paths)} sources...")

        for path in file_paths:
            full_path = os.path.abspath(path)
            if not os.path.exists(full_path):
                print(f"  [Warning] Dataset file not found: {full_path}")
                continue
            
            samples_in_file = 0
            with h5py.File(full_path, 'r') as f:
                if 'data' not in f: 
                    continue
                
                for demo_id in f['data'].keys():
                    demo_grp = f[f'data/{demo_id}']
                    
                    # Estimate trajectory length, accommodating datasets without explicit action keys
                    if 'actions' in demo_grp:
                        num_samples = len(demo_grp['actions'])
                    elif 'obs/robot0_eef_pos' in demo_grp:
                        num_samples = len(demo_grp['obs/robot0_eef_pos'])
                    else:
                        continue
                        
                    # Filter out trajectories shorter than the required temporal window
                    if num_samples <= self.horizon + self.n_obs_steps:
                        continue
                    
                    # Construct valid sliding windows
                    for i in range(num_samples - self.horizon - self.n_obs_steps):
                        self.indices.append((full_path, demo_id, i))
                        samples_in_file += 1
                        
            print(f"  [Status] Loaded {samples_in_file} samples from: {full_path}")
        
        print(f"[*] Total dataset size: {len(self.indices)} samples collected.\n")

    def __len__(self):
        """Returns the total number of valid temporal sequences across all datasets."""
        return len(self.indices)

    def get_normalizer(self, mode='limits', **kwargs):
        """
        Computes and returns a LinearNormalizer fitted to the dataset statistics.
        Randomly samples a subset of the data to efficiently compute min/max limits.
        """
        data_cache = {
            'robot0_eef_pos': [], 
            'robot0_eef_quat': [], 
            'robot0_gripper_qpos': [], 
            'action': []
        }
        
        # Stochastic sampling for efficient statistical fitting
        sample_indices = np.random.choice(len(self.indices), min(10000, len(self.indices)), replace=False)
        
        for idx in sample_indices:
            path, demo_id, start_idx = self.indices[idx]
            with h5py.File(path, 'r') as f:
                demo = f[f'data/{demo_id}']
                data_cache['robot0_eef_pos'].append(demo['obs/robot0_eef_pos'][start_idx])
                data_cache['robot0_eef_quat'].append(demo['obs/robot0_eef_quat'][start_idx])
                data_cache['robot0_gripper_qpos'].append(demo['obs/robot0_gripper_qpos'][start_idx])
                data_cache['action'].append(demo['actions'][start_idx])
                
        data_dict = {
            'robot0_eef_pos': np.array(data_cache['robot0_eef_pos']), 
            'robot0_eef_quat': np.array(data_cache['robot0_eef_quat']), 
            'robot0_gripper_qpos': np.array(data_cache['robot0_gripper_qpos']), 
            'action': np.array(data_cache['action']),
            # Dummy visual states to satisfy the normalizer's expected dictionary structure
            'agentview_image': np.array([[0.0], [1.0]], dtype=np.float32), 
            'subgoal': np.array([[0.0], [1.0]], dtype=np.float32)
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data_dict, last_n_dims=1, mode=mode)
        return normalizer

    def _process_image(self, img_array, augment=True):
        """
        Converts raw NumPy arrays to scaled PyTorch tensors (CHW format) 
        and applies optional augmentations.
        """
        if img_array.shape[0] == 3: 
            tensor = torch.from_numpy(img_array.copy()).float()
        else: 
            tensor = torch.from_numpy(img_array.copy()).float().permute(2, 0, 1)
            
        if tensor.max() > 1.0: 
            tensor /= 255.0
            
        if augment: 
            return self.augmentor(tensor)
        return T.Resize((84, 128))(tensor)

    def __getitem__(self, idx):
        """
        Retrieves a single training instance containing observation history, 
        full proprioceptive state, target action sequence, and conditional subgoal.
        """
        path, demo_id, start_idx = self.indices[idx]
        with h5py.File(path, 'r') as f:
            demo = f[f'data/{demo_id}']
            obs_dict = {}
            
            # 1. Visual Observations
            imgs = demo['obs/agentview_image'][start_idx : start_idx + self.n_obs_steps]
            obs_dict['agentview_image'] = torch.stack([self._process_image(img, augment=True) for img in imgs])
            
            # 2. Proprioceptive State (Pos, Quat, Gripper)
            obs_dict['robot0_eef_pos'] = torch.from_numpy(demo['obs/robot0_eef_pos'][start_idx : start_idx + self.n_obs_steps]).float()
            obs_dict['robot0_eef_quat'] = torch.from_numpy(demo['obs/robot0_eef_quat'][start_idx : start_idx + self.n_obs_steps]).float()
            obs_dict['robot0_gripper_qpos'] = torch.from_numpy(demo['obs/robot0_gripper_qpos'][start_idx : start_idx + self.n_obs_steps]).float()
            
            # 3. Action Trajectory
            actions = demo['actions'][start_idx : start_idx + self.horizon]
            
            # 4. Subgoal Conditioning with Classifier-Free Guidance (CFG)
            total_samples = len(demo['actions'] if 'actions' in demo else demo['obs/robot0_eef_pos'])
            subgoal_idx = np.random.randint(
                start_idx + 1, 
                min(start_idx + self.max_subgoal_dist, total_samples - 1) + 1
            )
            
            # Apply unconditional dropout
            if np.random.rand() < self.subgoal_drop_prob:
                subgoal_img = torch.zeros((3, 84, 128))
            else:
                # Retain high-fidelity target image (no spatial/color jittering)
                subgoal_img = self._process_image(demo['obs/agentview_image'][subgoal_idx], augment=False)
            
            obs_dict['subgoal'] = torch.stack([subgoal_img] * self.n_obs_steps)
            
        return {
            'obs': obs_dict, 
            'action': torch.from_numpy(actions).float()
        }

class DummyRunner(BaseImageRunner):
    """
    Headless runner utilized to bypass environment simulation rendering 
    during offline policy training.
    """
    def __init__(self, output_dir=None, **kwargs): 
        self.output_dir = output_dir
        
    def run(self, policy): 
        return {}
