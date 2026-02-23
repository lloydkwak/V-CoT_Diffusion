import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import copy
from torch.utils.data import ConcatDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset

class RobomimicHDF5Dataset(BaseImageDataset):
    def __init__(self, dataset_path, shape_meta, horizon=16, pad_before=1, pad_after=7, **kwargs):
        super().__init__()
        self.dataset_path = os.path.abspath(dataset_path)
        self.shape_meta = shape_meta
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.file = h5py.File(self.dataset_path, 'r')
        
        self.demos = list(self.file['data'].keys())
        self.indices = [] 
        for demo in self.demos:
            num_steps = self.file[f'data/{demo}/actions'].shape[0]
            for i in range(num_steps):
                self.indices.append((demo, i))
        
        self.n_steps = len(self.indices)
        print(f"[*] Indexed {len(self.demos)} demos, {self.n_steps} steps from {dataset_path}")

    def __len__(self):
        return self.n_steps

    def _get_padded_sequence(self, dataset, step_id, total_steps):
        # Calculate valid slice bounds
        start_idx = max(0, step_id - self.pad_before)
        end_idx = min(total_steps, step_id + self.horizon - self.pad_before)
        
        # Fetch clean slice from HDF5 (no duplicates)
        val_slice = dataset[start_idx:end_idx]
        
        # Calculate padding lengths
        pad_front = max(0, -(step_id - self.pad_before))
        pad_back = max(0, (step_id + self.horizon - self.pad_before) - total_steps)
        
        # Apply padding in numpy (h5py is safe now)
        if pad_front > 0:
            front_pad = np.repeat(val_slice[0:1], pad_front, axis=0)
            val_slice = np.concatenate([front_pad, val_slice], axis=0)
        if pad_back > 0:
            back_pad = np.repeat(val_slice[-1:], pad_back, axis=0)
            val_slice = np.concatenate([val_slice, back_pad], axis=0)
            
        return val_slice

    def __getitem__(self, idx):
        demo_id, step_id = self.indices[idx]
        demo_group = self.file[f'data/{demo_id}']
        total_steps = demo_group['actions'].shape[0]
        
        obs_dict = {}
        for key, attr in self.shape_meta['obs'].items():
            if key == 'subgoal':
                continue
            
            # Fetch and pad sequence safely
            val_seq = self._get_padded_sequence(demo_group['obs'][key], step_id, total_steps)
            obs_type = attr.get('type', 'low_dim')
            
            if obs_type == 'rgb':
                if val_seq.dtype == np.uint8:
                    val_seq = val_seq.astype(np.float32) / 255.0
                elif val_seq.dtype != np.float32:
                    val_seq = val_seq.astype(np.float32)
                
                # Convert (Horizon, H, W, C) -> (Horizon, C, H, W)
                if val_seq.shape[-1] == 3:
                    val_seq = np.transpose(val_seq, (0, 3, 1, 2))
                    
                # Auto-Resize
                expected_shape = attr.get('shape', None)
                if expected_shape is not None and val_seq.shape[1:] != tuple(expected_shape):
                    val_t = torch.from_numpy(val_seq)
                    val_t = F.interpolate(val_t, size=expected_shape[1:], mode='bilinear', align_corners=False, antialias=True)
                    val_seq = val_t.numpy()
                    
            obs_dict[key] = val_seq

        # Hindsight subgoal
        offset = 5
        if step_id < total_steps - (offset + 1):
            goal_step = np.random.randint(step_id + offset, total_steps)
        else:
            goal_step = total_steps - 1
            
        subgoal = demo_group['obs']['agentview_image'][goal_step]
        
        if subgoal.dtype == np.uint8:
            subgoal = subgoal.astype(np.float32) / 255.0
        elif subgoal.dtype != np.float32:
            subgoal = subgoal.astype(np.float32)
            
        if subgoal.shape[-1] == 3:
            subgoal = np.transpose(subgoal, (2, 0, 1))
            
        expected_subgoal_shape = self.shape_meta['obs'].get('subgoal', {}).get('shape', None)
        if expected_subgoal_shape is not None and subgoal.shape != tuple(expected_subgoal_shape):
            subgoal_t = torch.from_numpy(subgoal).unsqueeze(0)
            subgoal_t = F.interpolate(subgoal_t, size=expected_subgoal_shape[1:], mode='bilinear', align_corners=False, antialias=True)
            subgoal = subgoal_t.squeeze(0).numpy()
            
        # Replicate subgoal across temporal horizon
        subgoal_seq = np.stack([subgoal] * self.horizon, axis=0)
        obs_dict['subgoal'] = subgoal_seq

        # Action sequence
        action_seq = self._get_padded_sequence(demo_group['actions'], step_id, total_steps)

        return {
            'obs': obs_dict, 
            'action': action_seq
        }

    def get_normalizer(self, mode='limits', **kwargs):
        data = {}
        sample_limit = 50
        sample_demos = self.demos[:sample_limit] if len(self.demos) > sample_limit else self.demos
        
        all_actions = []
        for demo in sample_demos:
            all_actions.append(self.file[f'data/{demo}/actions'][:])
        data['action'] = np.concatenate(all_actions, axis=0)
        
        for key, attr in self.shape_meta['obs'].items():
            if key == 'subgoal':
                continue
            if attr.get('type', 'low_dim') == 'low_dim':
                all_low_dim = []
                for demo in sample_demos:
                    all_low_dim.append(self.file[f'data/{demo}/obs/{key}'][:])
                data[key] = np.concatenate(all_low_dim, axis=0)
        
        normalizer = LinearNormalizer()
        normalizer.fit(data, last_n_dims=1, mode=mode, output_max=1.0, output_min=-1.0)
        
        for key, attr in self.shape_meta['obs'].items():
            if attr.get('type', 'low_dim') == 'rgb' or key == 'subgoal':
                normalizer.params_dict[key] = nn.ParameterDict({
                    'offset': nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False),
                    'scale': nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                })
        
        return normalizer

class VCoTMultiTaskDataset(BaseImageDataset):
    def __init__(self, dataset_paths, shape_meta, horizon=16, pad_before=1, pad_after=7, **kwargs):
        super().__init__()
        
        self.shape_meta = copy.deepcopy(shape_meta)
        if 'subgoal' in self.shape_meta['obs']:
            self.shape_meta['obs']['subgoal']['type'] = 'rgb'
            
        self.datasets = [
            RobomimicHDF5Dataset(p, self.shape_meta, horizon, pad_before, pad_after) 
            for p in dataset_paths if os.path.exists(p)
        ]
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]

    def get_normalizer(self, mode='limits', **kwargs):
        return self.datasets[0].get_normalizer(mode=mode)
