content = """import os
import torch
import numpy as np
import copy
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

class RobomimicRelabelDataset(RobomimicReplayImageDataset):
    def __init__(self, dataset_path, shape_meta, **kwargs):
        # Only set cache flags if they are expected or simply force False on known ones
        kwargs['use_cache'] = False
        if 'use_disk_image_cache' in kwargs:
            kwargs['use_disk_image_cache'] = False
        
        meta = shape_meta
        if hasattr(meta, '_is_dict'):
            meta = OmegaConf.to_container(meta, resolve=True)
        else:
            meta = copy.deepcopy(meta)
            
        if 'obs' in meta and 'subgoal' in meta['obs']:
            del meta['obs']['subgoal']
            
        super().__init__(dataset_path=dataset_path, shape_meta=meta, **kwargs)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        ep_idx = self.replay_buffer.get_episode_index(idx)
        start_idx, end_idx = self.replay_buffer.get_episode_index_range(ep_idx)
        
        current_t = idx
        if current_t < end_idx - 1:
            goal_idx = np.random.randint(current_t + 1, end_idx)
        else:
            goal_idx = end_idx - 1
            
        goal_obs = self.replay_buffer.get_steps_data(goal_idx)
        subgoal_image = goal_obs['agentview_image'] 
        
        if subgoal_image.shape[-1] == 3:
            subgoal_image = np.transpose(subgoal_image, (2, 0, 1))
            
        if subgoal_image.dtype == np.uint8:
            subgoal_image = subgoal_image.astype(np.float32) / 255.0

        data['obs']['subgoal'] = subgoal_image
        return data

class VCoTMultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_paths, shape_meta, **kwargs):
        self.datasets = []
        for path in dataset_paths:
            if os.path.exists(path):
                print(f"[*] Loading dataset: {path}")
                # Filter out any arguments that might cause TypeError in the child class
                self.datasets.append(RobomimicRelabelDataset(dataset_path=path, shape_meta=shape_meta, **kwargs))
        
        if len(self.datasets) == 0:
            raise ValueError("No valid datasets loaded.")
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self): return len(self.concat_dataset)
    def __getitem__(self, idx): return self.concat_dataset[idx]
        
    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        all_actions = []
        all_obs_dict = {}
        for ds in self.datasets:
            ds_normalizer = ds.get_normalizer(mode=mode, **kwargs)
            if 'action' in ds_normalizer.params_dict:
                all_actions.append(ds.replay_buffer['action'][:])
            for key, attr in ds.shape_meta['obs'].items():
                if attr.get('type', 'low_dim') == 'low_dim' and key != 'subgoal':
                    buffer_key = f'obs/{key}'
                    if buffer_key in ds.replay_buffer:
                        val = ds.replay_buffer[buffer_key][:]
                        if key not in all_obs_dict: all_obs_dict[key] = []
                        all_obs_dict[key].append(val)
                    
        if len(all_actions) > 0:
            normalizer.fit(data=np.concatenate(all_actions, axis=0), last_n_dims=1, mode=mode, output_max=1.0, output_min=-1.0)
        for obs_key, obs_list in all_obs_dict.items():
            normalizer.fit(data=np.concatenate(obs_list, axis=0), last_n_dims=1, mode=mode, output_max=1.0, output_min=-1.0)
        return normalizer
"""
with open('diffusion_policy/dataset/vcot_multitask_dataset.py', 'w') as f:
    f.write(content)
print("✅ 안정화 패치 완료!")
