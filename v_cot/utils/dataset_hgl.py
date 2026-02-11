import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class HGLDataset(Dataset):
    def __init__(self, file_path, img_size=128):
        """
        Custom Dataset for V-CoT Diffusion.
        Args:
            file_path: Path to the rendered image_sample.hdf5
            img_size: Image resolution for resizing
        """
        self.file_path = file_path
        self.img_size = img_size
        self.data = h5py.File(self.file_path, 'r')
        self.demos = list(self.data['data'].keys())
        
        # Preprocessing: Convert HDF5 (H,W,C) to Torch (C,H,W) and Normalize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Indexing all frames across all demos
        self.indices = []
        for demo in self.demos:
            num_frames = self.data['data'][demo]['actions'].shape[0]
            for i in range(num_frames):
                self.indices.append((demo, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        demo_id, frame_idx = self.indices[idx]
        demo_grp = self.data['data'][demo_id]

        # 1. Current Observation (First-person view for policy)
        # Using robot0_eye_in_hand_image as the main control input
        curr_img = demo_grp['obs']['robot0_eye_in_hand_image'][frame_idx]
        
        # 2. Subgoal Observation (Third-person view for V-CoT)
        # In this simplified version, we use the last frame's agentview as the goal
        last_idx = demo_grp['actions'].shape[0] - 1
        goal_img = demo_grp['obs']['agentview_image'][last_idx]

        # 3. Action (Target for Diffusion)
        action = demo_grp['actions'][frame_idx]

        # Apply transforms
        curr_img_tensor = self.transform(curr_img)
        goal_img_tensor = self.transform(goal_img)

        return {
            "pixel_values": curr_img_tensor,     # I_curr (Input to Policy)
            "subgoal_values": goal_img_tensor,   # I_subgoal (Conditioning)
            "actions": torch.from_numpy(action).float(),
            "input_ids": "Lift the object to the goal position." # Dummy instruction
        }

if __name__ == "__main__":
    import os
    # Update path to your newly generated file
    path = "data/robomimic/lift/ph/image_sample.hdf5"
    
    if os.path.exists(path):
        dataset = HGLDataset(path)
        sample = dataset[0]
        print(f"Dataset Loaded. Total samples: {len(dataset)}")
        print(f"Current Image: {sample['pixel_values'].shape}")
        print(f"Subgoal Image: {sample['subgoal_values'].shape}")
        print(f"Action Shape: {sample['actions'].shape}")
        print(f"Instruction: {sample['input_ids']}")
    else:
        print(f"File not found at {path}. Check your Docker mount.")
