from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
import einops
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

class VCoTVisionEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str, nn.Module]],
            resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
            crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
            random_crop: bool = True,
            use_group_norm: bool = False,
            share_rgb_model: bool = False,
            imagenet_norm: bool = False,
            spatial_keys: list = ['agentview_subgoal'] # Keys to preserve spatial maps
        ):
        """
        Vision Encoder for V-CoT Diffusion.
        Extracts global vectors for FiLM and spatial maps for Cross-Attention.
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        if share_rgb_model:
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape

            if obs_type == 'rgb':
                rgb_keys.append(key)
                
                # Clone or assign backbone model
                this_model = rgb_model[key] if isinstance(rgb_model, dict) else copy.deepcopy(rgb_model)
                
                # Replace BN with GN for stable training in small batches
                if use_group_norm:
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(num_groups=x.num_features//16, num_channels=x.num_features)
                    )
                key_model_map[key] = this_model
                
                # Setup Image Transforms (Resize -> Crop -> Norm)
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    h, w = resize_shape[key] if isinstance(resize_shape, dict) else resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    h, w = crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(input_shape=input_shape, crop_height=h, crop_width=w, num_crops=1, pos_enc=False)
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))

                this_normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if imagenet_norm else nn.Identity()
                key_transform_map[key] = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                
            elif obs_type == 'low_dim':
                low_dim_keys.append(key)

        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.rgb_keys = sorted(rgb_keys)
        self.low_dim_keys = sorted(low_dim_keys)
        self.spatial_keys = spatial_keys

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        batch_size = None
        global_features = list()
        spatial_features = dict()

        # 1. Process RGB Observations
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None: batch_size = img.shape[0]
            
            # Apply augmentation and normalization
            img = self.key_transform_map[key](img)
            model = self.key_model_map[key]

            # Case A: Extract Global Feature Vector (Flattened)
            global_feat = model(img) 
            global_features.append(global_feat)

            # Case B: Extract Spatial Feature Map for Cross-Attention
            if key in self.spatial_keys:
                # Remove last two layers (GlobalAvgPool and Flatten) to get the 2D map
                # Output shape: [B, C, H, W] -> e.g., [B, 512, 4, 4]
                spatial_map = model[:-2](img)
                
                # Reshape for Transformer-style Attention: [B, C, H, W] -> [B, H*W, C]
                spatial_map = einops.rearrange(spatial_map, 'b c h w -> b (h w) c')
                spatial_features[key] = spatial_map

        # 2. Process Low-dimensional Observations (Joint states, etc.)
        for key in self.low_dim_keys:
            global_features.append(obs_dict[key])
        
        # 3. Return Dictionary for Policy Integration
        return {
            'global_feat': torch.cat(global_features, dim=-1),
            'spatial_feats': spatial_features
        }
