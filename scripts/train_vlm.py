"""
InstructPix2Pix LoRA Fine-tuning Script for Visual CoT.

This script trains a high-level Vision-Language Model (VLM) to generate future 
subgoal images conditioned on a current observation and a textual instruction.
It utilizes Low-Rank Adaptation (LoRA) on the UNet's cross-attention layers 
for memory-efficient and rapid convergence.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import json

from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
import wandb

# =====================================================================
# Hyperparameters
# =====================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "vlm_dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "vlm_outputs")

MODEL_ID = "timbrooks/instruct-pix2pix"
RESOLUTION = 256  # Downscaled for robotics domain speed and alignment
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MAX_TRAIN_EPOCHS = 10
SEED = 42
# =====================================================================

class VisualCoTDataset(torch.utils.data.Dataset):
    """Custom dataset loader for JSONL formatted image pairs."""
    def __init__(self, data_dir, tokenizer, size=256):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.size = size
        
        self.metadata = []
        with open(os.path.join(data_dir, "metadata.jsonl"), 'r') as f:
            for line in f:
                self.metadata.append(json.loads(line))
                
        # Transforms for Stable Diffusion [-1, 1] normalization
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load images
        input_image_path = os.path.join(self.data_dir, item["input_image"])
        edited_image_path = os.path.join(self.data_dir, item["edited_image"])
        
        input_image = Image.open(input_image_path).convert("RGB")
        edited_image = Image.open(edited_image_path).convert("RGB")
        
        # Apply transforms
        input_tensor = self.image_transforms(input_image)
        edited_tensor = self.image_transforms(edited_image)
        
        # Tokenize prompt
        text_inputs = self.tokenizer(
            item["edit_prompt"], max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {
            "input_image": input_tensor,
            "edited_image": edited_tensor,
            "input_ids": text_inputs.input_ids.squeeze(0)
        }

def main():
    set_seed(SEED)
    accelerator = Accelerator(
        mixed_precision="fp16", # Use "bf16" if supported by your GPU
        log_with="wandb",
        project_dir=OUTPUT_DIR
    )
    
    accelerator.init_trackers("visual_cot_vlm_training")

    # Load Base Models
    print(f"[*] Loading Pre-trained InstructPix2Pix Models...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Freeze base models (VAE, Text Encoder)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Initialize LoRA for UNet Cross-Attention Layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Move frozen components to device to save overhead
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Setup Dataset & DataLoader
    dataset = VisualCoTDataset(DATA_DIR, tokenizer, size=RESOLUTION)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * MAX_TRAIN_EPOCHS
    )

    # Prepare for distributed training
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # Training Loop
    global_step = 0
    print(f"[*] Beginning Training: {MAX_TRAIN_EPOCHS} Epochs")
    
    for epoch in range(MAX_TRAIN_EPOCHS):
        unet.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # 1. Encode Target Subgoal Image to Latent Space
                target_latents = vae.encode(batch["edited_image"].to(accelerator.device)).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor

                # 2. Encode Input Image (Condition) to Latent Space
                input_latents = vae.encode(batch["input_image"].to(accelerator.device)).latent_dist.sample()
                input_latents = input_latents * vae.config.scaling_factor

                # 3. Sample Noise & Add to Target Latent (Forward Diffusion)
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                # 4. Construct InstructPix2Pix Input: Concat [Noisy Target Latent (4), Input Image Latent (4)] -> 8 Channels
                latent_model_input = torch.cat([noisy_latents, input_latents], dim=1)

                # 5. Extract Text Embeddings (Condition)
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                # 6. Predict Noise Residual
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample

                # 7. Compute Loss & Backpropagate
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix(loss=loss.item())

        # Save Checkpoint at end of epoch
        if accelerator.is_main_process:
            save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch}")
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(save_path)
            print(f"[*] Saved LoRA weights to {save_path}")

    accelerator.end_training()
    print("[*] Training Complete.")

if __name__ == "__main__":
    main()
