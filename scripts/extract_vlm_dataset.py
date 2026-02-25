"""
Visual CoT Dataset Extraction Script for VLM Fine-Tuning.

[MODIFIED] 
Strictly uses 'ph' (Proficient Human) datasets. 
Play data is excluded to ensure the VLM learns only the optimal, noise-free 
pathways to task completion without exploratory hesitations.
"""

import os
import h5py
import cv2
import json
import numpy as np
import random

# ================= Configuration =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "vlm_dataset_ph_expert")
IMAGE_OUT_DIR = os.path.join(OUTPUT_DIR, "images")

# [CRITICAL FIX] Map to the exact 'ph' (Proficient Human) expert datasets
DATASET_PROMPTS = {
    "data/lift/ph/image.hdf5": "Pick up the red block.",
    "data/can/ph/image.hdf5": "Pick up the red can and place it in the bin.",
    "data/square/ph/image.hdf5": "Pick up the square nut and place it on the peg."
}

STRIDE = 10           
MIN_LOOKAHEAD = 20    
MAX_LOOKAHEAD = 40    
# =================================================

def main():
    os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
    metadata = []
    pair_count = 0

    print(f"[*] Starting EXPERT Visual CoT Dataset Extraction (ph datasets only)...")
    print(f"[*] Target Output Directory: {OUTPUT_DIR}\n")

    for rel_path, prompt in DATASET_PROMPTS.items():
        h5_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(h5_path):
            print(f"[Warning] Skipped: {rel_path} (File not found)")
            continue
        
        task_name = rel_path.split('/')[1]  # 'lift', 'can', or 'square'
        print(f"[*] Processing Expert Data for {task_name.upper()} task...")
        
        with h5py.File(h5_path, 'r') as f:
            demos = list(f['data'].keys())
            
            for demo_id in demos:
                images = f[f'data/{demo_id}/obs/agentview_image'][:]
                seq_len = len(images)
                
                if seq_len <= MIN_LOOKAHEAD:
                    continue
                    
                for t in range(0, seq_len - MIN_LOOKAHEAD, STRIDE):
                    k = random.randint(MIN_LOOKAHEAD, MAX_LOOKAHEAD)
                    target_t = min(t + k, seq_len - 1)
                    
                    img_t = images[t]
                    img_target = images[target_t]
                    
                    if img_t.shape[0] == 3:
                        img_t = np.transpose(img_t, (1, 2, 0))
                        img_target = np.transpose(img_target, (1, 2, 0))
                        
                    if img_t.max() <= 1.0:
                        img_t = (img_t * 255.0).astype(np.uint8)
                        img_target = (img_target * 255.0).astype(np.uint8)
                    else:
                        img_t = img_t.astype(np.uint8)
                        img_target = img_target.astype(np.uint8)
                        
                    img_t_bgr = cv2.cvtColor(img_t, cv2.COLOR_RGB2BGR)
                    img_target_bgr = cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)
                    
                    in_filename = f"{task_name}_expert_{demo_id}_t{t:04d}.jpg"
                    out_filename = f"{task_name}_expert_{demo_id}_t{t:04d}_tgt{target_t:04d}.jpg"
                    
                    in_path = os.path.join(IMAGE_OUT_DIR, in_filename)
                    out_path = os.path.join(IMAGE_OUT_DIR, out_filename)
                    
                    cv2.imwrite(in_path, img_t_bgr)
                    cv2.imwrite(out_path, img_target_bgr)
                    
                    metadata.append({
                        "input_image": f"images/{in_filename}",
                        "edit_prompt": prompt,
                        "edited_image": f"images/{out_filename}"
                    })
                    pair_count += 1

    jsonl_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    with open(jsonl_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
            
    print(f"\n[Success] Expert Dataset extraction completed!")
    print(f"  - Total Expert Image Pairs: {pair_count}")
    print(f"  - Metadata saved to: {jsonl_path}")

if __name__ == "__main__":
    main()
