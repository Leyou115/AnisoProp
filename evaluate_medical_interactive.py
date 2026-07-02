#!/usr/bin/env python
"""
Volume-level Interactive Evaluation for AMOS 2022
(Prompt-based / Class-agnostic)

Evaluates the model's ability to segment any organ given a reference mask (prompt)
on a single slice.

Methodology:
1. For each test volume:
2. Identify all organs present in the Ground Truth.
3. For each organ:
    a. Select the "best" slice (e.g., max area) as the Prompt/Reference.
    b. Propagate segmentation to the entire 3D volume (chunk-by-chunk).
       - Global Reference: The prompt slice is used as reference for all chunks.
       - Local Consistency: PhysioMamba ensures smoothness within chunks.
    c. Compute 3D Dice & IoU for this organ.
4. Report Mean Dice across all organs and all volumes.

Usage:
    python evaluate_medical_interactive.py --config configs/interactive_amos.yaml --checkpoint outputs/best_model.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import csv

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from model.interactive import DINOv3PhysioMambaInteractive

def compute_dice(pred, target):
    """Compute 3D Dice Score"""
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union

def compute_iou(pred, target):
    """Compute 3D IoU"""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

class VolumeInference:
    def __init__(self, model, device, chunk_size=8, crop_size=(512, 512), threshold=0.5, mode='global', max_memory=5):
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.crop_size = crop_size
        self.threshold = threshold
        self.mode = mode
        self.max_memory = max_memory
        
    @torch.no_grad()
    def predict_volume(self, volume, reference_slice, reference_mask, z_spacing, slice_axis=2):
        """
        Run interactive segmentation on a full volume.
        """
        self.model.eval()
        
        # Ensure volume is [D, H, W] for easier chunking
        if slice_axis == 2:
            volume = volume.transpose(2, 0, 1) # [D, H, W]
        
        D, H, W = volume.shape
        
        # Initial Reference (Global)
        ref_img_t = torch.from_numpy(reference_slice).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        ref_mask_t = torch.from_numpy(reference_mask).float().unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        
        ref_img_t = F.interpolate(ref_img_t, size=self.crop_size, mode='bilinear', align_corners=False)
        ref_mask_t = F.interpolate(ref_mask_t, size=self.crop_size, mode='nearest')
        
        ref_img_t = ref_img_t.repeat(1, 3, 1, 1).to(self.device) # [1, 3, H, W]
        ref_mask_t = ref_mask_t.to(self.device) # [1, 1, H, W]
        
        # Initialize Memory Bank
        # Memory stores tuples of (target_tokens, global_repr)
        memory_bank = []
        
        # Encode initial reference
        # We need to access the model's encoder helper
        # Check if model has encode_reference (added in recent edit)
        if hasattr(self.model, 'encode_reference'):
             init_tokens, init_global = self.model.encode_reference(ref_img_t.squeeze(0), ref_mask_t.squeeze(0))
             memory_bank.append({'tokens': init_tokens, 'global': init_global}) # Tokens: [1, N, D]
        else:
             # Fallback if method missing (shouldn't happen with correct flow)
             print("Warning: model.encode_reference not found. Memory bank disabled.")
             self.mode = 'global'
        
        # Prediction placeholder
        full_pred = np.zeros((D, H, W), dtype=np.uint8)
        
        # State for Sequential/Global legacy modes
        curr_ref_img = ref_img_t
        curr_ref_mask = ref_mask_t
        mamba_state = None
        
        # Chunk Processing
        for start_idx in range(0, D, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, D)
            chunk_len = end_idx - start_idx
            
            # Extract chunk
            chunk_data = volume[start_idx:end_idx] # [T, H, W]
            
            # Prepare Chunk Tensor
            chunk_t = torch.from_numpy(chunk_data).float().unsqueeze(1) # [T, 1, H, W]
            chunk_t = F.interpolate(chunk_t, size=self.crop_size, mode='bilinear', align_corners=False)
            chunk_t = chunk_t.repeat(1, 3, 1, 1) # [T, 3, H, W]
            chunk_t = chunk_t.unsqueeze(0).to(self.device) # [1, T, 3, H, W]
            
            # Prepare Spacing Tensor [1, T]
            spacing_t = torch.ones((1, chunk_len), device=self.device) * z_spacing
            if start_idx == 0:
                spacing_t[0, 0] = 0.0 # Relative to previous chunk end? Or relative within chunk? 
            # PhysioMamba expects delta_t.
            # Ideally, spacing[0] is distance from *previous frame* (last frame of previous chunk).
            # But here we treat chunk somewhat independently in 'forward', 
            # except state is not passed (stateless mamba in inference for now).
            # So spacing[0]=0 implies it's a restart. 
            # Phase 2 TODO: Implement stateful inference for Mamba to carry hidden state.
            
            # Inference based on mode
            if self.mode == 'memory_bank':
                # Prepare Memory Inputs
                # Concatenate tokens from all memories
                # tokens: [1, N, D] -> list -> [1, Total_N, D]
                input_tokens = torch.cat([m['tokens'] for m in memory_bank], dim=1)
                
                # Average global representations
                # global: [1, D]
                input_global = torch.mean(torch.stack([m['global'] for m in memory_bank], dim=0), dim=0)
                
                output = self.model(
                    chunk_t,
                    reference_tokens=input_tokens,
                    reference_global=input_global,
                    spacing=spacing_t,
                    mamba_state=mamba_state,
                    return_mamba_state=True,
                )
            else:
                # Legacy modes (Global / Sequential)
                output = self.model(
                    chunk_t,
                    reference_frame=curr_ref_img,
                    reference_mask=curr_ref_mask,
                    spacing=spacing_t,
                    mamba_state=mamba_state,
                    return_mamba_state=True,
                )

            if 'mamba_state' in output:
                mamba_state = output['mamba_state']
            
            pred_masks = output['masks'] # [1, T, 1, H, W]
            pred_prob = torch.sigmoid(pred_masks)
            pred_binary = (pred_prob > self.threshold).float()
            
            # --- Update Memory / Reference ---
            
            if self.mode == 'memory_bank' and end_idx < D:
                # Get last frame prediction
                last_pred_mask = pred_binary[:, -1:, :, :, :] # [1, 1, 1, H, W]
                last_pred_mask = last_pred_mask.squeeze(1) # [1, 1, H, W]
                
                # Check validity (not empty)
                if last_pred_mask.sum() > 10: # Min pixels
                    # Encode new memory
                    last_img = chunk_t[:, -1, :, :, :] # [1, 3, H, W]
                    
                    with torch.no_grad():
                        new_tokens, new_global = self.model.encode_reference(last_img, last_pred_mask)
                    
                    # Add to bank
                    memory_bank.append({'tokens': new_tokens, 'global': new_global})
                    
                    # Prune if too large
                    # Keep Index 0 (Ground Truth) + last (max_memory-1)
                    if len(memory_bank) > self.max_memory:
                        # Remove index 1 (oldest generated memory)
                        memory_bank.pop(1)
            
            elif self.mode == 'sequential' and end_idx < D:
                # Legacy Sequential update
                last_pred_mask = pred_binary[:, -1:, :, :, :]
                last_pred_mask = last_pred_mask.squeeze(1)
                if last_pred_mask.sum() > 0:
                    curr_ref_mask = last_pred_mask
                    curr_ref_img = chunk_t[:, -1, :, :, :]
            
            # Resize back
            pred_binary_orig = F.interpolate(
                pred_binary.squeeze(0), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).cpu().numpy() # [T, H, W]
            
            full_pred[start_idx:end_idx] = pred_binary_orig.astype(np.uint8)
            
        if slice_axis == 2:
            full_pred = full_pred.transpose(1, 2, 0) # [H, W, D]
            
        return full_pred

def load_volume(image_path, label_path):
    img_obj = nib.load(image_path)
    img = img_obj.get_fdata()
    lbl = nib.load(label_path).get_fdata()
    
    # Get Z-spacing
    z_spacing = img_obj.header['pixdim'][3]
    
    # Normalize Image
    img = np.clip(img, -1000, 1000)
    img = (img - (-1000)) / 2000.0
    
    return img, lbl, z_spacing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/interactive_amos.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/amos22_preprocessed')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--save_preds', action='store_true', help='Save prediction NIfTI files')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/interactive')
    parser.add_argument('--max_volumes', type=int, default=-1)
    parser.add_argument('--class_ids', type=str, default=None)
    parser.add_argument('--chunk_size', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--volume_names', type=str, default=None)
    parser.add_argument('--mode', type=str, default='global', choices=['global', 'sequential', 'memory_bank'], help='Propagation mode')
    parser.add_argument('--max_memory', type=int, default=5, help='Max memories for memory_bank mode')
    parser.add_argument('--slice_stride', type=int, default=1, help='Subsample slices along z-axis (stride>1) for extrapolation test')
    parser.add_argument('--spacing_scale', type=float, default=1.0, help='Multiply z-spacing by this factor for extrapolation test')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    # Build Model
    print("Building model...")
    model = DINOv3PhysioMambaInteractive(
        model_name=config['model'].get('name', 'dinov3_vitb16'),
        pretrained_weights=None, # Loading from checkpoint
        unfreeze_last_n_blocks=config['model'].get('unfreeze_last_n_blocks', 4),
        d_state=config['model'].get('d_state', 16),
        dt_rank=config['model'].get('dt_rank', 8),
        expand=config['model'].get('expand', 2),
        num_target_tokens=config['model'].get('num_target_tokens', 64),
        use_mamba=config['model'].get('use_mamba', True),
        use_parallel_scan=config['model'].get('use_parallel_scan', False),
        rope_type=config['model'].get('rope_type', 'segmentation'),
    ).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    # Handle DDP prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    # Inference Engine
    inference_engine = VolumeInference(
        model, 
        device, 
        chunk_size=args.chunk_size if args.chunk_size is not None else config['data'].get('seq_length', 8),
        crop_size=tuple(config['data'].get('crop_size', [512, 512])),
        threshold=args.threshold,
        mode=args.mode,
        max_memory=args.max_memory
    )
    
    # Data Search
    images_dir = Path(args.data_root) / ('imagesTr' if args.split == 'train' else 'imagesVa')
    labels_dir = Path(args.data_root) / ('labelsTr' if args.split == 'train' else 'labelsVa')
    
    if not images_dir.exists():
         # Fallback
         images_dir = Path(args.data_root) / 'images' / args.split
         labels_dir = Path(args.data_root) / 'labels' / args.split
         
    volume_files = sorted(list(images_dir.glob('*.nii.gz')))
    if args.volume_names:
        name_set = set()
        for part in args.volume_names.split(','):
            part = part.strip()
            if part:
                name_set.add(part)
        volume_files = [p for p in volume_files if p.name in name_set]
    if args.max_volumes is not None and args.max_volumes > 0:
        volume_files = volume_files[:args.max_volumes]
    print(f"Found {len(volume_files)} volumes in {images_dir}")

    class_filter = None
    if args.class_ids:
        class_filter = set()
        for part in args.class_ids.split(','):
            part = part.strip()
            if part:
                class_filter.add(int(part))
    
    results = []
    
    # Evaluation Loop
    for vol_file in tqdm(volume_files, desc="Evaluating Volumes"):
        vol_name = vol_file.name
        label_file = labels_dir / vol_name
        
        try:
            vol, lbl, z_spacing = load_volume(vol_file, label_file)
        except Exception as e:
            print(f"Error loading {vol_name}: {e}")
            continue

        if args.slice_stride is not None and args.slice_stride > 1:
            vol = vol[:, :, ::args.slice_stride]
            lbl = lbl[:, :, ::args.slice_stride]
            z_spacing = z_spacing * args.slice_stride

        if args.spacing_scale is not None and args.spacing_scale != 1.0:
            z_spacing = z_spacing * float(args.spacing_scale)
            
        # Identify classes
        classes = np.unique(lbl)
        classes = classes[classes > 0] # Skip background

        if class_filter is not None:
            classes = np.array([c for c in classes if int(c) in class_filter])
        
        vol_results = {'name': vol_name}
        
        for cls_id in classes:
            # 1. Prepare Binary Ground Truth
            gt_binary = (lbl == cls_id).astype(np.uint8)
            
            # 2. Select Prompt Slice (Max Area)
            # Sum across H,W to find area per slice
            area_per_slice = gt_binary.sum(axis=(0, 1))
            best_slice_idx = np.argmax(area_per_slice)
            
            if area_per_slice[best_slice_idx] == 0:
                continue # Should not happen
                
            # 3. Get Reference Info
            ref_slice_img = vol[:, :, best_slice_idx]
            ref_slice_mask = gt_binary[:, :, best_slice_idx]
            
            # 4. Run Inference
            pred_binary = inference_engine.predict_volume(
                vol, ref_slice_img, ref_slice_mask, z_spacing=z_spacing, slice_axis=2
            )
            
            # 5. Compute Metrics
            dice = compute_dice(pred_binary, gt_binary)
            iou = compute_iou(pred_binary, gt_binary)
            
            vol_results[f'class_{int(cls_id)}_dice'] = dice
            vol_results[f'class_{int(cls_id)}_iou'] = iou
            
            # Save prediction if requested (only save first class or merge? Usually merge)
            # For simplicity, we skip merging save here to save space/time, unless needed.
            
        results.append(vol_results)
        
    # Summarize
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    # Save results
    csv_path = output_dir / 'evaluation_metrics.csv'
    fieldnames = ['name'] + sorted([k for k in all_keys if k != 'name'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Metrics saved to {csv_path}")
    
    # Calculate Mean Dice per Class across volumes
    print("\n=== Evaluation Results ===")
    mean_dices = {}
    for col in sorted(all_keys):
        if col == 'name' or 'dice' not in col:
            continue
        values = []
        for r in results:
            v = r.get(col, None)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            if np.isnan(v):
                continue
            values.append(v)
        if len(values) == 0:
            continue
        mean_dices[col] = float(np.mean(values))
        print(f"{col}: Dice={mean_dices[col]:.4f} (n={len(values)})")
            
    # Calculate Macro Average Dice (Average of class averages)
    macro_dice = float(np.mean(list(mean_dices.values()))) if len(mean_dices) > 0 else float('nan')
    print(f"\nMacro Average Dice: {macro_dice:.4f}")

if __name__ == '__main__':
    main()
