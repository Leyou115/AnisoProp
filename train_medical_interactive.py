#!/usr/bin/env python
"""
DINOv3-PhysioMamba Medical Interactive Training Script

Trains a class-agnostic, prompt-based segmentation model on AMOS 2022.
Uses a Video Object Segmentation (VOS) approach:
- Input: Volume as a sequence of frames
- Prompt: Reference mask for a specific organ (on one slice)
- Output: Segmentation of that organ across the sequence

Usage:
    python train_medical_interactive.py --config configs/interactive_amos.yaml --gpu 0
"""

import os
import sys
import argparse
import yaml
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dataset_amos_interactive import AMOSInteractiveDataset
from evaluate_medical_interactive import VolumeInference, load_volume
from model.dinov3_src.utils import fix_random_seeds

def setup_ddp():
    """Initialize Distributed Data Parallel"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        # Optimization: Enable TF32 on Ampere+ GPUs (A100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return rank, local_rank, world_size
    return 0, 0, 1

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

def dice_loss(pred, target, smooth=1e-5):
    """
    Dice loss for binary segmentation.
    Computes Dice per sample and averages, better for small organs/imbalanced batches.
    """
    pred = torch.sigmoid(pred)
    
    # Flatten spatial/temporal dims but keep batch dim
    # pred: [B, T, 1, H, W] -> [B, -1]
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    import sys
    import traceback
    print(f"[Debug][train_one_epoch] Entering function, epoch={epoch}", flush=True)
    model.train()
    print(f"[Debug][train_one_epoch] Model set to train mode.", flush=True)
    total_loss = 0
    total_dice = 0
    n_batches = len(dataloader)
    if n_batches == 0:
        raise RuntimeError(
            "Train dataloader has 0 batches. "
            f"Check config['data']['root_dir']={config.get('data', {}).get('root_dir')}"
        )
    
    print(f"[Debug][train_one_epoch] Creating dataloader iterator...", flush=True)
    batch_idx = 0
    try:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=not is_main_process())
        print(f"[Debug][train_one_epoch] Starting batch loop...", flush=True)
        for batch in pbar:
            if batch_idx == 0:
                print(f"[Debug][train_one_epoch] First batch received. Keys: {list(batch.keys())}", flush=True)
            frames = batch['frames'].to(device) # [B, T, 3, H, W]
            masks = batch['masks'].to(device)   # [B, T, 1, H, W]
            ref_frame = batch['reference_frame'].to(device) # [B, 3, H, W]
            ref_mask = batch['reference_mask'].to(device)   # [B, 1, H, W]
            
            # Ablation control: Disable physio spacing if requested
            if config['model'].get('use_physio_spacing', True):
                spacing = batch['spacing'].to(device)           # [B, T]
            else:
                spacing = None
            
            if batch_idx == 0:
                spacing_shape = spacing.shape if spacing is not None else "None"
                print(f"[Debug][train_one_epoch] First batch moved to device. frames={frames.shape}, masks={masks.shape}, spacing={spacing_shape}", flush=True)
            
            optimizer.zero_grad()

            train_chunk_size = int(
                config['training'].get(
                    'train_chunk_size',
                    config.get('data', {}).get('seq_length', 8),
                )
            )
            T_total = frames.shape[1]
            mamba_state = None
            batch_loss = 0.0
            batch_dice = 0.0

            if batch_idx == 0:
                print(
                    f"[Debug][train_one_epoch] Streaming forward: T_total={T_total}, train_chunk_size={train_chunk_size}, TF32={torch.backends.cuda.matmul.allow_tf32}",
                    flush=True,
                )

            # Check if we should reset state between chunks (Windowed/Stateless training)
            # Default is True (Stateful / TBPTT)
            use_stateful = config['training'].get('use_stateful', True)

            for start in range(0, T_total, train_chunk_size):
                end = min(start + train_chunk_size, T_total)
                chunk_len = end - start
                chunk_weight = float(chunk_len) / float(T_total)

                chunk_frames = frames[:, start:end]
                chunk_masks = masks[:, start:end]
                chunk_spacing = spacing[:, start:end] if spacing is not None else None

                if batch_idx == 0:
                    print(f"[Debug][train_one_epoch] Chunk {start}:{end} (len={chunk_len})", flush=True)

                with autocast('cuda', enabled=config['training'].get('use_amp', True)):
                    output = model(
                        chunk_frames,
                        reference_frame=ref_frame,
                        reference_mask=ref_mask,
                        spacing=chunk_spacing,
                        mamba_state=mamba_state,
                        return_mamba_state=True,
                    )

                    if batch_idx == 0 and start == 0:
                        print(f"[Debug][train_one_epoch] Forward pass done. output keys: {list(output.keys())}", flush=True)

                    pred_masks = output['masks']  # [B, chunk_len, 1, H, W]

                    loss_dice = dice_loss(pred_masks, chunk_masks)
                    loss_bce = F.binary_cross_entropy_with_logits(pred_masks, chunk_masks)

                    dice_w = config['loss'].get('dice_weight', 0.7)
                    bce_w = config['loss'].get('bce_weight', 0.3)
                    loss = dice_w * loss_dice + bce_w * loss_bce

                    loss_to_backward = loss * chunk_weight

                if batch_idx == 0 and start == 0:
                    print(f"[Debug][train_one_epoch] Loss computed. Running backward...", flush=True)

                scaler.scale(loss_to_backward).backward()

                batch_loss += loss.detach().item() * chunk_weight

                with torch.no_grad():
                    probs = torch.sigmoid(pred_masks)
                    pred_binary = (probs > 0.5).float()

                    dims = (1, 2, 3, 4)
                    intersection = (pred_binary * chunk_masks).sum(dim=dims)
                    union = pred_binary.sum(dim=dims) + chunk_masks.sum(dim=dims)
                    dice_tensor = (2.0 * intersection + 1e-5) / (union + 1e-5)
                    dice_chunk = dice_tensor.mean().item()
                    batch_dice += dice_chunk * chunk_weight

                    if batch_idx == 0 and start == 0 and is_main_process():
                        mask_fg = chunk_masks.mean().item()
                        pred_fg = pred_binary.mean().item()
                        prob_mean = probs.mean().item()
                        prob_min = probs.min().item()
                        prob_max = probs.max().item()
                        print(
                            "[Debug][train_one_epoch] First batch stats: "
                            f"mask_fg={mask_fg:.6f}, pred_fg={pred_fg:.6f}, "
                            f"prob_mean={prob_mean:.6f}, prob_min={prob_min:.6f}, prob_max={prob_max:.6f}",
                            flush=True,
                        )

                if 'mamba_state' in output:
                    mamba_state = output['mamba_state']
                    if mamba_state is not None:
                        if use_stateful:
                            mamba_state = tuple(s.detach() for s in mamba_state)
                        else:
                            # Windowed training: Reset state after every chunk
                            mamba_state = None

            scaler.unscale_(optimizer)

            if batch_idx == 0 and is_main_process():
                try:
                    m = model.module if hasattr(model, 'module') else model
                    head_grad = m.head.weight.grad
                    if head_grad is None:
                        print("[Debug][train_one_epoch] head.weight.grad is None", flush=True)
                    else:
                        print(f"[Debug][train_one_epoch] head.grad_norm={head_grad.norm().item():.6f}", flush=True)
                except Exception as _e:
                    print(f"[Debug][train_one_epoch] grad debug failed: {_e}", flush=True)

            scaler.step(optimizer)
            scaler.update()

            if batch_idx == 0:
                print(f"[Debug][train_one_epoch] Backward done.", flush=True)

            total_loss += float(batch_loss)
            total_dice += float(batch_dice)

            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'dice': f'{batch_dice:.4f}'})
            batch_idx += 1
    except Exception as e:
        print(f"[Debug][train_one_epoch] ERROR at batch {batch_idx}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
    
    return total_loss / n_batches, total_dice / n_batches

def validate_3d(model, val_dataset, device, config):
    """
    Perform full 3D volume validation using VolumeInference.
    """
    model.eval()
    
    # Initialize Inference Engine
    # Use 'memory_bank' mode for best performance during validation
    inference_engine = VolumeInference(
        model, 
        device, 
        chunk_size=config['data'].get('seq_length', 8),
        crop_size=tuple(config['data'].get('crop_size', [512, 512])),
        threshold=0.5,
        mode='memory_bank', 
        max_memory=5
    )
    
    total_dice = 0.0
    n_samples = 0
    class_metrics = {}
    
    # Use all validation volumes
    # Note: We run this only on the main process to simplify metric aggregation
    # If the validation set is very large, this might be slow. 
    # For AMOS val set (~30-40 volumes), it's manageable.
    
    volume_files = val_dataset.volume_files
    
    # Limit number of volumes for validation speed if needed (e.g. first 20)
    # volume_files = volume_files[:20] 

    for vol_filename in tqdm(volume_files, desc="3D Validation", disable=not is_main_process()):
        try:
             vol, lbl, z_spacing = val_dataset.load_volume(vol_filename)
        except Exception as e:
             print(f"Error loading {vol_filename}: {e}")
             continue
             
        # Identify classes
        classes = np.unique(lbl)
        classes = classes[classes > 0]
        
        for cls_id in classes:
            gt_binary = (lbl == cls_id).astype(np.uint8)
            
            # Select prompt (max area slice)
            area_per_slice = gt_binary.sum(axis=(0, 1))
            best_slice_idx = np.argmax(area_per_slice)
            
            if area_per_slice[best_slice_idx] == 0: continue
            
            ref_slice = vol[:, :, best_slice_idx]
            ref_mask = gt_binary[:, :, best_slice_idx]
            
            # Predict
            z_spacing_in = z_spacing if config['model'].get('use_physio_spacing', True) else None
            pred_binary = inference_engine.predict_volume(
                vol, ref_slice, ref_mask, z_spacing=z_spacing_in, slice_axis=2
            )
            
            # Compute Dice
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum()
            dice = 2.0 * intersection / (union + 1e-5) if union > 0 else 1.0
            
            if cls_id not in class_metrics:
                class_metrics[cls_id] = []
            class_metrics[cls_id].append(dice)
            
            total_dice += dice
            n_samples += 1
            
    # Summarize
    mean_dice = total_dice / n_samples if n_samples > 0 else 0.0
    
    if is_main_process():
        print("\n=== 3D Validation Results ===")
        print(f"Mean Dice (Global): {mean_dice:.4f}")
        # Per class stats (Macro Dice)
        macro_dice_sum = 0
        for cls_id in sorted(class_metrics.keys()):
            d_cls = np.mean(class_metrics[cls_id])
            print(f"Class {cls_id}: {d_cls:.4f}")
            macro_dice_sum += d_cls
        if len(class_metrics) > 0:
            print(f"Macro Average Dice: {macro_dice_sum / len(class_metrics):.4f}")
        print("=============================")
        
    return mean_dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/interactive_amos.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--d_model_dir', type=str, default=None)
    parser.add_argument('--d_summary_dir', type=str, default=None)
    parser.add_argument('--d_result_dir', type=str, default=None)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1) # For torchrun compatibility
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--rope_type', type=str, default=None, choices=['standard', 'physio_v1', 'segmentation'])
    parser.add_argument('--rope_base_temporal', type=float, default=None)
    parser.add_argument('--rope_base_spatial', type=float, default=None)
    parser.add_argument('--rope_temporal_ratio', type=float, default=None)
    parser.add_argument('--use_metric_positions', type=int, default=None, choices=[0, 1])
    parser.add_argument('--use_physio_spacing', type=int, default=None, choices=[0, 1])
    args = parser.parse_args()
    
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    config.setdefault('experiment', {})
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    
    exp_name = config['experiment'].get('name', Path(args.config).stem)
    cfg_output_dir = config['experiment'].get('output_dir') or f"outputs/{exp_name}"
    output_dir_path = Path(cfg_output_dir)
    project_root = Path(__file__).resolve().parent
    if not output_dir_path.is_absolute():
        if output_dir_path.parts and output_dir_path.parts[0] == 'outputs':
            output_dir_path = project_root / output_dir_path
        else:
            output_dir_path = project_root / 'outputs' / output_dir_path
    config['experiment']['output_dir'] = str(output_dir_path)

    config.setdefault('data', {})
    cfg_data_root = config['data'].get('root_dir')
    if cfg_data_root is not None:
        data_root_path = Path(cfg_data_root)
        if not data_root_path.is_absolute():
            data_root_path = project_root / data_root_path
        config['data']['root_dir'] = str(data_root_path)

    config.setdefault('model', {})
    pretrained_weights = config['model'].get('pretrained_weights')
    if pretrained_weights:
        pretrained_weights_path = Path(pretrained_weights)
        if not pretrained_weights_path.is_absolute():
            pretrained_weights_path = project_root / pretrained_weights_path
        config['model']['pretrained_weights'] = str(pretrained_weights_path)
    config.setdefault('training', {})

    config['training'].setdefault('seed', 42)
    config['model'].setdefault('use_metric_positions', True)
    config['model'].setdefault('use_physio_spacing', True)
    config['model'].setdefault('rope_type', 'segmentation')
    config['model'].setdefault('rope_base_temporal', 50000.0)
    config['model'].setdefault('rope_base_spatial', 10000.0)
    config['model'].setdefault('rope_temporal_ratio', 0.25)

    if args.seed is not None:
        config['training']['seed'] = int(args.seed)
    if args.use_metric_positions is not None:
        config['model']['use_metric_positions'] = bool(args.use_metric_positions)
    if args.use_physio_spacing is not None:
        config['model']['use_physio_spacing'] = bool(args.use_physio_spacing)
    if args.rope_type is not None:
        config['model']['rope_type'] = args.rope_type
    if args.rope_base_temporal is not None:
        config['model']['rope_base_temporal'] = float(args.rope_base_temporal)
    if args.rope_base_spatial is not None:
        config['model']['rope_base_spatial'] = float(args.rope_base_spatial)
    if args.rope_temporal_ratio is not None:
        config['model']['rope_temporal_ratio'] = float(args.rope_temporal_ratio)

    seed = int(config['training'].get('seed', 42))
    fix_random_seeds(seed + int(rank))
        
    if is_main_process():
        print("="*60)
        print(f"DINOv3-PhysioMamba Medical Interactive Training ({world_size} GPUs)")
        print("="*60)
        
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process():
        with open(output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)
            
    # Dataset
    if is_main_process():
        print("Loading AMOS datasets...")
        
    train_dataset = AMOSInteractiveDataset(
        config['data']['root_dir'],
        split='train',
        seq_length=config['data'].get('train_seq_length', config['data'].get('seq_length', 8)),
        crop_size=tuple(config['data'].get('crop_size', [512, 512])),
        max_jump=config['data'].get('max_jump', 5), # Default jump up to 5 slices
        enable_zoom=config['data'].get('enable_zoom', False),
        zoom_ratio=config['data'].get('zoom_ratio', 0.5)
    )
    
    val_dataset = AMOSInteractiveDataset(
        config['data']['root_dir'],
        split='val',
        seq_length=config['data'].get('seq_length', 8),
        crop_size=tuple(config['data'].get('crop_size', [512, 512])),
        max_jump=1 
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    
    # Use DistributedSampler for validation too if we want parallel eval, 
    # but for 3D validate_3d we are running on main process only for now.
    # The val_loader is still used if we want quick checks, but we are switching to validate_3d.
    # We keep val_loader instantiation for compatibility if needed, but validate_3d uses dataset directly.
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    # Model
    if is_main_process():
        print("Building model...")
        
    from model.interactive import DINOv3PhysioMambaInteractive
    
    import traceback
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        print(f"[Debug] Creating model on CPU first...", flush=True)
        model = DINOv3PhysioMambaInteractive(
            model_name=config['model'].get('name', 'dinov3_vitb16'),
            pretrained_weights=config['model'].get('pretrained_weights'),
            d_state=config['model'].get('d_state', 16),
            dt_rank=config['model'].get('dt_rank', 8),
            expand=config['model'].get('expand', 2),
            num_target_tokens=config['model'].get('num_target_tokens', 64),
            use_mamba=config['model'].get('use_mamba', True),
            use_parallel_scan=config['model'].get('use_parallel_scan', False),
            use_metric_positions=config['model'].get('use_metric_positions', True),
            rope_type=config['model'].get('rope_type', 'segmentation'),
            rope_base_temporal=config['model'].get('rope_base_temporal', 50000.0),
            rope_base_spatial=config['model'].get('rope_base_spatial', 10000.0),
            rope_temporal_ratio=config['model'].get('rope_temporal_ratio', 0.25),
        )
        print(f"[Debug] Model created on CPU. Moving to device {device}...", flush=True)
        model = model.to(device)
        print(f"[Debug] Model moved to {device} successfully.", flush=True)

        if not config['training'].get('find_unused_parameters', True):
            ddp_debug_param_idx = os.environ.get('DDP_DEBUG_PARAM_IDX', '')
            if ddp_debug_param_idx:
                try:
                    idx_list = [int(x) for x in ddp_debug_param_idx.split(',') if x.strip()]
                    for i, (n, p) in enumerate(model.named_parameters()):
                        if i in idx_list:
                            print(
                                f"[Debug][DDP] param_idx={i} name={n} shape={tuple(p.shape)} "
                                f"requires_grad={p.requires_grad}",
                                flush=True,
                            )
                except Exception as e:
                    print(
                        f"[Warning] Failed to parse DDP_DEBUG_PARAM_IDX='{ddp_debug_param_idx}': {e}",
                        flush=True,
                    )
            modules_to_freeze = []
            if hasattr(model, 'occlusion_head'):
                modules_to_freeze.append(model.occlusion_head)
            if hasattr(model, 'click_encoder'):
                modules_to_freeze.append(model.click_encoder)
            if hasattr(model, 'feature_matcher'):
                modules_to_freeze.append(model.feature_matcher)
            if hasattr(model, 'memory_bank'):
                modules_to_freeze.append(model.memory_bank)
            if hasattr(model, 'reference_encoder') and hasattr(model.reference_encoder, 'mask_conv'):
                modules_to_freeze.append(model.reference_encoder.mask_conv)
            if hasattr(model, 'cross_matching') and hasattr(model.cross_matching, 'confidence_head'):
                modules_to_freeze.append(model.cross_matching.confidence_head)
            for m_ in modules_to_freeze:
                for p in m_.parameters():
                    p.requires_grad = False

        # Optimization: torch.compile for PyTorch 2.0+
        if config['training'].get('compile', False):
            print(f"[Debug] Compiling model with torch.compile...", flush=True)
            try:
                model = torch.compile(model)
                print(f"[Debug] Model compiled successfully.", flush=True)
            except Exception as e:
                print(f"[Warning] torch.compile failed: {e}. Proceeding without compilation.", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR during model initialization: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
    
    try:
        if world_size > 1:
            print(f"[Debug] Wrapping model with DDP (world_size={world_size}, local_rank={local_rank})...", flush=True)
            # DDP Optimization: find_unused_parameters=False is faster if graph is static
            find_unused = config['training'].get('find_unused_parameters', True)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)
            print(f"[Debug] DDP wrapper created successfully. find_unused_parameters={find_unused}", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR during DDP wrapping: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
        
    # Optimizer
    try:
        print(f"[Debug] Creating optimizer...", flush=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training'].get('weight_decay', 0.01))
        )
        print(f"[Debug] Optimizer created. LR={config['training']['learning_rate']}", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR during optimizer creation: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
    
    try:
        print(f"[Debug] Creating scheduler...", flush=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs']
        )
        print(f"[Debug] Scheduler created. T_max={config['training']['epochs']}", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR during scheduler creation: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
    
    try:
        print(f"[Debug] Creating GradScaler...", flush=True)
        scaler = GradScaler('cuda', enabled=config['training'].get('use_amp', True))
        print(f"[Debug] GradScaler created. AMP={config['training'].get('use_amp', True)}", flush=True)
    except Exception as e:
        print(f"CRITICAL ERROR during GradScaler creation: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise e
    
    # Training Loop
    start_epoch = 0
    best_dice = 0
    
    print(f"[Debug] Preparing training loop...", flush=True)
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0)
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}")
    
    print(f"[Debug] Starting training from epoch {start_epoch} to {config['training']['epochs']}...", flush=True)
    print(f"[Debug] Train loader has {len(train_loader)} batches.", flush=True)
    
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            print(f"[Debug] Epoch {epoch} starting...", flush=True)
            if train_sampler:
                train_sampler.set_epoch(epoch)
                print(f"[Debug] Sampler epoch set.", flush=True)
                
            print(f"[Debug] Calling train_one_epoch...", flush=True)
            loss, train_dice = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
            print(f"[Debug] train_one_epoch returned. Loss={loss:.4f}, Dice={train_dice:.4f}", flush=True)
            scheduler.step()
            
            if is_main_process():
                print(f"Epoch {epoch}: Loss={loss:.4f}, TrainDice={train_dice:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
                
            if (epoch + 1) % config['training'].get('val_every', 5) == 0:
                # Use validate_3d only on main process
                if is_main_process():
                    # Unwrap model for validation engine
                    # VolumeInference needs access to custom methods like 'encode_reference'
                    # which are not exposed by DDP wrapper.
                    val_model = model.module if hasattr(model, 'module') else model
                    
                    val_dice = validate_3d(val_model, val_dataset, device, config)
                    print(f"Validation (3D Volume): Dice={val_dice:.4f}")
                    
                    if val_dice > best_dice:
                        best_dice = val_dice
                        torch.save({
                            'epoch': epoch,
                            'model': model.module.state_dict() if world_size > 1 else model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_dice': best_dice
                        }, output_dir / 'best_model.pth')
                        print(f"Saved best model (Dice={best_dice:.4f})")
                
                # Sync barrier
                if world_size > 1:
                    dist.barrier()
                        
            if is_main_process() and (epoch + 1) % config['training'].get('save_every', 10) == 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice
                }, output_dir / f'checkpoint_{epoch}.pth')
                
    except Exception as e:
        print(f"[Debug] ERROR in training loop at epoch {epoch}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        cleanup_ddp()
        raise e

    cleanup_ddp()

    if is_main_process():
        print("Training completed.")

if __name__ == '__main__':
    main()
