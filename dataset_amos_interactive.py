import os
import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F

class AMOSInteractiveDataset(Dataset):
    """
    AMOS 2022 Interactive Dataset
    
    Treats medical segmentation as a Video Object Segmentation (VOS) task.
    For each sequence:
    1. Selects a random organ class present in the volume.
    2. Converts the multi-class label to a binary mask (Selected Organ vs Background).
    3. Returns a sequence of 2.5D slices or 3D frames along with the reference mask.
    
    This enables class-agnostic training compatible with SAM2-style prompting.
    """
    def __init__(
        self, 
        data_root, 
        split='train', 
        seq_length=8, 
        crop_size=(512, 512),
        min_organ_pixels=100,
        max_jump=1,  # Max frame skip for long-range training
        enable_zoom=False, # Small organ optimization
        zoom_ratio=0.5     # Probability to apply zoom when organ is detected
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.seq_length = seq_length
        self.crop_size = crop_size
        self.min_organ_pixels = min_organ_pixels
        self.max_jump = max_jump
        self.enable_zoom = enable_zoom
        self.zoom_ratio = zoom_ratio
        
        # Paths - adjust based on actual AMOS structure
        # Assuming AMOS structure:
        # data_root/
        #   imagesTr/ (nii.gz)
        #   labelsTr/ (nii.gz)
        #   imagesVa/
        #   labelsVa/
        
        self.images_dir = self.data_root / ('imagesTr' if split == 'train' else 'imagesVa')
        self.labels_dir = self.data_root / ('labelsTr' if split == 'train' else 'labelsVa')
        
        if not self.images_dir.exists():
             # Fallback to Task07/Task08 structure if AMOS not found, or maybe standard AMOS structure
             # Try simple structure
             self.images_dir = self.data_root / 'images' / split
             self.labels_dir = self.data_root / 'labels' / split
        
        # Scan for available volumes
        self.volume_files = sorted([f.name for f in self.images_dir.glob('*.nii.gz')])
        if len(self.volume_files) == 0:
             print(f"Warning: No .nii.gz files found in {self.images_dir}")
        else:
             print(f"Found {len(self.volume_files)} volumes for {split}")
        
    def __len__(self):
        # We can sample multiple times from each volume
        return len(self.volume_files) * 20 

    def load_volume(self, filename):
        img_path = self.images_dir / filename
        lbl_path = self.labels_dir / filename
        
        # Load NIfTI
        img_obj = nib.load(str(img_path))
        lbl_obj = nib.load(str(lbl_path))
        
        img = img_obj.get_fdata()
        lbl = lbl_obj.get_fdata()
        
        # Get Z-spacing (usually index 2 in pixdim, index 0 is qfac)
        # header['pixdim'] -> [qfac, x, y, z, t, ...]
        z_spacing = img_obj.header['pixdim'][3]
        
        # Normalize Image
        # CT usually -1000 to 1000 HU. Clip and Normalize to 0-1
        img = np.clip(img, -1000, 1000)
        img = (img - (-1000)) / 2000.0
        
        return img, lbl, z_spacing

    def __getitem__(self, idx):
        # Select volume
        vol_idx = idx % len(self.volume_files)
        filename = self.volume_files[vol_idx]
        
        try:
            vol, label, z_spacing = self.load_volume(filename) # [H, W, D] usually
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        # Ensure dimensions [H, W, D] -> We treat D as Time (T)
        # Some NIfTI are [D, H, W]? Usually nibabel loads as [H, W, D]
        if vol.ndim != 3:
            return self.__getitem__(random.randint(0, len(self) - 1))
            
        # Find present classes
        present_classes = np.unique(label)
        present_classes = present_classes[present_classes > 0] # Exclude background
        
        if len(present_classes) == 0:
             return self.__getitem__(random.randint(0, len(self) - 1))
             
        # Pick a random target organ
        target_class = np.random.choice(present_classes)
        
        # Find slices containing this organ
        organ_presence = (label == target_class).sum(axis=(0, 1))
        valid_slices = np.where(organ_presence > self.min_organ_pixels)[0]
        
        if len(valid_slices) < self.seq_length:
             # Organ too small/few slices, retry
             return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Dynamic Dilated Sampling (Random Walk)
        # We want to pick seq_length frames with variable gaps
        # Gap range: [1, max_jump]
        
        # 1. Pick a random start slice
        # To ensure we can fit the sequence, we need to be careful.
        # Simple heuristic: try to construct a valid sequence.
        
        max_attempts = 10
        selected_indices = None
        
        for _ in range(max_attempts):
            start_idx_in_valid = random.randint(0, len(valid_slices) - 1)
            current_valid_idx = start_idx_in_valid
            
            sequence_indices = [valid_slices[current_valid_idx]]
            spacings = [0.0] # Relative distance from previous frame
            
            success = True
            for _ in range(self.seq_length - 1):
                # Determine jump size (in terms of valid slices index, or actual slice index?)
                # Let's jump in terms of *actual slices* to simulate physical distance better, 
                # but we must land on a valid slice.
                # Actually, simply jumping in valid_slices is safer to ensure organ presence.
                # But to learn "long range" where organ might disappear/reappear, we strictly need 
                # to handle frames where organ is NOT present? 
                # For Phase 2, let's stick to valid_slices for stability, but increase jump size.
                
                step = random.randint(1, self.max_jump)
                next_valid_idx = current_valid_idx + step
                
                if next_valid_idx >= len(valid_slices):
                    success = False
                    break
                
                current_valid_idx = next_valid_idx
                next_slice_idx = valid_slices[current_valid_idx]
                prev_slice_idx = sequence_indices[-1]
                
                sequence_indices.append(next_slice_idx)
                spacings.append((next_slice_idx - prev_slice_idx) * z_spacing)
            
            if success:
                selected_indices = sequence_indices
                # spacing tensor: delta_t from previous frame
                # spacing[0] is 0 (reference)
                spacing_tensor = torch.tensor(spacings, dtype=torch.float32)
                break
        
        if selected_indices is None:
            # Fallback to contiguous
            if len(valid_slices) < self.seq_length:
                return self.__getitem__(random.randint(0, len(self) - 1))
            start = random.randint(0, len(valid_slices) - self.seq_length)
            selected_indices = valid_slices[start : start + self.seq_length]
            
            # Calculate actual spacing
            spacings = [0.0]
            for i in range(1, self.seq_length):
                dist = (selected_indices[i] - selected_indices[i-1]) * z_spacing
                spacings.append(dist)
            spacing_tensor = torch.tensor(spacings, dtype=torch.float32)
        
        # Extract frames and masks
        frames = []
        masks = []
        
        # 1. Load Raw Slices
        raw_slices_img = []
        raw_slices_lbl = []
        
        for d in selected_indices:
            img_slice = vol[:, :, d]
            lbl_slice = (label[:, :, d] == target_class).astype(np.float32)
            raw_slices_img.append(img_slice)
            raw_slices_lbl.append(lbl_slice)
            
        # 2. Determine Crop/Zoom Coordinates
        H, W = vol.shape[:2]
        crop_y_min, crop_y_max, crop_x_min, crop_x_max = 0, H, 0, W
        do_zoom = False
        
        if self.enable_zoom and random.random() < self.zoom_ratio:
            # Find bbox across sequence
            stack_lbl = np.stack(raw_slices_lbl) # [T, H, W]
            ys, xs = np.where(stack_lbl.sum(axis=0) > 0)
            
            if len(ys) > 0:
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                
                # Add padding (e.g. 20% or min 20 pixels)
                h_bbox = y_max - y_min
                w_bbox = x_max - x_min
                
                pad_h = int(max(h_bbox * 0.2, 20))
                pad_w = int(max(w_bbox * 0.2, 20))
                
                # Random shift of padding
                shift_h = random.randint(-pad_h // 2, pad_h // 2) if pad_h > 0 else 0
                shift_w = random.randint(-pad_w // 2, pad_w // 2) if pad_w > 0 else 0
                
                y_c_min = max(0, y_min - pad_h + shift_h)
                y_c_max = min(H, y_max + pad_h + shift_h)
                x_c_min = max(0, x_min - pad_w + shift_w)
                x_c_max = min(W, x_max + pad_w + shift_w)
                
                # Check if zoom is meaningful (not essentially full image)
                if (y_c_max - y_c_min) < H * 0.9 or (x_c_max - x_c_min) < W * 0.9:
                    crop_y_min, crop_y_max = y_c_min, y_c_max
                    crop_x_min, crop_x_max = x_c_min, x_c_max
                    do_zoom = True

        # 3. Process Slices (Crop -> Resize -> Tensor)
        for i in range(len(selected_indices)):
            img_slice = raw_slices_img[i]
            lbl_slice = raw_slices_lbl[i]
            
            # Apply Crop if Zooming
            if do_zoom:
                img_slice = img_slice[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
                lbl_slice = lbl_slice[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            # Convert to Tensor
            img_t = torch.from_numpy(img_slice).float().unsqueeze(0) # [1, H, W]
            lbl_t = torch.from_numpy(lbl_slice).float().unsqueeze(0) # [1, H, W]
            
            # Resize
            img_t = F.interpolate(img_t.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=False).squeeze(0)
            lbl_t = F.interpolate(lbl_t.unsqueeze(0), size=self.crop_size, mode='nearest').squeeze(0)
            
            # Replicate channels to 3 for DINOv3 input
            img_t = img_t.repeat(3, 1, 1)
            
            frames.append(img_t)
            masks.append(lbl_t)
            
        frames = torch.stack(frames) # [T, 3, H, W]
        masks = torch.stack(masks)   # [T, 1, H, W]
        
        # Reference: Frame 0
        reference_frame = frames[0]
        reference_mask = masks[0]
        
        return {
            'frames': frames,
            'masks': masks,
            'reference_frame': reference_frame,
            'reference_mask': reference_mask,
            'spacing': spacing_tensor, # [T]
            'target_class': int(target_class),
            'seq_name': filename
        }
