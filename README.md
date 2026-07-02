# AnisoProp

**Sampling-Aware Hybrid State-Space Propagation for Prompt-Guided 3D Medical Image Segmentation**

This repository contains the **minimal, reproducible** code for the method described in the paper — the model, training, and evaluation. It is intentionally trimmed to the core contribution.

## Method overview

AnisoProp performs slice-wise encoding with a DINOv3 ViT-B/16 backbone and propagates segmentation state along the depth (z) axis with a **spacing-aware** hybrid state-space model. The key idea is that state transitions and positional embeddings are modulated by the **real physical slice spacing (Δz)** rather than by discrete slice indices, which makes propagation robust to anisotropic / variable-thickness volumes. Interactive prompts (reference slice + clicks) and a memory bank keep long-volume inference stable.

## File map

| Path | Role |
|------|------|
| `model/mamba3/physio_mamba_primitives.py` | Core: metric/Δz-aware state dynamics + Segmentation-MRoPE positional encoding |
| `model/mamba3/mimo_adapter.py` | `PhysioMambaBlock` — depth-wise recurrent propagation block |
| `model/dinov3_multi_modal_seg.py` | DINOv3 encoder wrapper + FPN decoder |
| `model/interactive/interactive_model.py` | Main model `DINOv3PhysioMambaInteractive` |
| `model/interactive/reference_encoder.py` | Reference-slice / click prompt (bounded prompt memory) encoding |
| `model/interactive/cross_frame_matching.py` | Cross-frame matching + `MemoryBank` for drift-free long-volume inference |
| `dataset_amos_interactive.py` | AMOS dataset loader with dynamic dilated sampling (physical spacing) |
| `train_medical_interactive.py` | Training loop |
| `evaluate_medical_interactive.py` | Volume-level inference & evaluation (global / sequential / memory_bank) |
| `configs/interactive_amos.yaml` | Reference configuration |

## Dependencies

```bash
pip install -r requirements.txt
```

### DINOv3 backbone (required, external)

The encoder loads the **official** DINOv3 ViT source. To keep this repo minimal, the backbone is **not vendored** here. Place the official DINOv3 source package at `model/dinov3_src/` so that the following imports resolve:

- `from model.dinov3_src.models import vit_base, vit_large, vit_huge2, vit_giant2`
- `from model.dinov3_src.utils import fix_random_seeds`

Then download the pretrained weights referenced in `configs/interactive_amos.yaml` (`model.pretrained_weights`, e.g. `weights/dinov3_vitb16_pretrain_lvd1689m.pth`).

> Obtain the DINOv3 code and weights from the official DINOv3 release (facebookresearch/dinov3).

## Data

Expects a preprocessed AMOS 2022 dataset under `data/amos22_preprocessed` (set `data.root_dir` in the config). Each volume retains its physical z-spacing, which drives the spacing-aware propagation.

## Train

```bash
python train_medical_interactive.py --config configs/interactive_amos.yaml --gpu 0
```

Ablation (disable spacing awareness) via CLI overrides, e.g.:

```bash
python train_medical_interactive.py --config configs/interactive_amos.yaml \
    --rope_type standard --use_metric_positions 0 --use_physio_spacing 0
```

## Evaluate

```bash
python evaluate_medical_interactive.py --config configs/interactive_amos.yaml \
    --checkpoint outputs/best_model.pth --mode memory_bank
```

Thickness / anisotropy robustness (subsample z + rescale spacing):

```bash
python evaluate_medical_interactive.py --config configs/interactive_amos.yaml \
    --checkpoint outputs/best_model.pth --slice_stride 2 --spacing_scale 2.0
```
