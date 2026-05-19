# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

MoCE-IR (CVPR 2025) — "Complexity Experts are Task-Discriminative Learners for Any Image Restoration." A Mixture-of-Experts (MoE) architecture for all-in-one image restoration that handles multiple degradations (denoising, dehazing, deraining, deblurring, low-light enhancement) in a single model. Built on PyTorch Lightning.

Two model variants:
- **MoCE-IR** (heavy): `dim=48`, `num_blocks=[4,6,6,8]`, `num_dec_blocks=[2,4,4]`
- **MoCE-IR-S** (lightweight): `dim=32`, same block layout

## Environment

Conda environment defined in `environment.yml` (name: `moce`, Python 3.10, PyTorch 2.5.1 + CUDA 11.8).

```bash
conda env create -f environment.yml
conda activate moce
```

There is no `install.sh` in the repo; dependencies come from the conda environment file. The pip section installs additional packages including `pytorch-lightning`, `einops`, `fvcore`, `wandb`, `torchmetrics`, etc.

## Commands

### Training

```bash
# MoCE-IR-S on 3 degradations (standard config)
python src/train.py --model MoCE_IR_S --batch_size 8 --de_type denoise_15 denoise_25 denoise_50 dehaze derain \
    --trainset standard --num_gpus 4 --loss_type FFT --fft_loss_weight 0.1 --balance_loss_weight 0.01

# MoCE-IR on 5 degradations
python src/train.py --model MoCE_IR --batch_size 8 --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie \
    --trainset standard --num_gpus 4 --loss_type FFT --fft_loss_weight 0.1 --balance_loss_weight 0.01

# On CDD11 composited degradations
python src/train.py --model MoCE_IR --batch_size 8 --trainset CDD11_all --num_gpus 4 \
    --loss_type FFT --balance_loss_weight 0.01 --fft_loss_weight 0.1 --de_type denoise_15 denoise_25 denoise_50 dehaze derain
```

`--batch_size` is per-GPU. `--accum_grad` > 1 divides the batch size. Checkpoints save to `checkpoints/<timestamp>/` every 5 epochs plus `last.ckpt`. Logs go to `output/logs/<timestamp>/` (TensorBoard by default; pass `--wblogger` for Weights & Biases).

### Evaluation

```bash
# Deraining (Rain100L) — 3-deg model
python src/test.py --model MoCE_IR --benchmarks derain --checkpoint_id <dir> \
    --de_type denoise_15 denoise_25 denoise_50 dehaze derain

# Dehazing (SOTS) — 5-deg model
python src/test.py --model MoCE_IR --benchmarks dehaze --checkpoint_id <dir> \
    --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie

# CDD11 evaluation
python src/test.py --model MoCE_IR --checkpoint_id <dir> --trainset CDD11_haze_rain \
    --benchmarks cdd11 --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie
```

Checkpoints are loaded from `checkpoints/<checkpoint_id>/last.ckpt`. Results saved to `output/results/<checkpoint_id>/<benchmark>/` when `--save_results` is passed.

### Quick model smoketest

```bash
python src/net/moce_ir.py
```

Instantiates the model, runs a random input, prints loss, memory, and FLOPS.

## Architecture

```
Input → PatchEmbed → [EncoderGroup → Downsample]×3 → LatentGroup → FrequencyEmbedding
                                                                            ↓
  Output ← OutputConv ← RefinementGroup ← [Upsample → Concat ← DecoderGroup(Adapter)]×3
```

- **Encoder**: `OverlapPatchEmbed` (3×3 conv) → 4-level U-Net encoder. Each level: `EncoderResidualGroup` (stacked `EncoderBlock`s with self-attention + GDFN) followed by `Downsample` (conv + pixel-unshuffle).
- **Latent**: `EncoderResidualGroup` at the bottleneck, then `FrequencyEmbedding` (high-pass conv → GELU → MLP) produces a frequency embedding used for expert routing.
- **Decoder**: 3 levels, each: `Upsample` (pixel-shuffle), feature fusion (1×1 conv after concat with encoder skip), then `DecoderResidualGroup`. Each `DecoderBlock` splits input into a shared path (self-attention) and an adapter path (MoE), then cross-attends them.
- **Refinement**: Final `EncoderResidualGroup` + 3×3 conv output, with residual connection to input.

### MoE adapter (the key innovation)

`AdapterLayer` (`src/net/moce_ir.py:388`) contains:
1. **RoutingFunction**: spatial average pool → linear gate + frequency embedding gate → noisy top-k gating with auxiliary importance/load balancing losses. Optionally uses `complexity` bias (expert parameter count) to prefer simpler experts.
2. **Experts** (`ModExpert`): each expert is a bottleneck (project down → transform → project up) with a different FFT-based attention layer. The experts have **varying complexity** controlled by:
   - `depth`: number of stacked transform blocks (set by `depth_type`: constant, lin, double, exp, fact)
   - `rank`: bottleneck dimension (set by `rank_type`: constant, lin, double, exp, fact, spread)
   - `patch_size` for FFT attention: `[4, 8, 16, 32]` (2^(i+2))
   - `kernel_size`: `[3, 5, 7, 9]` (3+2i)

During training: sparse dispatch/combine. During inference: top-k gating directly.

### Loss

- Primary: L1 in pixel space (`nn.L1Loss`)
- Optional FFT loss: L1 in frequency domain (`FFTLoss` in `src/utils/loss_utils.py`), weighted by `--fft_loss_weight`
- Balance loss: auxiliary MoE routing losses (importance imbalance + load imbalance), weighted by `--balance_loss_weight`

## Data

Datasets live under `--data_file_dir` (default `~/datasets`). Expected layout:

| Task | Path |
|------|------|
| Denoising (train) | `<dir>/denoising/WaterlooED/`, `<dir>/denoising/BSD400/` |
| Denoising (test) | `<dir>/denoising/cBSD68/original_png/` |
| Dehazing | `<dir>/dehazing/RESIDE/`, SOTS at `<dir>/dehazing/SOTS/` |
| Deraining | `<dir>/deraining/RainTrainL/`, `<dir>/deraining/Rain100L/` |
| Deblurring | `<dir>/deblurring/GoPro/` |
| Low-light | `<dir>/llie/LOLv1/` |
| CDD11 | `<dir>/cdd11/` |

Training datasets use augmentation: random crop (`--patch_size`, default 128), random horizontal/vertical flips, random 90° rotations. Some datasets are repeated to balance class distribution (e.g., derain ×120, deblur ×5, synllie ×20).

## Key CLI arguments

| Arg | Purpose |
|-----|---------|
| `--model` | `MoCE_IR` or `MoCE_IR_S` (required) |
| `--de_type` | List of degradation types: `denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie` |
| `--trainset` | `standard` or `CDD11_<subset>` (single/double/triple/all) |
| `--with_complexity` | Enable complexity-biased expert routing |
| `--complexity_scale` | `max` or `min` — normalize complexity by max or min |
| `--rank_type` | Expert rank pattern: `spread` (dim//2^i reversed), `constant`, `lin`, `double`, `exp`, `fact` |
| `--depth_type` | Expert depth pattern: `constant`, `lin`, `double`, `exp`, `fact`, or an integer |
| `--stage_depth` | Base depth per decoder stage, e.g. `[1, 1, 1]` |
| `--topk` | Number of experts activated per token (default 1) |
| `--data_file_dir` | Root path to datasets (default `~/datasets`) |
| `--output_path` | Root for logs/results (default `output/` → CWD) |
