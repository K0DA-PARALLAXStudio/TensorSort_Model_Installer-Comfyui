<div align="center">

```
================================================================================
                    COMFYUI-TENSORSORT MODEL INSTALLER
================================================================================

                 "Because filenames lie, but tensors don't."
```

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)

**Intelligent model file analysis, sorting, and naming for ComfyUI**

[Features](#features) • [Quick Start](#quick-start) • [Modules](#the-15-modules) • [Documentation](#documentation)

---

**Author:** K0DA Parallax Studio

</div>

---

## What is TensorSort?

TensorSort analyzes your model files by reading the actual **tensor data** inside - not just the filename.

It detects:
- **File type**: Checkpoint, LoRA, VAE, ControlNet, Embedding, Upscaler, IP-Adapter, and more
- **Base model**: Flux, SDXL, SD 1.5, Pony, Illustrious, Z-Image
- **Precision**: FP32, FP16, BF16, FP8
- **Components**: Full, NoVAE, NoCLIP, UNET-only
- **GGUF quantization**: Q2_K through Q8_0

Then it:
1. **Sorts** files into the correct ComfyUI folders
2. **Renames** them with semantic, consistent naming conventions
3. **Detects duplicates** via SHA256 hash comparison

---

## The Problem

Filenames on CivitAI are often... creative:

```
aDetailedPurposeV2_v14e.safetensors
```

What is this? Checkpoint? LoRA? SDXL or SD 1.5? No idea.

Even if you know what it is - **do you know where it goes?**

```
Checkpoints      → checkpoints/
BUT GGUF models  → unet/           (or ComfyUI won't load them!)
LoRAs            → loras/
BUT LyCORIS      → loras/LyCORIS/
VAEs             → vae/
BUT TAESD        → vae_approx/
ControlNets      → controlnet/
BUT T2I-Adapter  → t2i_adapter/
IP-Adapters      → ???             (THREE different folders!)
```

**TensorSort handles all of this automatically.**

---

## Features

### Two Modes

**Mode A - Installation**
```
New downloads → Analyze → Sort → Rename
```

**Mode B - Cleanup**
```
Existing models/ folder → Find problems → Fix
```

### Preview Before Action

Nothing happens without your confirmation:

```
[PREVIEW] mysterious_model_v3.safetensors (12.4 GB)
     Detected: FluxD, Full, FP16
     → checkpoints/FluxD_Full-FP16_General_MysteriousModel_v3.safetensors

[PREVIEW] some_lora.safetensors (184 MB)
     Detected: SDXL, LoRA, Style
     → loras/SDXL_Style_SomeLora_v1.safetensors

Total: 2 files (12.6 GB)
Continue? [y/n]
```

### Semantic Naming

**Before:**
```
ponyDiffusionV6XL_v6StartWithThisOne.safetensors
```

**After:**
```
Pony_Full-FP16_Pony_V6.safetensors
```

At a glance: Base model, components, precision, version.

---

## Quick Start

### Requirements

- Python 3.10+
- ComfyUI installed

### Installation

```bash
git clone https://github.com/K0DA-Parallax-Studio/ComfyUI-TensorSort.git
cd ComfyUI-TensorSort
```

No additional dependencies required.

### Usage

**Windows:**
```
TensorSort_Model_Installer.bat
```

**Python:**
```bash
python all_modules.py
```

### Typical Workflow

**Installing new models:**
1. Download models from CivitAI/HuggingFace
2. Put them in `downloads/` folder
3. Run TensorSort → Choose Mode A
4. Check preview
5. Confirm
6. Done!

**Cleaning up existing collection:**
1. Run TensorSort → Choose Mode B
2. Choose module (or "All")
3. Review problems found
4. Confirm fixes
5. Done!

---

## The 15 Modules

| # | Module | ComfyUI Folder | Handles |
|---|--------|----------------|---------|
| 1 | Base Models | `checkpoints/`, `unet/` | Checkpoints, GGUF |
| 2 | VAE | `vae/`, `vae_approx/` | VAE, TAESD |
| 3 | Text Encoders | `clip/`, `text_encoders/` | CLIP, T5 |
| 4 | LoRAs | `loras/`, `loras/LyCORIS/` | LoRA, LyCORIS |
| 5 | ControlNet | `controlnet/`, `t2i_adapter/` | ControlNet, T2I |
| 6 | Upscalers | `upscale_models/` | ESRGAN, SwinIR, etc. |
| 7 | Embeddings | `embeddings/` | Textual Inversion |
| 8 | PhotoMaker | `photomaker/` | PhotoMaker v1/v2 |
| 9 | InsightFace | `insightface/` | ONNX face models |
| 10 | IP-Adapter | `ipadapter/`, `ipadapter-flux/`, `xlabs/` | 3 formats! |
| 11 | AnimateDiff | `animatediff_models/` | Motion modules |
| 12 | SAM | `sams/` | Segment Anything |
| 13 | Grounding DINO | `grounding-dino/` | Open-set detection |
| 14 | YOLO | `ultralytics/bbox/`, `ultralytics/segm/` | Object detection |
| 15 | VLM & LLM | `VLM/`, `LLM/` | Vision-language models |

---

## How It Works

### Tensor-Based Detection

TensorSort reads the actual tensor keys inside files:

```
"double_blocks.*"        → Flux architecture
"conditioner.embedders"  → SDXL
"lora_unet.*"           → LoRA
"controlnet_*"          → ControlNet
```

This works regardless of filename.

### Cross-Module Rescue

Found a VAE in `checkpoints/`? TensorSort's 2-pass system handles it:

1. **Pass 1**: Module 1 scans checkpoints, finds VAE → adds to queue
2. **Pass 2**: Module 2 reads queue → rescues file to `vae/`

No files get lost.

### Duplicate Detection

SHA256 hash comparison catches duplicates even with different names:

```
downloads/awesome_lora.safetensors
loras/SDXL_Style_Awesome_v1.safetensors
→ Same content? Duplicate detected, skipped.
```

---

## Important Notes

### GGUF Goes in unet/

GGUF-quantized models **must** go in `unet/`, not `checkpoints/`.
ComfyUI won't load them otherwise. TensorSort handles this automatically.

### IP-Adapter: 3 Folders

Flux IP-Adapters exist in incompatible formats:

| Format | Folder | Size |
|--------|--------|------|
| SD 1.5 / SDXL | `ipadapter/` | 22-900 MB |
| Flux InstantX | `ipadapter-flux/` | ~5 GB |
| Flux XLabs | `xlabs/ipadapters/` | ~936 MB |

Wrong folder = crash. TensorSort detects the format automatically.

### T5-XXL VRAM Warning

Flux needs T5-XXL: **9.2 GB VRAM** in FP16.
Use FP8 variants if VRAM is tight.

### After Changes: Hard Refresh Required!

After TensorSort renames/moves files, a ComfyUI restart is **not enough**.
The browser caches old filenames.

**Solution:** `Ctrl + Shift + R` (Windows/Linux) or `Cmd + Shift + R` (Mac)

---

## Documentation

**Quick Start Guides** (Recommended for new users):

| Language | File |
|----------|------|
| English | [`README_EN.txt`](README_EN.txt) - Quick guide (~10 min read) |
| German | [`README_DE.txt`](README_DE.txt) - Schnellanleitung (~10 min Lesezeit) |

**Complete Manuals** (All 15 modules explained):

| Language | File |
|----------|------|
| English | [`MANUAL_EN.txt`](MANUAL_EN.txt) - Full documentation (~1300 lines) |
| German | [`MANUAL_DE.txt`](MANUAL_DE.txt) - Vollständige Dokumentation (~1300 Zeilen) |

---

## Naming Examples

**Checkpoints:**
```
Before: aDetailedPurposeV2_v14e.safetensors
After:  SDXL_Full-FP16_Realism_DetailedPurpose_v14.safetensors
```

**LoRAs:**
```
Before: add_detail_xl_v2.safetensors
After:  SDXL_Enhancement_Add-Detail_v2.safetensors
```

**GGUF:**
```
Before: flux1-dev-Q4_K_M.gguf
After:  FluxD_GGUF-Q4_K_M_Dev.gguf
```

**ControlNet:**
```
Before: control_v11p_sd15_openpose_fp16.safetensors
After:  SD15_CN-OpenPose_FP16_v1.1.safetensors
```

---

## Support

**Questions? Problems? Feedback?**

- **Email:** kodaparallax@gmail.com
- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Author:** K0DA Parallax Studio

We're happy to help with technical issues, unrecognized files, or any questions you might have!

---

## Contributing

Issues and PRs welcome!

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Credits

**Author:** K0DA Parallax Studio

**Built for the ComfyUI community.**

**References:**
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Stability AI](https://stability.ai/)
- [Black Forest Labs (Flux)](https://blackforestlabs.ai/)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)

---

<div align="center">

```
================================================================================
          TensorSort - Because filenames lie, but tensors don't.
================================================================================
```

**K0DA Parallax Studio** | 2024

</div>
