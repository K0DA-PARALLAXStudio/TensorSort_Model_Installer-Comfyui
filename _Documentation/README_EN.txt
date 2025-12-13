================================================================================
                    COMFYUI-TENSORSORT MODEL INSTALLER
                       Quick Start Guide (EN) - v1.2.0
================================================================================

                    "Because filenames lie, but tensors don't."

Welcome to TensorSort! This guide will get you up and running in 10 minutes.


================================================================================
1. WHAT IS TENSORSORT?
================================================================================

TensorSort is an intelligent sorting and naming tool for ComfyUI model files.
It analyzes the CONTENTS of each file (the tensor structures), not just the
filename.

What it detects:
  - File type (Checkpoint, LoRA, VAE, ControlNet, ...)
  - Base model (Flux, SDXL, SD 1.5, Pony, ...)
  - Precision (FP32, FP16, BF16, FP8, GGUF quantization)
  - Components (Full, NoVAE, NoCLIP, UNET-only)

What it does:
  1. Sorts the file into the correct ComfyUI folder
  2. Renames using a consistent naming convention
  3. Detects and handles duplicates (via SHA256 hash)

The result: A clean, searchable model library.


================================================================================
2. WHO IS THIS FOR?
================================================================================

BEGINNERS:
  - Getting "Error loading LoRA" and don't know why?
  - TensorSort shows you what each file is and where it belongs
  - You learn about different file types along the way

COLLECTORS:
  - Downloads folder overflowing (50, 100, 200+ files)?
  - Duplicates hidden by different filenames?
  - TensorSort brings order, eliminates duplicates

POWER USERS:
  - 400+ models, organically grown, wildly named?
  - Scrolling forever through unsorted lists in the LoRA loader?
  - TensorSort gives you clean structure, visible at a glance


================================================================================
3. QUICK START - YOUR FIRST RUN
================================================================================

STEP 1 - START THE PROGRAM:

  Double-click: TensorSort_Model_Installer.bat

  → Main menu appears

STEP 2 - INSTALL NEW MODELS:

  1. Place your downloaded files in the downloads/ folder

  2. In the menu select:
     [1] INSTALL NEW MODELS - Process downloads/ folder

  3. Choose:
     [1] BATCH PROCESSING (All Types) - RECOMMENDED

     (This processes all 15 module types sequentially)

  4. TensorSort shows you a PREVIEW list:
     - Which files were detected
     - What each file is (Base Model, Type, Precision)
     - Where it will be moved
     - What the new name will be

  5. You decide:
     [1] Keep originals in downloads/
     [2] Delete originals (saves disk space)
     [0] Cancel

  6. Final confirmation:
     [1] Start installation
     [0] Cancel

  7. Installation runs → Summary is displayed

DONE! Your models are now sorted and correctly named.

IMPORTANT - AFTER INSTALLATION:
  → In ComfyUI: Press Ctrl + Shift + R (Hard Refresh)
  → Otherwise ComfyUI still shows old filenames (browser cache!)


================================================================================
4. THE TWO MODES
================================================================================

MODE A - INSTALLATION (New files):

  You have downloaded new files and want to install them.

  Workflow:
    1. Put files in downloads/
    2. TensorSort → Mode A → Batch Processing
    3. Review preview → Confirm → Done

MODE B - CHECK & FIX (Existing files):

  Your models/ folder is already full but disorganized.

  Workflow:
    1. TensorSort → Mode B
    2. Scans all existing files
    3. Finds: Wrong folders, wrong names, duplicates
    4. Shows what would be changed
    5. You confirm → Problems are fixed


================================================================================
5. THE 15 MODULES (OVERVIEW)
================================================================================

Each module handles a specific file type:

 Nr | Module              | ComfyUI Folder                    | What it handles
----|---------------------|-----------------------------------|------------------
  1 | Base Models         | checkpoints/, unet/               | SD Checkpoints, GGUF
  2 | VAE                 | vae/                              | VAE, TAESD
  3 | Text Encoders       | clip/, text_encoders/             | CLIP, T5, BERT
  4 | LoRAs               | loras/, loras/LyCORIS/            | LoRA, LyCORIS
  5 | ControlNet          | controlnet/, t2i_adapter/         | ControlNet, T2I
  6 | Upscalers           | upscale_models/                   | ESRGAN, etc.
  7 | Embeddings          | embeddings/                       | Textual Inversion
  8 | PhotoMaker          | photomaker/                       | PhotoMaker v1/v2
  9 | InsightFace         | insightface/                      | Face Swap Models
 10 | IP-Adapter          | ipadapter/, ipadapter-flux/       | Image Prompting
 11 | AnimateDiff         | animatediff_models/               | Motion Modules
 12 | SAM                 | sams/                             | Segment Anything
 13 | Grounding DINO      | grounding-dino/                   | Object Detection
 14 | YOLO                | ultralytics/bbox/, /segm/         | YOLO Detection
 15 | VLM & LLM           | VLM/, LLM/                        | Vision-Language

→ For details on each module see: MANUAL_EN.txt


================================================================================
6. NAMING CONVENTION - WHAT DO THE NAMES MEAN?
================================================================================

TensorSort renames files following this pattern:

{BaseModel}_{Component}_{Precision}_{Category}_{Name}_{Version}.ext

EXAMPLES:

FluxD_Full-FP16_NSFW_Persephone_v10.safetensors
  ↑      ↑         ↑     ↑           ↑
  │      │         │     │           └─ Version (v10)
  │      │         │     └───────────── Name (Persephone)
  │      │         └─────────────────── Category (NSFW)
  │      └───────────────────────────── Components + Precision (Full-FP16)
  └──────────────────────────────────── Base Model (FluxD = Flux Dev)

SDXL_VAE-FP16_v1.safetensors
  ↑    ↑
  │    └─ VAE in FP16 precision
  └────── Trained for SDXL

FluxD_Style-LoRA_Watercolor_v2.safetensors
  ↑     ↑            ↑
  │     │            └─ Name (Watercolor)
  │     └────────────── LoRA type (Style)
  └──────────────────── For Flux Dev

WHY NAMED LIKE THIS?

  ✓ Sortable: All Flux models appear together
  ✓ Searchable: "FP16" finds all FP16 models
  ✓ Understandable: You see immediately what the file is
  ✓ Consistent: Always the same structure


================================================================================
7. COMMON PROBLEMS & SOLUTIONS
================================================================================

PROBLEM: "ComfyUI doesn't show new filenames"
SOLUTION:
  → Press Ctrl + Shift + R in browser (Hard Refresh)
  → This clears the cache and reloads the UI
  → IMPORTANT: Regular refresh (F5) is NOT enough!


PROBLEM: "Error loading LoRA"
CAUSE:
  → LoRA is for wrong base model (e.g. Flux LoRA with SDXL checkpoint)
SOLUTION:
  → Check the prefix in the filename:
     FluxD_ = Only use with Flux Dev
     SDXL_  = Only use with SDXL
     SD15_  = Only use with Stable Diffusion 1.5


PROBLEM: "Model not found"
CAUSE:
  → File is in the wrong folder
SOLUTION:
  → Run Mode B: [2] CHECK & FIX MODELS
  → Shows all misplaced files
  → Moves them automatically


PROBLEM: "GGUF checkpoint won't load in CheckpointLoader"
CAUSE:
  → GGUF files must be in unet/, NOT in checkpoints/
  → This is a ComfyUI limitation, not a TensorSort issue
SOLUTION:
  → TensorSort automatically sorts GGUF to unet/
  → In ComfyUI: Use "Load Diffusion Model" node (not CheckpointLoader)


PROBLEM: "Out of Memory (OOM) when loading"
CAUSE:
  → Model too large for your VRAM
  → FP16 Flux Dev = ~24 GB VRAM
  → T5-XXL Encoder = ~9.2 GB VRAM (!!!)
SOLUTION:
  → Use GGUF-quantized model:
     Q8_0 = ~12 GB (minimal quality loss)
     Q4_K_M = ~6 GB (slight quality loss)
  → Or: Skip T5-XXL (use only CLIP)


PROBLEM: "IP-Adapter not found"
CAUSE:
  → IP-Adapter comes in 3 different formats
  → Each format has its own folder
SOLUTION:
  → TensorSort automatically detects:
     SDXL/SD15 → ipadapter/
     Flux (InstantX) → ipadapter-flux/
     Flux (XLabs) → xlabs/ipadapters/
  → Check which node you're using and which format you need


PROBLEM: "File was skipped (SKIP)"
CAUSE:
  → TensorSort didn't recognize the file
  → Can happen with:
     - Exotic/custom formats
     - Corrupted downloads
     - Non-ML files (.txt, .json, etc.)
SOLUTION:
  → Check if download completed successfully
  → If file is important: Contact support (see below)
  → Unrecognized files stay in downloads/ (safe!)


================================================================================
8. FURTHER INFORMATION
================================================================================

COMPLETE MANUAL:

  This quick-start guide covers ~20% of the features.

  For complete information see:
    → MANUAL_EN.txt (~1300 lines)

  Contains:
    - Detailed explanation of all 15 modules
    - Technical details (Detection hierarchy, Queue system)
    - Naming conventions for all module types
    - Advanced troubleshooting tips
    - References and sources


GERMAN VERSION:

  → README_DE.txt (Diese Anleitung auf Deutsch)
  → MANUAL_DE.txt (Vollständiges Manual auf Deutsch)


================================================================================
9. SUPPORT & CONTACT
================================================================================

QUESTIONS? PROBLEMS? FEEDBACK?

Author: K0DA Parallax Studio
Email:  kodaparallax@gmail.com

We're happy to help with:
  - Technical issues
  - Unrecognized files
  - Feature requests
  - Bug reports

Please include in your email:
  - Which module (1-15)
  - Which mode (A or B)
  - Error message (if any)
  - Filename and size


================================================================================
                              GOOD LUCK!
================================================================================

TensorSort v1.2.0
"Because filenames lie, but tensors don't."

Author: K0DA Parallax Studio
Email:  kodaparallax@gmail.com

================================================================================
