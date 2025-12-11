"""
Modul 3 - CLIP & Text Encoders

Verarbeitet Text-Encoder und Vision-Encoder:
- models/clip/ (CLIP-L, CLIP-G)
- models/text_encoders/ (T5-XXL, BERT, etc.)
- models/clip_vision/ (ViT-H, ViT-L, ViT-bigG)

Modus A: Installation aus downloads/
Modus B: Reinstall/Check bestehender Encoder
"""

import struct
import json
import os
import sys
from pathlib import Path
import argparse

# ============================================================================
# PATH SETUP - Für Unterordner-Struktur
# ============================================================================
_SCRIPT_DIR = Path(__file__).parent
_SHARED_DIR = _SCRIPT_DIR.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

# Import shared utilities
from shared_utils import (
    handle_duplicate_move,
    DOWNLOADS_DIR as DOWNLOADS_PATH,
    CLIP_DIR,
    TEXT_ENCODERS_DIR,
    CLIP_VISION_DIR,
    read_misplaced_files,
    add_misplaced_file,
    remove_misplaced_file,
    ask_keep_or_delete,
    ask_confirm_installation,
    # Output Helpers (Mode A)
    Colors,
    print_mode_a_header,
    print_no_files_found,
    print_analysis,
    print_skipped_section,
    print_preview_header,
    print_preview_item,
    print_total,
    print_no_files_to_install,
    print_installation_header,
    print_install_item,
    print_summary,
    # Output Helpers (Mode B)
    print_mode_b_header,
    print_analysis_b,
    print_already_correct_section,
    print_problems_header,
    print_problem_item,
    print_fix_result
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (from shared_utils.py)
DOWNLOADS_DIR = DOWNLOADS_PATH

# Size limits (with 20% buffer)
SIZE_MIN = 188 * 1024 * 1024  # 188 MB
SIZE_MAX = 12 * 1024 * 1024 * 1024  # 12 GB

# ============================================================================
# SAFETENSORS READING
# ============================================================================

def read_safetensors_keys(file_path):
    """Liest Keys aus Safetensors Datei"""
    try:
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            metadata_bytes = f.read(header_size)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            keys = [k for k in metadata.keys() if k != "__metadata__"]
            return keys, metadata
    except Exception as e:
        print(f"[ERROR] Kann Keys nicht lesen: {e}")
        return [], {}

# ============================================================================
# MODULE BOUNDARY CHECKS
# ============================================================================

def has_unet_keys(keys):
    """Prüft ob UNET Keys vorhanden (Base Model)"""
    unet_patterns = [
        "model.diffusion_model", "down_blocks", "up_blocks", "mid_block",
        "double_blocks", "single_blocks", "input_blocks", "output_blocks"
    ]
    return any(any(p in k for p in unet_patterns) for k in keys)

def has_vae_keys(keys):
    """Prüft ob VAE Keys vorhanden (ohne Vision Model)"""
    vae_patterns = ["first_stage_model", "decoder.conv_in", "encoder.conv_in"]
    return any(any(p in k for p in vae_patterns) for k in keys)

def has_lora_keys(keys):
    """Prüft ob LoRA Keys vorhanden"""
    lora_patterns = ["lora_", ".alpha", "lora.down", "lora.up"]
    return any(any(p in k for p in lora_patterns) for k in keys)

def has_vision_model_keys(keys):
    """Prüft ob Vision Model Keys vorhanden"""
    return any("vision_model" in k or "visual.transformer" in k for k in keys)

# ============================================================================
# ENCODER TYPE DETECTION
# ============================================================================

def detect_encoder_type(keys):
    """Unterscheidet: clip / text_encoder / clip_vision

    Returns:
        str: "clip", "text_encoder", "clip_vision", oder None
    """
    # 1. CLIP Vision (Image Encoder)
    if any("vision_model" in k or "visual.transformer" in k for k in keys):
        return "clip_vision"

    # 2. T5/BERT (Non-CLIP Text Encoder)
    if any("encoder.block" in k or "decoder.block" in k for k in keys):
        return "text_encoder"

    # 3. Standard CLIP (Text Encoder)
    if any("text_model.encoder" in k or "clip_l" in k or "clip_g" in k for k in keys):
        return "clip"

    return None

def is_encoder(file_path):
    """Prüft ob Datei ein Encoder ist (kein Base Model, kein LoRA)

    Returns:
        tuple: (bool, str, dict) - (is_encoder, reason/type, details)
    """
    keys, metadata = read_safetensors_keys(file_path)

    if not keys:
        return False, "SKIP: Kann Keys nicht lesen", {}

    # MUSS NICHT HABEN: UNET
    if has_unet_keys(keys):
        return False, "SKIP: Hat UNET (Base Model - Modul 1)", {}

    # MUSS NICHT HABEN: VAE (außer bei Vision Model)
    if has_vae_keys(keys) and not has_vision_model_keys(keys):
        return False, "SKIP: Hat VAE (Modul 2)", {}

    # MUSS NICHT HABEN: LoRA
    if has_lora_keys(keys):
        return False, "SKIP: Hat LoRA (Modul 4)", {}

    # Welcher Encoder-Typ?
    encoder_type = detect_encoder_type(keys)

    if encoder_type:
        return True, encoder_type, {"keys": keys, "metadata": metadata}
    else:
        return False, "SKIP: Keine Encoder-Keys", {}

# ============================================================================
# CLIP TEXT ENCODER DETECTION
# ============================================================================

def detect_clip_size(keys, filename):
    """Erkennt CLIP Size (L, G)"""
    # Aus Keys
    if any("clip_l" in k.lower() for k in keys):
        return "L"
    elif any("clip_g" in k.lower() for k in keys):
        return "G"

    # Aus Filename
    filename_lower = filename.lower()
    if "clip_l" in filename_lower or "clip-l" in filename_lower:
        return "L"
    elif "clip_g" in filename_lower or "clip-g" in filename_lower:
        return "G"

    # Aus Layer-Count (Fallback)
    encoder_layers = [k for k in keys if "encoder.layers." in k]
    if encoder_layers:
        layer_nums = []
        for k in encoder_layers:
            parts = k.split("encoder.layers.")
            if len(parts) > 1:
                num_part = parts[1].split(".")[0]
                if num_part.isdigit():
                    layer_nums.append(int(num_part))

        if layer_nums:
            max_layer = max(layer_nums)
            if max_layer <= 12:
                return "L"
            elif max_layer <= 24:
                return "G"

    return "Unknown"

def detect_clip_details(file_path, keys, metadata):
    """Analysiert CLIP Text Encoder Details

    Returns:
        dict: {
            "clip_size": "L" oder "G",
            "base_model": None (für L) oder "SDXL" (für G),
            "precision": "FP16" etc.
        }
    """
    filename = os.path.basename(file_path)

    # Size
    clip_size = detect_clip_size(keys, filename)

    # Base Model Tag (nur für CLIP-G)
    base_model = "SDXL" if clip_size == "G" else None

    # Precision (meist FP16, nicht kritisch)
    precision = detect_precision(filename, metadata)

    return {
        "clip_size": clip_size,
        "base_model": base_model,
        "precision": precision
    }

# ============================================================================
# TEXT ENCODER DETECTION
# ============================================================================

def detect_text_encoder_type(keys, filename):
    """Erkennt Text Encoder Typ (T5-XXL, BERT, etc.)"""
    # T5
    if any("encoder.block" in k or "decoder.block" in k for k in keys):
        # T5-XXL vs T5-Large aus Größe oder Keys
        return "T5-XXL"

    # BERT
    if any("bert.encoder" in k for k in keys):
        return "BERT"

    return "Unknown"

def detect_text_encoder_variant(filename):
    """Erkennt Variante (Scaled, etc.)"""
    filename_lower = filename.lower()

    if "scaled" in filename_lower:
        return "Scaled"

    return None

def detect_text_encoder_details(file_path, keys, metadata):
    """Analysiert Text Encoder Details

    Returns:
        dict: {
            "encoder_type": "T5-XXL",
            "precision": "FP16",
            "base_model": "Flux",
            "variant": "Scaled" oder None
        }
    """
    filename = os.path.basename(file_path)

    # Type
    encoder_type = detect_text_encoder_type(keys, filename)

    # Precision (KRITISCH!)
    precision = detect_precision(filename, metadata)

    # Base Model (T5-XXL nur für Flux)
    base_model = "Flux" if encoder_type == "T5-XXL" else None

    # Variant
    variant = detect_text_encoder_variant(filename)

    return {
        "encoder_type": encoder_type,
        "precision": precision,
        "base_model": base_model,
        "variant": variant
    }

# ============================================================================
# CLIP VISION DETECTION
# ============================================================================

def detect_clipvision_architecture(filename):
    """Erkennt CLIP Vision Architecture + Size"""
    filename_lower = filename.lower()

    if "bigg" in filename_lower:
        return "ViT-bigG"
    elif "vit-h" in filename_lower or "vit_h" in filename_lower:
        return "ViT-H"
    elif "vit-g" in filename_lower or "vit_g" in filename_lower or "vitg" in filename_lower:
        return "ViT-G"
    elif "vit-l" in filename_lower or "vit_l" in filename_lower or "large" in filename_lower:
        return "ViT-L"
    elif "vit-b" in filename_lower or "vit_b" in filename_lower or "base" in filename_lower:
        return "ViT-B"

    return "Unknown"

def detect_patch_size(filename):
    """Erkennt Patch Size"""
    filename_lower = filename.lower()

    if "patch14" in filename_lower or "-14-" in filename_lower or "_14_" in filename_lower or "-14" in filename_lower:
        return "14"
    elif "patch16" in filename_lower or "-16-" in filename_lower or "_16_" in filename_lower or "-16" in filename_lower:
        return "16"
    elif "patch32" in filename_lower or "-32-" in filename_lower or "_32_" in filename_lower or "-32" in filename_lower:
        return "32"

    return "Unknown"

def detect_dataset(filename):
    """Erkennt Dataset/Source"""
    filename_lower = filename.lower()

    if "laion2b" in filename_lower or "laion-2b" in filename_lower:
        return "LAION2B"
    elif "laion400m" in filename_lower:
        return "LAION400M"
    elif "datacomp" in filename_lower:
        return "DataComp"
    elif "openai" in filename_lower:
        return "OpenAI"

    # Fallback
    return "OpenAI"

def detect_clipvision_details(file_path, keys, metadata):
    """Analysiert CLIP Vision Details

    Returns:
        dict: {
            "architecture": "ViT-H",
            "patch": "14",
            "dataset": "LAION2B"
        }
    """
    filename = os.path.basename(file_path)

    architecture = detect_clipvision_architecture(filename)
    patch = detect_patch_size(filename)
    dataset = detect_dataset(filename)

    return {
        "architecture": architecture,
        "patch": patch,
        "dataset": dataset
    }

# ============================================================================
# PRECISION DETECTION
# ============================================================================

def detect_precision(filename, metadata):
    """Erkennt Precision (FP16, FP8, FP32, BF16)"""
    # Aus Filename (Priorität 1)
    filename_lower = filename.lower()

    if "fp32" in filename_lower:
        return "FP32"
    elif "fp16" in filename_lower:
        return "FP16"
    elif "fp8" in filename_lower:
        return "FP8"
    elif "bf16" in filename_lower:
        return "BF16"

    # Aus Metadata (Fallback)
    if "__metadata__" in metadata:
        # ... (meist leer bei Encodern)
        pass

    # Default
    return "FP16"

# ============================================================================
# NAME GENERATION
# ============================================================================

def generate_clip_name(details):
    """Generiert Namen für CLIP Text Encoder

    Format:
        CLIP-{Size}.safetensors (universal)
        CLIP-{Size}_{BaseModel}.safetensors (spezifisch)
    """
    size = details["clip_size"]
    base_model = details["base_model"]

    if base_model:
        return f"CLIP-{size}_{base_model}.safetensors"
    else:
        return f"CLIP-{size}.safetensors"

def generate_text_encoder_name(details):
    """Generiert Namen für Text Encoder

    Format:
        {Type}_{Precision}_{BaseModel}.safetensors
        {Type}_{Precision}_{BaseModel}_{Variant}.safetensors
    """
    encoder_type = details["encoder_type"]
    precision = details["precision"]
    base_model = details["base_model"]
    variant = details["variant"]

    parts = [encoder_type, precision]

    if base_model:
        parts.append(base_model)

    if variant:
        parts.append(variant)

    return "_".join(parts) + ".safetensors"

def generate_clipvision_name(details):
    """Generiert Namen für CLIP Vision

    Format:
        CLIPVision_{Architecture}-{Patch}_{Dataset}.safetensors
    """
    arch = details["architecture"]
    patch = details["patch"]
    dataset = details["dataset"]

    name_parts = [f"CLIPVision_{arch}-{patch}"]

    if dataset and dataset != "Unknown":
        name_parts.append(dataset)

    return "_".join(name_parts) + ".safetensors"

def generate_encoder_name(encoder_type, details):
    """Generiert Namen basierend auf Encoder-Typ"""
    if encoder_type == "clip":
        return generate_clip_name(details)
    elif encoder_type == "text_encoder":
        return generate_text_encoder_name(details)
    elif encoder_type == "clip_vision":
        return generate_clipvision_name(details)
    else:
        return None

# ============================================================================
# TARGET FOLDER ASSIGNMENT
# ============================================================================

def get_target_folder(encoder_type):
    """Gibt Target-Ordner zurück"""
    if encoder_type == "clip":
        return CLIP_DIR
    elif encoder_type == "text_encoder":
        return TEXT_ENCODERS_DIR
    elif encoder_type == "clip_vision":
        return CLIP_VISION_DIR
    else:
        return None

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def check_duplicate(target_path):
    """Prüft ob Datei bereits existiert (Size + Hash)"""
    if not os.path.exists(target_path):
        return False, None

    # Für jetzt: Simple existiert-Prüfung
    # TODO: Implement Size + Hash check wie in shared_utils.py
    return True, target_path

# ============================================================================
# EXPORTABLE FUNCTIONS (for all_modules.py batch processing)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads and return file list (for batch processing)"""
    files_to_install = []
    skipped = []

    safetensors_files = list(downloads_path.glob("*.safetensors"))

    for file_path in safetensors_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024

        file_size = file_path.stat().st_size
        if file_size < SIZE_MIN or file_size > SIZE_MAX:
            skipped.append({'filename': filename, 'reason': 'size out of range', 'size_gb': size_gb})
            continue

        is_enc, reason, details = is_encoder(file_path)

        if not is_enc:
            skipped.append({'filename': filename, 'reason': reason, 'size_gb': size_gb})
            continue

        encoder_type = reason
        keys = details["keys"]
        metadata = details["metadata"]

        # Detect details
        if encoder_type == "clip":
            enc_details = detect_clip_details(file_path, keys, metadata)
        elif encoder_type == "text_encoder":
            enc_details = detect_text_encoder_details(file_path, keys, metadata)
        elif encoder_type == "clip_vision":
            enc_details = detect_clipvision_details(file_path, keys, metadata)

        new_name = generate_encoder_name(encoder_type, enc_details)
        target_folder = get_target_folder(encoder_type)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'size_gb': size_gb,
            'new_name': new_name,
            'target_folder': target_folder,
            'encoder_type': encoder_type,
            'details': enc_details
        })

    return {
        'module_name': 'CLIP & Encoders',
        'files': files_to_install,
        'skipped': skipped
    }

# ============================================================================
# MODUS A - INSTALLATION (Standalone mode)
# ============================================================================

def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="TEXT ENCODERS (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.safetensors",
        module_type="Text Encoders",
        target_folders="clip/, text_encoders/, clip_vision/"
    )

    # Scan downloads/
    if not DOWNLOADS_DIR.exists():
        print(f"[ERROR] Downloads-Ordner nicht gefunden: {DOWNLOADS_DIR}")
        return

    safetensors_files = list(DOWNLOADS_DIR.glob("*.safetensors"))

    if not safetensors_files:
        print_no_files_found("text encoder files")
        return

    # ========================================================================
    # PHASE 1: PREVIEW - Analyze all files (silently collect)
    # ========================================================================
    files_to_install = []
    skipped = []

    for file_path in safetensors_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024

        # Size check (Fast-Skip)
        file_size = file_path.stat().st_size
        if file_size < SIZE_MIN or file_size > SIZE_MAX:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # Module Boundary Check
        is_enc, reason, details = is_encoder(file_path)

        if not is_enc:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # reason ist jetzt encoder_type: "clip", "text_encoder", "clip_vision"
        encoder_type = reason
        keys = details["keys"]
        metadata = details["metadata"]

        # Detect details
        if encoder_type == "clip":
            enc_details = detect_clip_details(file_path, keys, metadata)
            detected_str = f"CLIP-{enc_details['clip_size']}, {enc_details['precision']}"
        elif encoder_type == "text_encoder":
            enc_details = detect_text_encoder_details(file_path, keys, metadata)
            detected_str = f"{enc_details['encoder_type']}, {enc_details['precision']}"
        elif encoder_type == "clip_vision":
            enc_details = detect_clipvision_details(file_path, keys, metadata)
            detected_str = f"{enc_details['architecture']}-{enc_details['patch']}, {enc_details['dataset']}"

        # Generate name
        new_name = generate_encoder_name(encoder_type, enc_details)

        # Target folder
        target_folder = get_target_folder(encoder_type)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'new_name': new_name,
            'target_folder': target_folder,
            'detected_str': detected_str,
            'size_mb': size_mb,
            'size_gb': size_gb
        })

    # ========================================================================
    # ANALYSIS (using shared helper with colors)
    # ========================================================================
    print_analysis(len(safetensors_files), len(files_to_install), len(skipped))

    # ========================================================================
    # SKIPPED SECTION (using shared helper)
    # ========================================================================
    print_skipped_section(
        skipped_files=skipped,
        skip_reason="Not a text encoder (no CLIP/T5/BERT keys found)",
        max_show=5
    )

    # ========================================================================
    # PREVIEW SECTION (using shared helpers)
    # ========================================================================
    if not files_to_install:
        print_no_files_to_install()
        return

    print_preview_header(len(files_to_install))

    for i, file_info in enumerate(files_to_install, 1):
        print_preview_item(
            index=i,
            filename=file_info['filename'],
            size_mb=file_info['size_mb'],
            detected_info=file_info['detected_str'],
            target_path=f"{file_info['target_folder'].name}/{file_info['new_name']}"
        )

    # Calculate total size
    total_size = sum(f['size_gb'] for f in files_to_install)

    # TOTAL (using shared helper)
    print_total(len(files_to_install), total_size)

    # ========================================================================
    # PHASE 2: ASK DELETE/KEEP (after preview!)
    # ========================================================================
    keep_source = ask_keep_or_delete(total_size)
    if keep_source is None:
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Installation cancelled.")
        sys.exit(1)

    # ========================================================================
    # PHASE 3: CONFIRMATION
    # ========================================================================
    if not ask_confirm_installation():
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Installation cancelled.")
        sys.exit(1)

    # ========================================================================
    # PHASE 4: INSTALLATION (using shared helpers)
    # ========================================================================
    print_installation_header()

    installed = 0
    errors = 0
    collisions = 0

    for idx, file_info in enumerate(files_to_install, 1):
        file_path = file_info['path']
        new_name = file_info['new_name']
        target_folder = file_info['target_folder']
        target_path = target_folder / new_name

        success, final_path, msg = handle_duplicate_move(
            file_path,
            target_path,
            expected_target_name=new_name,
            mode="A",
            keep_source_option=keep_source,
            dry_run=False
        )

        if success:
            if "collision" in msg.lower():
                collisions += 1
            installed += 1
        else:
            errors += 1

        print_install_item(idx, len(files_to_install), file_info['filename'], success, msg)

    # Final Summary (using shared helper)
    print_summary(len(files_to_install), installed, collisions, errors, keep_source)

# ============================================================================
# MODUS B - REINSTALL/CHECK
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Reinstall/Check bestehender Encoder

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="TEXT ENCODERS",
        folders="clip/, text_encoders/, clip_vision/",
        extensions="*.safetensors",
        module_type="Text Encoders",
        target_folders="clip/, text_encoders/, clip_vision/",
        preview_mode=preview_mode
    )

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode, not preview/scan)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        if misplaced:
            for file_path in misplaced:
                is_enc, reason, details = is_encoder(file_path)

                if is_enc:
                    encoder_type = reason
                    keys = details["keys"]
                    metadata = details["metadata"]
                    filename = file_path.name
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)

                    print(f"[RESCUE] Found misplaced Encoder: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                    if encoder_type == "clip":
                        enc_details = detect_clip_details(file_path, keys, metadata)
                    elif encoder_type == "text_encoder":
                        enc_details = detect_text_encoder_details(file_path, keys, metadata)
                    elif encoder_type == "clip_vision":
                        enc_details = detect_clipvision_details(file_path, keys, metadata)

                    correct_name = generate_encoder_name(encoder_type, enc_details)
                    target_folder = get_target_folder(encoder_type)
                    target_path = target_folder / correct_name

                    success, final_path, msg = handle_duplicate_move(
                        file_path,
                        target_path,
                        expected_target_name=correct_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    if success:
                        print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: {target_folder.name}/{final_path.name}")
                        remove_misplaced_file(file_path)
                        rescued += 1
                    else:
                        print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                    print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced Encoder(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    folders = [
        ("clip", CLIP_DIR),
        ("text_encoders", TEXT_ENCODERS_DIR),
        ("clip_vision", CLIP_VISION_DIR)
    ]

    all_files = []
    for folder_name, folder_path in folders:
        if not folder_path.exists():
            continue

        safetensors_files = list(folder_path.glob("*.safetensors"))
        for file_path in safetensors_files:
            all_files.append((folder_name, folder_path, file_path))

    if not all_files:
        print_no_files_found("text encoder files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []
    renamed = 0

    for folder_name, folder_path, file_path in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        is_enc, reason, details = is_encoder(file_path)

        if not is_enc:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason, folder_name))
            continue

        encoder_type = reason
        keys = details["keys"]
        metadata = details["metadata"]

        if scan_only:
            continue

        # Check: Richtiger Ordner?
        expected_folder = get_target_folder(encoder_type)

        if expected_folder != folder_path:
            if encoder_type == "clip":
                enc_details = detect_clip_details(file_path, keys, metadata)
            elif encoder_type == "text_encoder":
                enc_details = detect_text_encoder_details(file_path, keys, metadata)
            elif encoder_type == "clip_vision":
                enc_details = detect_clipvision_details(file_path, keys, metadata)

            correct_name = generate_encoder_name(encoder_type, enc_details)
            precision = enc_details.get("precision", "Unknown")
            problems_list.append(('wrong_folder', filename, file_size_mb, encoder_type, precision, folder_name, expected_folder, correct_name, file_path))
            continue

        # Check: Richtiger Name?
        if encoder_type == "clip":
            enc_details = detect_clip_details(file_path, keys, metadata)
        elif encoder_type == "text_encoder":
            enc_details = detect_text_encoder_details(file_path, keys, metadata)
        elif encoder_type == "clip_vision":
            enc_details = detect_clipvision_details(file_path, keys, metadata)

        correct_name = generate_encoder_name(encoder_type, enc_details)

        if filename != correct_name:
            precision = enc_details.get("precision", "Unknown")
            problems_list.append(('wrong_name', filename, file_size_mb, encoder_type, precision, folder_name, correct_name, file_path))
        else:
            correct_files.append((folder_name, filename, file_size_mb))

    # ========================================================================
    # SCAN-ONLY MODE: Just return after building queue
    # ========================================================================
    if scan_only:
        return

    # ========================================================================
    # ANALYSIS (using shared helper with colors)
    # ========================================================================
    print_analysis_b(len(all_files), len(correct_files), len(problems_list))

    # ========================================================================
    # ALREADY CORRECT FILES (using shared helper)
    # ========================================================================
    print_already_correct_section(correct_files)

    # ========================================================================
    # PROBLEMS FOUND
    # ========================================================================
    if problems_list:
        print_problems_header()

        for idx, problem in enumerate(problems_list, 1):
            problem_type = problem[0]

            if problem_type == 'misplaced':
                _, fname, size_mb, reason, folder = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=None,
                    target_path="Added to rescue queue for other modules",
                    warning=f"Not an encoder file: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, enc_type, prec, curr_folder, expected_folder, correct_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{enc_type.replace('_', ' ').title()}, {prec}",
                    target_path=f"{expected_folder.name}/{correct_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    target_path = expected_folder / correct_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=correct_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    print_fix_result(success, "Moved and renamed to correct location" if success else msg)
                    if success:
                        renamed += 1

            elif problem_type == 'wrong_name':
                _, fname, size_mb, enc_type, prec, folder, correct_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{enc_type.replace('_', ' ').title()}, {prec}",
                    target_path=f"{folder}/{correct_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    target_path = fpath.parent / correct_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=correct_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    print_fix_result(success, "Renamed to standard format" if success else msg)
                    if success:
                        renamed += 1

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Modul 3 - CLIP & Text Encoders")
    parser.add_argument("mode", choices=["A", "B"], help="A = Installation, B = Reinstall/Check")
    parser.add_argument("--scan-only", action="store_true", help="PASS 1: Nur Queue aufbauen (nur Modus B)")
    parser.add_argument("--preview", action="store_true", help="Preview mode: show problems without fixing")
    parser.add_argument("--batch", action="store_true", help="Batch mode (skip user prompts)")

    args = parser.parse_args()

    if args.mode == "A":
        modus_a()
    elif args.mode == "B":
        modus_b(scan_only=args.scan_only, batch_mode=args.batch, preview_mode=args.preview)

if __name__ == "__main__":
    main()
