"""
Modul 15 - Vision-Language & Language Models

Verarbeitet Vision-Language & Language Models:
- models/VLM/ (Qwen-VL, LLaVA, MiniCPM-V, etc.)
- models/LLM/ (Florence-2, GPT-style, etc.)

Modus A: Installation aus downloads/
Modus B: Reinstall/Check bestehender Models
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
    VLM_DIR,
    LLM_DIR,
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

# Size limits (mit 20% buffer)
SIZE_MIN = 500 * 1024 * 1024  # 500 MB (VLMs sind groß)
SIZE_MAX = 30 * 1024 * 1024 * 1024  # 30 GB

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
    """Prüft ob VAE Keys vorhanden"""
    vae_patterns = ["first_stage_model", "decoder.conv_in", "encoder.conv_in"]
    return any(any(p in k for p in vae_patterns) for k in keys)

def has_lora_keys(keys):
    """Prüft ob LoRA Keys vorhanden"""
    lora_patterns = ["lora_", ".alpha", "lora.down", "lora.up"]
    return any(any(p in k for p in lora_patterns) for k in keys)

def has_standard_clip_keys(keys):
    """Prüft ob Standard CLIP/Text Encoder (Modul 3)"""
    # CLIP Vision (aber NUR vision, kein language model)
    if any("vision_model" in k or "visual.transformer" in k for k in keys):
        has_language = any(k.startswith("model.layers.") for k in keys)
        if not has_language:
            return True  # Pure CLIP Vision -> Modul 3

    # CLIP Text / T5
    if any("text_model.encoder" in k or "encoder.block" in k for k in keys):
        return True

    return False

# ============================================================================
# VLM/LLM TYPE DETECTION
# ============================================================================

def detect_vlm_llm_type(keys):
    """Unterscheidet: VLM / LLM / None

    VLM = Vision-Language Model (Image→Text)
      - Hat BEIDE: Visual Encoder UND Language Model
      - Keys: "visual.*" + "model.layers.*"
      - Beispiel: Qwen-VL, LLaVA, MiniCPM-V

    LLM = Pure Language Model (Text→Text)
      - Hat NUR Language Model, KEIN Visual
      - Keys: "model.layers.*" OHNE "visual.*"
      - Beispiel: Florence-2, GPT-style

    Returns:
        str: "VLM", "LLM", oder None
    """
    # Check: Hat Language Model?
    has_language = any(k.startswith("model.layers.") for k in keys)

    # Check: Hat Visual Encoder?
    # WICHTIG: Verschiedene Patterns!
    # - Qwen: "visual.blocks.*"
    # - LLaVA: "vision_tower.*"
    # - MiniCPM: "vpm.*"
    has_visual = any(
        k.startswith("visual.") or
        k.startswith("vision_tower.") or
        k.startswith("vpm.") or
        "image_encoder" in k
        for k in keys
    )

    if has_language and has_visual:
        return "VLM"
    elif has_language and not has_visual:
        return "LLM"
    else:
        return None

def is_vlm_llm(file_path):
    """Prüft ob Datei ein VLM/LLM ist

    Returns:
        tuple: (bool, str, dict) - (is_vlm_llm, reason/type, details)
    """
    keys, metadata = read_safetensors_keys(file_path)
    filename = os.path.basename(str(file_path)).lower()

    if not keys:
        return False, "SKIP: Kann Keys nicht lesen", {}

    # AUSNAHME: Qwen3 ist ein Text Encoder für Z-Image, KEIN VLM!
    # Wird von Modul 3 (CLIP & Text Encoders) behandelt
    if 'qwen' in filename and any(x in filename for x in ['_3_', 'qwen3', '_4b', '4b.']):
        return False, "SKIP: Qwen3 Text Encoder (Modul 3)", {}

    # AUSNAHME: Qwen2.5-VL ist primär Visual Encoder für Qwen-Image-Edit
    # Wird von Modul 3 (CLIP & Text Encoders) behandelt → text_encoders/
    if 'qwen' in filename and any(x in filename for x in ['2.5', '2_5']) and 'vl' in filename:
        return False, "SKIP: Qwen2.5-VL Visual Encoder (Modul 3)", {}

    # MUSS NICHT HABEN: UNET
    if has_unet_keys(keys):
        return False, "SKIP: Hat UNET (Base Model - Modul 1)", {}

    # MUSS NICHT HABEN: VAE (ohne Language Model)
    if has_vae_keys(keys):
        has_language = any(k.startswith("model.layers.") for k in keys)
        if not has_language:
            return False, "SKIP: Hat VAE (Modul 2)", {}

    # MUSS NICHT HABEN: LoRA
    if has_lora_keys(keys):
        return False, "SKIP: Hat LoRA (Modul 4)", {}

    # MUSS NICHT HABEN: Standard CLIP/Text Encoder
    if has_standard_clip_keys(keys):
        return False, "SKIP: Standard Encoder (Modul 3)", {}

    # Welcher Typ?
    model_type = detect_vlm_llm_type(keys)

    if model_type:
        return True, model_type, {"keys": keys, "metadata": metadata}
    else:
        return False, "SKIP: Keine VLM/LLM Keys", {}

# ============================================================================
# MODEL ARCHITECTURE DETECTION
# ============================================================================

def detect_architecture(keys, filename):
    """Erkennt Modell-Architektur

    Priorität:
    1. Keys (zuverlässig)
    2. Filename (Fallback - kann umbenannt sein!)

    Returns:
        str: "Qwen-VL", "LLaVA", "Florence-2", etc.
    """
    # Priorität 1: Aus Keys (zuverlässig!)
    # Qwen: "visual.blocks.*"
    if any(k.startswith("visual.blocks.") for k in keys):
        # Check für Version 2.5 aus Filename (nur für Version-Tag)
        filename_lower = filename.lower()
        if "2.5" in filename_lower or "2_5" in filename_lower:
            return "Qwen2.5-VL"
        return "Qwen-VL"

    # LLaVA: "vision_tower.*"
    if any(k.startswith("vision_tower.") for k in keys):
        return "LLaVA"

    # MiniCPM-V: "vpm.*"
    if any(k.startswith("vpm.") for k in keys):
        return "MiniCPM-V"

    # Priorität 2: Aus Filename (Fallback)
    filename_lower = filename.lower()

    if "qwen" in filename_lower:
        if "2.5" in filename_lower or "2_5" in filename_lower:
            return "Qwen2.5-VL"
        return "Qwen-VL"

    elif "llava" in filename_lower:
        return "LLaVA"

    elif "minicpm" in filename_lower:
        return "MiniCPM-V"

    elif "florence" in filename_lower:
        if "florence-2" in filename_lower or "florence_2" in filename_lower:
            return "Florence-2"
        return "Florence"

    elif "cogvlm" in filename_lower:
        return "CogVLM"

    elif "internvl" in filename_lower:
        return "InternVL"

    return "Unknown"

def detect_size(keys, filename, file_size_bytes):
    """Erkennt Modell-Größe (7B, 14B, etc.)

    Priorität:
    1. Layer Count (Fakten aus Keys!)
    2. Filename (Fallback - kann falsch sein!)

    Returns:
        str: "7B", "14B", "Large", etc.
    """
    # Priorität 1: Aus Layer Count (zuverlässig!)
    layer_keys = [k for k in keys if "model.layers." in k]
    if layer_keys:
        layer_nums = []
        for k in layer_keys:
            parts = k.split("model.layers.")
            if len(parts) > 1:
                num_part = parts[1].split(".")[0]
                if num_part.isdigit():
                    layer_nums.append(int(num_part))

        if layer_nums:
            max_layer = max(layer_nums)
            # Heuristik: Layer-Count → Size
            # 28 Layers (0-27) = 7B (Qwen 2.5 VL 7B)
            # 40 Layers (0-39) = 14B
            if max_layer <= 12:
                return "3B"
            elif max_layer <= 28:  # 28 Layers = 7B
                return "7B"
            elif max_layer <= 40:  # 40 Layers = 14B
                return "14B"
            elif max_layer <= 60:
                return "32B"
            else:
                return "72B"

    # Priorität 2: Aus Filename (Fallback)
    filename_lower = filename.lower()

    size_patterns = [
        ("0.5b", "0.5B"),
        ("1.8b", "1.8B"),
        ("3b", "3B"),
        ("7b", "7B"),
        ("8b", "8B"),
        ("14b", "14B"),
        ("32b", "32B"),
        ("72b", "72B"),
    ]

    for pattern, size in size_patterns:
        if pattern in filename_lower:
            return size

    # Size-Tags
    if "large" in filename_lower:
        return "Large"
    elif "base" in filename_lower:
        return "Base"
    elif "small" in filename_lower:
        return "Small"

    return "Unknown"

def detect_variant(filename):
    """Erkennt Modell-Variante (Chat, Instruct, PromptGen, etc.)

    Returns:
        str oder None
    """
    filename_lower = filename.lower()

    if "promptgen" in filename_lower:
        return "PromptGen"
    elif "chat" in filename_lower:
        return "Chat"
    elif "instruct" in filename_lower:
        return "Instruct"
    elif "vision" in filename_lower:
        return "Vision"

    return None

def detect_precision(filename, metadata):
    """Erkennt Precision (FP16, FP8, INT4, etc.)"""
    filename_lower = filename.lower()

    # Quantisierung (Priorität 1)
    if "int4" in filename_lower or "4bit" in filename_lower:
        return "INT4"
    elif "int8" in filename_lower or "8bit" in filename_lower:
        return "INT8"
    elif "fp8" in filename_lower:
        # Check für Scaled/Dynamic
        if "scaled" in filename_lower:
            return "FP8-Scaled"
        elif "dynamic" in filename_lower:
            return "FP8-Dynamic"
        return "FP8"
    elif "fp16" in filename_lower:
        return "FP16"
    elif "fp32" in filename_lower:
        return "FP32"
    elif "bf16" in filename_lower:
        return "BF16"

    # Metadata (Fallback)
    if "__metadata__" in metadata:
        # TODO: Metadata parsing wenn nötig
        pass

    # Default
    return "FP16"

def detect_version(filename):
    """Erkennt Version (v1.0, v2.5, etc.)

    Returns:
        str oder None
    """
    filename_lower = filename.lower()

    # Version Pattern
    import re
    version_patterns = [
        r"v(\d+\.\d+)",
        r"v(\d+)",
        r"_(\d+\.\d+)_",
    ]

    for pattern in version_patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return f"v{match.group(1)}"

    return None

def detect_details(file_path, keys, metadata, model_type):
    """Analysiert VLM/LLM Details

    Returns:
        dict: {
            "model_type": "VLM" oder "LLM",
            "architecture": "Qwen2.5-VL",
            "size": "7B",
            "precision": "FP8-Scaled",
            "variant": "Chat" oder None,
            "version": "v2.5" oder None
        }
    """
    filename = os.path.basename(file_path)
    file_size_bytes = os.path.getsize(file_path)

    architecture = detect_architecture(keys, filename)
    size = detect_size(keys, filename, file_size_bytes)
    precision = detect_precision(filename, metadata)
    variant = detect_variant(filename)
    version = detect_version(filename)

    return {
        "model_type": model_type,
        "architecture": architecture,
        "size": size,
        "precision": precision,
        "variant": variant,
        "version": version
    }

# ============================================================================
# NAME GENERATION
# ============================================================================

def generate_vlm_llm_name(details):
    """Generiert standardisierten Namen

    Format VLM:
        {Architecture}_{Size}_{Precision}_vlm.safetensors
        {Architecture}_{Size}_{Precision}_{Variant}_vlm.safetensors
        {Architecture}_{Size}_{Precision}_{Variant}_{Version}_vlm.safetensors

    Format LLM:
        {Architecture}_{Size}_{Precision}_llm.safetensors
        {Architecture}_{Size}_{Precision}_{Variant}_llm.safetensors

    Beispiele:
        Qwen2.5-VL_7B_FP8-Scaled_vlm.safetensors
        Florence-2_Large_FP16_PromptGen_v2.0_llm.safetensors
    """
    model_type = details["model_type"]
    architecture = details["architecture"]
    size = details["size"]
    precision = details["precision"]
    variant = details["variant"]
    version = details["version"]

    # Base Name
    parts = [architecture, size, precision]

    # Optional: Variant
    if variant:
        parts.append(variant)

    # Optional: Version
    if version:
        parts.append(version)

    # Type Suffix
    if model_type == "VLM":
        parts.append("vlm")
    elif model_type == "LLM":
        parts.append("llm")

    return "_".join(parts) + ".safetensors"

# ============================================================================
# TARGET FOLDER ASSIGNMENT
# ============================================================================

def get_target_folder(model_type):
    """Gibt Target-Ordner zurück"""
    if model_type == "VLM":
        return VLM_DIR
    elif model_type == "LLM":
        return LLM_DIR
    else:
        return None

# ============================================================================
# EXPORTABLE FUNCTIONS (for all_modules.py batch processing)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads and return file list (for batch processing)"""
    files_to_install = []
    skipped = []

    safetensors_files = list(downloads_path.glob("**/*.safetensors"))  # recursive

    for file_path in safetensors_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024

        file_size = file_path.stat().st_size
        if file_size < SIZE_MIN or file_size > SIZE_MAX:
            skipped.append({'filename': filename, 'reason': 'size out of range', 'size_gb': size_gb})
            continue

        is_model, reason, details = is_vlm_llm(file_path)

        if not is_model:
            skipped.append({'filename': filename, 'reason': reason, 'size_gb': size_gb})
            continue

        model_type = reason
        keys = details["keys"]
        metadata = details["metadata"]

        model_details = detect_details(file_path, keys, metadata, model_type)
        new_name = generate_vlm_llm_name(model_details)
        target_folder = get_target_folder(model_type)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'size_gb': size_gb,
            'new_name': new_name,
            'target_folder': target_folder,
            'model_type': model_type,
            'details': model_details
        })

    return {
        'module_name': 'VLM & LLM',
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
        module_name_caps="VLM & LLM (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.safetensors",
        module_type="VLM & LLM",
        target_folders="VLM/, LLM/"
    )

    all_files = list(DOWNLOADS_DIR.glob("**/*.safetensors"))  # recursive

    if not all_files:
        print_no_files_found("VLM/LLM files")
        return

    # ========================================================================
    # PHASE 1: PREVIEW - Analyze all files (silently collect)
    # ========================================================================
    files_to_install = []
    skipped = []

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024 * 1024)
        size_gb = size_mb / 1024

        # Size check (Fast-Skip)
        file_size = file_path.stat().st_size
        if file_size < SIZE_MIN or file_size > SIZE_MAX:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # Module Boundary Check
        is_model, reason, details = is_vlm_llm(file_path)

        if not is_model:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # reason ist model_type: "VLM" oder "LLM"
        model_type = reason
        keys = details["keys"]
        metadata = details["metadata"]

        # Detect details
        model_details = detect_details(file_path, keys, metadata, model_type)

        # Generate name
        new_name = generate_vlm_llm_name(model_details)

        # Target folder
        target_folder = get_target_folder(model_type)

        # Build detected string
        arch = model_details['architecture']
        size = model_details['size']
        precision = model_details['precision']
        detected_str = f"{model_type}, {arch}-{size}, {precision}"

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
    print_analysis(len(all_files), len(files_to_install), len(skipped))

    # ========================================================================
    # SKIPPED SECTION (using shared helper)
    # ========================================================================
    print_skipped_section(
        skipped_files=skipped,
        skip_reason="Not a VLM/LLM model (size/type mismatch)",
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
        folder_name = file_info['target_folder'].name
        print_preview_item(
            index=i,
            filename=file_info['filename'],
            size_mb=file_info['size_mb'],
            detected_info=file_info['detected_str'],
            target_path=f"{folder_name}/{file_info['new_name']}"
        )

    total_size = sum(f['size_gb'] for f in files_to_install)
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

    VLM_DIR.mkdir(parents=True, exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)

    installed = 0
    errors = 0
    collisions = 0

    for idx, file_info in enumerate(files_to_install, 1):
        source_path = file_info['path']
        filename = file_info['filename']
        new_name = file_info['new_name']
        target_folder = file_info['target_folder']
        target_path = target_folder / new_name

        success, final_path, msg = handle_duplicate_move(
            source_path,
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

        print_install_item(idx, len(files_to_install), filename, success, msg)

    print_summary(len(files_to_install), installed, collisions, errors, keep_source)

# ============================================================================
# MODUS B - REINSTALL/CHECK
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Reinstall/Check bestehender Models

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="VLM & LLM",
        folders="VLM/, LLM/",
        extensions="*.safetensors",
        module_type="VLM & LLM",
        target_folders="VLM/, LLM/",
        preview_mode=preview_mode
    )

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode, not preview/scan)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        for file_path in misplaced:
            is_model, reason, details = is_vlm_llm(file_path)
            if not is_model:
                continue

            model_type = reason
            keys = details["keys"]
            metadata = details["metadata"]
            filename = file_path.name
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            print(f"[RESCUE] Found misplaced VLM/LLM: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

            model_details = detect_details(file_path, keys, metadata, model_type)
            correct_name = generate_vlm_llm_name(model_details)
            target_folder = get_target_folder(model_type)
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
            print(f"[SUCCESS] Rescued {rescued} misplaced VLM/LLM(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    # Scan both folders
    folders = [
        ("VLM", VLM_DIR),
        ("LLM", LLM_DIR)
    ]

    # Collect all files first
    all_files = []
    for folder_name, folder_path in folders:
        if not folder_path.exists():
            continue

        safetensors_files = list(folder_path.glob("**/*.safetensors"))  # recursive
        for file_path in safetensors_files:
            all_files.append((folder_name, folder_path, file_path))

    if not all_files:
        print_no_files_found("VLM/LLM files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []

    for folder_name, folder_path, file_path in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Module Boundary Check
        is_model, reason, details = is_vlm_llm(file_path)

        if not is_model:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason, folder_name))
            continue

        # reason ist model_type
        model_type = reason
        keys = details["keys"]
        metadata = details["metadata"]

        if scan_only:
            continue

        # Check: Richtiger Ordner?
        expected_folder = get_target_folder(model_type)

        if expected_folder != folder_path:
            model_details = detect_details(file_path, keys, metadata, model_type)
            correct_name = generate_vlm_llm_name(model_details)
            arch = model_details['architecture']
            size = model_details['size']
            precision = model_details['precision']
            detected_str = f"{model_type}, {arch}-{size}, {precision}"
            problems_list.append(('wrong_folder', filename, file_size_mb, detected_str, expected_folder, correct_name, file_path))
            continue

        # Check: Richtiger Name?
        model_details = detect_details(file_path, keys, metadata, model_type)
        correct_name = generate_vlm_llm_name(model_details)
        arch = model_details['architecture']
        size = model_details['size']
        precision = model_details['precision']
        detected_str = f"{model_type}, {arch}-{size}, {precision}"

        if filename != correct_name:
            problems_list.append(('wrong_name', filename, file_size_mb, detected_str, folder_name, correct_name, file_path))
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
                    warning=f"Not a VLM/LLM file: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, detected_str, expected_folder, correct_name, file_path = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=detected_str,
                    target_path=f"{expected_folder.name}/{correct_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    target_path = expected_folder / correct_name
                    success, final_path, msg = handle_duplicate_move(
                        file_path,
                        target_path,
                        expected_target_name=correct_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )
                    print_fix_result(success, "Moved and renamed to correct location" if success else msg)

            elif problem_type == 'wrong_name':
                _, fname, size_mb, detected_str, folder, correct_name, file_path = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=detected_str,
                    target_path=f"{folder}/{correct_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    target_path = file_path.parent / correct_name
                    success, final_path, msg = handle_duplicate_move(
                        file_path,
                        target_path,
                        expected_target_name=correct_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )
                    print_fix_result(success, "Renamed to standard format" if success else msg)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Modul 23 - Vision-Language & Language Models")
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
