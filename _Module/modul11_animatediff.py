#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODUL 11 - ANIMATEDIFF (Motion Modules + Motion LoRAs)

Verwaltet zwei AnimateDiff Komponenten:
1. Motion Modules: Große Modelle (0.8-1.7 GB) für Animation
2. Motion LoRAs: Kleine Modelle (~77 MB) für Kamera-Bewegungen

Detection: Key-basiert (PRIMARY) - nicht Size-basiert!
"""

import sys
import os
from pathlib import Path

# ============================================================================
# PATH SETUP - Für Unterordner-Struktur
# ============================================================================
_SCRIPT_DIR = Path(__file__).parent
_SHARED_DIR = _SCRIPT_DIR.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from shared_utils import (
    DOWNLOADS_DIR,
    MODELS_DIR,
    ANIMATEDIFF_MODELS_DIR,
    MOTION_LORA_DIR,
    handle_duplicate_move,
    read_misplaced_files,
    add_misplaced_file,
    remove_misplaced_file,
    ask_keep_or_delete,
    ask_confirm_installation,
    ask_confirm_fixes,
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
# KEY READING FUNCTIONS
# ============================================================================

def read_safetensors_keys(file_path):
    """Read keys from safetensors file"""
    try:
        from safetensors import safe_open
        keys = []
        with safe_open(file_path, framework="pt") as f:
            keys = list(f.keys())
        return keys
    except Exception as e:
        return []

def read_safetensors_metadata(file_path):
    """Read metadata from safetensors file"""
    try:
        from safetensors import safe_open
        with safe_open(file_path, framework="pt") as f:
            metadata = f.metadata() if hasattr(f, 'metadata') else {}
        return metadata or {}
    except Exception as e:
        return {}

def read_torch_keys(file_path):
    """Read keys from PyTorch file (.ckpt, .pth, .pt)"""
    try:
        import torch
        data = torch.load(file_path, map_location='cpu', weights_only=True)
        if isinstance(data, dict):
            return list(data.keys())
        return []
    except Exception as e:
        return []

def load_keys(file_path):
    """Load keys from any supported format"""
    if file_path.suffix == '.safetensors':
        return read_safetensors_keys(file_path)
    else:
        return read_torch_keys(file_path)

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_animatediff_file(file_path):
    """
    Detect if file is AnimateDiff and distinguish Motion Module vs Motion LoRA

    CRITICAL: Key-based detection is PRIMARY (not size!)

    Returns:
        {
            'is_animatediff': bool,
            'type': 'MotionModule' | 'MotionLoRA' | None,
            'base_model': 'SD15' | 'SDXL' | 'Unknown',
            'version': str,
            'motion_type': str (only LoRAs),
            'variant': str | None,
            'reason': str (if not animatediff)
        }
    """

    # Check extension
    if file_path.suffix.lower() not in ['.safetensors', '.ckpt', '.pth', '.pt']:
        return {'is_animatediff': False, 'reason': 'Wrong extension'}

    # Load keys (MANDATORY for reliable detection)
    keys = load_keys(file_path)

    if not keys:
        return {'is_animatediff': False, 'reason': 'Could not load keys'}

    # Check 1: Has motion_modules keys?
    has_motion_modules = any('motion_modules' in k for k in keys)

    # Check 2: Has lora patterns?
    has_lora_patterns = any(('.lora.down' in k or '.lora.up' in k or
                             '_lora.down' in k or '_lora.up' in k) for k in keys)

    # Decision Logic
    if has_motion_modules and has_lora_patterns:
        type_detected = "MotionLoRA"
    elif has_motion_modules and not has_lora_patterns:
        type_detected = "MotionModule"
    elif has_lora_patterns and not has_motion_modules:
        # Regular LoRA, NOT AnimateDiff
        return {'is_animatediff': False, 'reason': 'Regular LoRA (no motion_modules)'}
    else:
        # No AnimateDiff patterns
        return {'is_animatediff': False, 'reason': 'No AnimateDiff patterns'}

    # Detail Detection
    filename_lower = file_path.name.lower()

    # Base Model Detection
    base_model = detect_base_model(filename_lower, keys)

    # Version Detection
    metadata = read_safetensors_metadata(file_path) if file_path.suffix == '.safetensors' else {}
    version = detect_version(filename_lower, metadata)

    # Variant Detection
    variant = detect_variant(filename_lower)

    # Motion Type (only for LoRAs)
    motion_type = None
    if type_detected == "MotionLoRA":
        motion_type = detect_motion_type(filename_lower)

    return {
        'is_animatediff': True,
        'type': type_detected,
        'base_model': base_model,
        'version': version,
        'motion_type': motion_type,
        'variant': variant
    }

def detect_base_model(filename_lower, keys):
    """
    Detect base model (SD15 vs SDXL)

    Priority 1: Filename (90% reliable)
    Priority 2: Keys (100% reliable)
    Priority 3: Fallback SD15 (most common)
    """

    # Priority 1: Filename
    if 'sdxl' in filename_lower or 'xl' in filename_lower or 'hotshot' in filename_lower:
        return "SDXL"

    if 'sd15' in filename_lower or 'sd_15' in filename_lower or 'v15' in filename_lower or 'v1.5' in filename_lower:
        return "SD15"

    # Priority 2: Keys (definitive)
    # SD1.5 has 4 down_blocks (0-3), SDXL has 3 (0-2)
    if keys:
        has_down_blocks_3 = any('down_blocks.3' in k for k in keys)
        if has_down_blocks_3:
            return "SD15"

        # Check if down_blocks.2 exists (both have it)
        has_down_blocks_2 = any('down_blocks.2' in k for k in keys)
        if has_down_blocks_2 and not has_down_blocks_3:
            return "SDXL"

    # Fallback: SD15 (most common)
    return "SD15"

def detect_version(filename_lower, metadata=None):
    """
    Detect version (v14, v15, v2, v3)

    Returns: "v14" | "v15" | "v2" | "v3" | "Unknown"
    """

    # Priority 1: Filename
    if '_v3' in filename_lower or 'v3' in filename_lower:
        return "v3"
    elif '_v2' in filename_lower or 'v2' in filename_lower:
        return "v2"
    elif 'v15' in filename_lower or 'v1.5' in filename_lower or '_v15' in filename_lower:
        return "v15"
    elif 'v14' in filename_lower or 'v1.4' in filename_lower or '_v14' in filename_lower:
        return "v14"

    # Priority 2: Metadata (if available)
    if metadata and 'version' in metadata:
        version_str = str(metadata['version']).lower()
        if 'v3' in version_str or '3.' in version_str:
            return "v3"
        elif 'v2' in version_str or '2.' in version_str:
            return "v2"
        elif 'v15' in version_str or '1.5' in version_str:
            return "v15"
        elif 'v14' in version_str or '1.4' in version_str:
            return "v14"

    return "Unknown"

def detect_motion_type(filename_lower):
    """
    Detect motion type from filename (only for LoRAs)

    Returns: "ZoomIn" | "ZoomOut" | "PanLeft" | "PanRight" |
             "TiltUp" | "TiltDown" | "RollCW" | "RollCCW" | "General"
    """

    motion_types = {
        "ZoomIn": ["zoomin", "zoom_in"],
        "ZoomOut": ["zoomout", "zoom_out"],
        "PanLeft": ["panleft", "pan_left"],
        "PanRight": ["panright", "pan_right"],
        "TiltUp": ["tiltup", "tilt_up"],
        "TiltDown": ["tiltdown", "tilt_down"],
        "RollCW": ["rollingclockwise", "rollcw", "roll_cw", "clockwise"],
        "RollCCW": ["rollinganticlockwise", "rollccw", "roll_ccw", "anticlockwise", "counterclockwise"],
    }

    for motion_type, keywords in motion_types.items():
        if any(kw in filename_lower for kw in keywords):
            return motion_type

    return "General"

def detect_variant(filename_lower):
    """
    Detect special variants

    Returns: str | None
    """

    variants = []

    if 'hotshot' in filename_lower or 'hsxl' in filename_lower:
        variants.append("HotShot")

    if 'fp16' in filename_lower or 'f16' in filename_lower or '.f16.' in filename_lower:
        variants.append("FP16")

    if 'lightning' in filename_lower:
        # Extract step count if available
        import re
        match = re.search(r'(\d+)step', filename_lower)
        if match:
            variants.append(f"Lightning-{match.group(1)}step")
        else:
            variants.append("Lightning")

    if 'beta' in filename_lower:
        variants.append("Beta")

    if 'stabilized' in filename_lower:
        variants.append("Stabilized")

    if 'temporaldiff' in filename_lower:
        variants.append("TemporalDiff")

    if 'lcm' in filename_lower or 'animatelcm' in filename_lower:
        variants.append("LCM")

    return '-'.join(variants) if variants else None

def generate_proper_name(result, file_path):
    """Generate proper filename based on detection results"""

    base_model = result['base_model']
    version = result['version']
    variant = result['variant']
    file_ext = file_path.suffix

    if result['type'] == 'MotionModule':
        # AnimateDiff_{BaseModel}_{Version}_MotionModule[_Variant].ext
        if variant:
            return f"AnimateDiff_{base_model}_{version}_MotionModule_{variant}{file_ext}"
        else:
            return f"AnimateDiff_{base_model}_{version}_MotionModule{file_ext}"

    else:  # MotionLoRA
        # AnimateDiff_{BaseModel}_{Version}_MotionLoRA_{MotionType}[_Variant].ext
        motion_type = result['motion_type']
        if variant:
            return f"AnimateDiff_{base_model}_{version}_MotionLoRA_{motion_type}_{variant}{file_ext}"
        else:
            return f"AnimateDiff_{base_model}_{version}_MotionLoRA_{motion_type}{file_ext}"

# ============================================================================
# MODUS A - INSTALLATION
# ============================================================================

def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="ANIMATEDIFF (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.safetensors, *.ckpt, *.pth",
        module_type="AnimateDiff",
        target_folders="animatediff_models/, animatediff_motion_lora/"
    )

    all_files = []
    all_files.extend(DOWNLOADS_DIR.glob("*.safetensors"))
    all_files.extend(DOWNLOADS_DIR.glob("*.ckpt"))
    all_files.extend(DOWNLOADS_DIR.glob("*.pth"))
    all_files.extend(DOWNLOADS_DIR.glob("*.pt"))

    if not all_files:
        print_no_files_found("AnimateDiff files")
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

        result = detect_animatediff_file(file_path)

        if not result['is_animatediff']:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # Build detected string
        if result['type'] == 'MotionModule':
            variant_str = f", {result['variant']}" if result['variant'] else ""
            detected_str = f"Motion Module, {result['base_model']}, {result['version']}{variant_str}"
            target_folder_name = "animatediff_models"
        else:
            variant_str = f", {result['variant']}" if result['variant'] else ""
            detected_str = f"Motion LoRA, {result['base_model']}, {result['version']}, {result['motion_type']}{variant_str}"
            target_folder_name = "animatediff_motion_lora"

        proper_name = generate_proper_name(result, file_path)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'result': result,
            'proper_name': proper_name,
            'target_folder_name': target_folder_name,
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
        skip_reason="Not an AnimateDiff model (no motion module keys found)",
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
            target_path=f"{file_info['target_folder_name']}/{file_info['proper_name']}"
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

    installed = 0
    errors = 0
    collisions = 0

    for idx, file_info in enumerate(files_to_install, 1):
        source_path = file_info['path']
        result = file_info['result']

        if result['type'] == 'MotionModule':
            target_folder = ANIMATEDIFF_MODELS_DIR
        else:
            target_folder = MOTION_LORA_DIR

        proper_name = file_info['proper_name']
        target_path = target_folder / proper_name

        target_folder.mkdir(parents=True, exist_ok=True)

        success, final_path, msg = handle_duplicate_move(
            source_path,
            target_path,
            expected_target_name=proper_name,
            mode="A",
            keep_source_option=keep_source
        )

        if success:
            if "collision" in msg.lower():
                collisions += 1
            installed += 1
        else:
            errors += 1

        print_install_item(idx, len(files_to_install), file_info['filename'], success, msg)

    print_summary(len(files_to_install), installed, collisions, errors, keep_source)

# ============================================================================
# MODUS B - REINSTALL/CHECK
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check bestehender AnimateDiff Files"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="ANIMATEDIFF",
        folders="animatediff_models/, animatediff_motion_lora/",
        extensions="*.safetensors, *.ckpt, *.pth",
        module_type="AnimateDiff",
        target_folders="animatediff_models/, animatediff_motion_lora/",
        preview_mode=preview_mode
    )

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode, not preview/scan)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        for file_path in misplaced:
            result = detect_animatediff_file(file_path)

            if result['is_animatediff']:
                filename = file_path.name
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                if result['type'] == 'MotionModule':
                    target_folder = ANIMATEDIFF_MODELS_DIR
                else:
                    target_folder = MOTION_LORA_DIR

                proper_name = generate_proper_name(result, file_path)
                target_path = target_folder / proper_name

                print(f"[RESCUE] Found misplaced AnimateDiff: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                target_folder.mkdir(parents=True, exist_ok=True)

                success, final_path, msg = handle_duplicate_move(
                    file_path,
                    target_path,
                    expected_target_name=proper_name,
                    mode="B",
                    keep_source_option=False
                )

                if success:
                    print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: {target_folder.name}/{final_path.name}")
                    remove_misplaced_file(file_path)
                    rescued += 1
                else:
                    print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced AnimateDiff file(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    all_files = []
    if ANIMATEDIFF_MODELS_DIR.exists():
        all_files.extend(ANIMATEDIFF_MODELS_DIR.rglob("*.safetensors"))
        all_files.extend(ANIMATEDIFF_MODELS_DIR.rglob("*.ckpt"))
        all_files.extend(ANIMATEDIFF_MODELS_DIR.rglob("*.pth"))
        all_files.extend(ANIMATEDIFF_MODELS_DIR.rglob("*.pt"))

    if MOTION_LORA_DIR.exists():
        all_files.extend(MOTION_LORA_DIR.rglob("*.safetensors"))
        all_files.extend(MOTION_LORA_DIR.rglob("*.ckpt"))
        all_files.extend(MOTION_LORA_DIR.rglob("*.pth"))
        all_files.extend(MOTION_LORA_DIR.rglob("*.pt"))

    if not all_files:
        print_no_files_found("AnimateDiff files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []
    renamed = 0
    moved = 0

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024 * 1024)

        result = detect_animatediff_file(file_path)

        if not result['is_animatediff']:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, size_mb, result['reason']))
            continue

        if scan_only:
            continue

        if result['type'] == 'MotionModule':
            expected_folder = ANIMATEDIFF_MODELS_DIR
        else:
            expected_folder = MOTION_LORA_DIR

        proper_name = generate_proper_name(result, file_path)
        current_folder = file_path.parent

        if current_folder != expected_folder:
            problems_list.append(('wrong_folder', filename, size_mb, current_folder.name, expected_folder.name, expected_folder, proper_name, file_path, result))
        elif file_path.name != proper_name:
            problems_list.append(('wrong_name', filename, size_mb, file_path.parent.name, proper_name, file_path, result))
        else:
            correct_files.append((file_path.parent.name, filename, size_mb))

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
                _, fname, size_mb, reason = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=None,
                    target_path="Added to rescue queue for other modules",
                    warning=f"Not an AnimateDiff: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, curr_folder, exp_folder_name, exp_folder, proper_name, fpath, result = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"Current: {curr_folder}/",
                    target_path=f"{exp_folder_name}/{proper_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    exp_folder.mkdir(parents=True, exist_ok=True)
                    new_path = exp_folder / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        new_path,
                        expected_target_name=proper_name,
                        mode="B",
                        keep_source_option=False
                    )

                    print_fix_result(success, "Moved to correct folder" if success else msg)
                    if success:
                        moved += 1

            elif problem_type == 'wrong_name':
                _, fname, size_mb, folder_name, proper_name, fpath, result = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=None,
                    target_path=f"{folder_name}/{proper_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    new_path = fpath.parent / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        new_path,
                        expected_target_name=proper_name,
                        mode="B",
                        keep_source_option=False
                    )

                    print_fix_result(success, "Renamed to standard format" if success else msg)
                    if success:
                        renamed += 1

# ============================================================================
# BATCH PROCESSING (für all_modules.py)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads for AnimateDiff files

    Returns:
        {
            'module_name': str,
            'files': [
                {
                    'file_path': Path,
                    'filename': str,
                    'size_gb': float,
                    'result': dict,
                    'target_folder': Path,
                    'proper_name': str,
                    'type': str
                },
                ...
            ]
        }
    """

    all_files = []
    all_files.extend(downloads_path.glob("*.safetensors"))
    all_files.extend(downloads_path.glob("*.ckpt"))
    all_files.extend(downloads_path.glob("*.pth"))
    all_files.extend(downloads_path.glob("*.pt"))

    files_to_install = []

    for file_path in all_files:
        result = detect_animatediff_file(file_path)

        if not result['is_animatediff']:
            continue

        # Determine target folder
        if result['type'] == 'MotionModule':
            target_folder = ANIMATEDIFF_MODELS_DIR
        else:
            target_folder = MOTION_LORA_DIR

        # Generate proper name
        proper_name = generate_proper_name(result, file_path)

        files_to_install.append({
            'file_path': file_path,
            'filename': file_path.name,
            'size_gb': file_path.stat().st_size / (1024 * 1024 * 1024),
            'result': result,
            'target_folder': target_folder,
            'proper_name': proper_name,
            'type': result['type']
        })

    return {
        'module_name': 'AnimateDiff (Motion Modules + Motion LoRAs)',
        'files': files_to_install
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python modul15_animatediff.py <A|B> [--scan-only] [--batch]")
        print("  A = Modus A (Installation from downloads/)")
        print("  B = Modus B (Check/Fix existing files)")
        print("  --scan-only = Scan only (PASS 1 for Modus B)")
        print("  --batch = Batch mode (skip confirmations)")
        sys.exit(1)

    mode = sys.argv[1].upper()
    scan_only = '--scan-only' in sys.argv
    batch_mode = '--batch' in sys.argv

    if mode == 'A':
        modus_a()
    elif mode == 'B':
        modus_b(scan_only=scan_only, batch_mode=batch_mode)
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'A' for Installation or 'B' for Check/Fix")
        sys.exit(1)

if __name__ == "__main__":
    main()
