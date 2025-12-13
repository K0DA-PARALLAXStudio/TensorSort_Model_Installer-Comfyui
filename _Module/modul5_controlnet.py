#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 5 - ControlNet & T2I-Adapter
Vollständige Implementation mit Modus A und B
"""

import os
import sys
import json
import struct
from pathlib import Path

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
    DOWNLOADS_DIR,
    CONTROLNET_DIR,
    T2I_ADAPTER_DIR,
    MODELS_DIR,
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
# KONFIGURATION
# ============================================================================

# Paths (from shared_utils.py)
DOWNLOADS_PATH = DOWNLOADS_DIR
CONTROLNET_PATH = CONTROLNET_DIR
T2I_ADAPTER_PATH = T2I_ADAPTER_DIR

# Größen-Grenze für ControlNet/T2I-Adapter
MIN_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB

# ============================================================================
# PHASE 1: BASIS-FUNKTIONEN
# ============================================================================

def read_safetensors_keys(file_path):
    """Liest Keys aus Safetensors (ohne komplette Datei zu laden)"""
    try:
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            metadata_bytes = f.read(header_size)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            return list(metadata.keys())
    except Exception as e:
        return []


def read_safetensors_metadata(file_path):
    """Liest komplette Metadata aus Safetensors"""
    try:
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            metadata_bytes = f.read(header_size)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            return metadata
    except Exception as e:
        return {}


def is_controlnet_or_adapter(file_path):
    """Prüft ob Datei ein ControlNet oder T2I-Adapter ist (Module Boundary)

    Returns:
        tuple: (is_mine, detected_type, reason)
        - is_mine: bool
        - detected_type: "controlnet" | "t2i_adapter" | None
        - reason: str (for logging)
    """

    # 1. Extension Check
    if not str(file_path).endswith((".safetensors", ".pth")):
        return False, None, "Wrong extension"

    # 2. Size Check (Performance)
    file_size = os.path.getsize(file_path)
    if file_size < MIN_SIZE:
        return False, None, "Too small (< 100 MB)"
    if file_size > MAX_SIZE:
        return False, None, "Too large (> 10 GB, probably Base Model)"

    # 3. Keys lesen
    keys = read_safetensors_keys(file_path)
    if not keys:
        return False, None, "Cannot read keys"

    # 4. T2I-Adapter Detection (separate Architektur)
    has_adapter_body = any(k.startswith("adapter.body.") for k in keys)
    if has_adapter_body:
        return True, "t2i_adapter", "T2I-Adapter keys detected"

    # 5. ControlNet Detection (BEIDE Formate!)
    has_control_model = any(k.startswith("control_model.") for k in keys)      # Original (lllyasviel)
    has_controlnet = any(k.startswith("controlnet_") for k in keys)             # Diffusers
    has_controlnet_blocks = any(k.startswith("controlnet_blocks.") for k in keys)  # Flux

    if has_control_model or has_controlnet or has_controlnet_blocks:
        # Weitere Validierung: NICHT Base Model!

        # Base Models haben diese Keys OHNE control prefix:
        has_base_model_unet = any(k.startswith("model.diffusion_model.") and
                                   not k.startswith("control_model.") for k in keys)
        has_flux_base = any("double_blocks.0.img_attn" in k and
                           not k.startswith("controlnet_") for k in keys)

        if has_base_model_unet or has_flux_base:
            return False, None, "Is Base Model, not ControlNet"

        # LoRA ausschließen
        lora_patterns = ["lora_up.weight", "lora_down.weight", ".alpha"]
        has_lora = any(any(p in k for p in lora_patterns) for k in keys)
        if has_lora:
            return False, None, "Is LoRA (Modul 4)"

        return True, "controlnet", "ControlNet keys detected"

    return False, None, "No ControlNet or T2I-Adapter keys"


# ============================================================================
# PHASE 2: DETECTION FUNCTIONS
# ============================================================================

def detect_precision_from_tensors(metadata):
    """Erkennt Precision aus Tensor dtypes (Ground Truth!)"""

    dtypes = {}
    for key, value in metadata.items():
        if key == "__metadata__":
            continue
        if isinstance(value, dict) and "dtype" in value:
            dtype = value["dtype"]
            dtypes[dtype] = dtypes.get(dtype, 0) + 1

    if not dtypes:
        return None

    # Dominanten dtype finden (>50%)
    total = sum(dtypes.values())
    for dtype, count in dtypes.items():
        if count / total > 0.5:
            # Mapping
            if "F32" in dtype or "float32" in dtype:
                return "FP32"
            elif "F16" in dtype or "float16" in dtype:
                return "FP16"
            elif "BF16" in dtype or "bfloat16" in dtype:
                return "BF16"
            elif "F8" in dtype or "float8" in dtype:
                return "FP8"

    return "Mixed"


def detect_precision(metadata, filename):
    """Erkennt Precision

    Priority:
    1. Tensor dtype (Ground Truth!)
    2. Filename (Fallback)
    """

    # 1. Tensor dtype (ALWAYS available, Ground Truth!)
    precision = detect_precision_from_tensors(metadata)
    if precision:
        return precision

    # 2. Filename Fallback
    filename_lower = filename.lower()
    if "fp16" in filename_lower:
        return "FP16"
    elif "fp32" in filename_lower:
        return "FP32"
    elif "bf16" in filename_lower:
        return "BF16"
    elif "fp8" in filename_lower:
        return "FP8"

    return "Unknown"


def detect_base_model(keys, filename):
    """Erkennt Base Model

    Priority:
    1. Keys (Ground Truth für Flux/SDXL)
    2. Filename (Fallback für SD1.5/SD2.1/T2I-Adapter)
    """

    # SDXL: add_embedding.* (unique marker!)
    if any(k.startswith("add_embedding.") for k in keys):
        return "SDXL"

    # Flux: controlnet_single_blocks.* + context_embedder.*
    if any(k.startswith("controlnet_single_blocks.") for k in keys):
        if any("context_embedder" in k for k in keys):
            return "Flux"

    # SD1.5 vs SD2.1: Filename Fallback
    filename_lower = filename.lower()
    if "sdxl" in filename_lower or "xl" in filename_lower:
        return "SDXL"
    elif "flux" in filename_lower:
        return "Flux"
    elif "sd21" in filename_lower or "sd2.1" in filename_lower or "sd_v2" in filename_lower:
        return "SD21"
    elif "sd15" in filename_lower or "sd1.5" in filename_lower or "sd_v1" in filename_lower:
        return "SD15"

    # Default zu SD1.5 (häufigster)
    return "SD15"


def detect_control_type(keys, metadata, filename):
    """Erkennt Control Type

    Priority:
    1. Keys (Union Detection only)
    2. Metadata (control_type field - rare)
    3. Filename Pattern Matching
    4. "Unknown" Fallback
    """

    # 1. Union Detection (Keys)
    if any("controlnet_mode_embedder" in k for k in keys):
        return "Union"

    # 2. Metadata Check (rare)
    if "__metadata__" in metadata:
        meta_info = metadata["__metadata__"]
        control_type = meta_info.get("control_type")
        if control_type:
            return control_type.capitalize()

    # 3. Filename Pattern Matching
    filename_lower = filename.lower()

    control_types = {
        "Canny": ["canny"],
        "Depth": ["depth", "midas", "zoe"],
        "OpenPose": ["openpose", "pose"],
        "Lineart": ["lineart"],
        "Scribble": ["scribble"],
        "SoftEdge": ["softedge", "hed"],
        "MLSD": ["mlsd"],
        "Normal": ["normal", "normalmap"],
        "Tile": ["tile"],
        "Seg": ["seg", "segmentation"],
        "Blur": ["blur"],
        "Shuffle": ["shuffle"],
        "IP2P": ["ip2p", "instruct"],
        "Inpaint": ["inpaint"],
    }

    for control_type, keywords in control_types.items():
        if any(kw in filename_lower for kw in keywords):
            return control_type

    # 4. Unknown Fallback
    return "Unknown"


def detect_version(filename):
    """Erkennt Version (Filename only - NO keys/metadata available)"""

    filename_lower = filename.lower()

    # ControlNet 1.1
    if "v11" in filename_lower:
        if "v11f1" in filename_lower:
            return "v1.1-f1"  # Bug-fix
        elif "v11e" in filename_lower:
            return "v1.1-e"   # Experimental
        else:
            return "v1.1"

    # ControlNet 1.0
    if "v10" in filename_lower or "v1.0" in filename_lower:
        return "v1.0"

    # Flux variants
    if "pro" in filename_lower:
        return "pro"
    elif "alpha" in filename_lower:
        return "alpha"
    elif "beta" in filename_lower:
        return "beta"

    # Generic version pattern
    import re
    match = re.search(r'v(\d+)', filename_lower)
    if match:
        return f"v{match.group(1)}"

    return "v1"  # Default


# ============================================================================
# PHASE 3: NAMING
# ============================================================================

def generate_proper_name(file_info):
    """Generiert korrekten Namen nach Konvention

    ControlNet: {BaseModel}_CN-{ControlType}_{Precision}_{Version}.safetensors
    T2I-Adapter: {BaseModel}_T2I-{ControlType}_{Precision}_{Version}.safetensors
    """

    base_model = file_info["base_model"]
    control_type = file_info["control_type"]
    precision = file_info["precision"]
    version = file_info["version"]
    detected_type = file_info["detected_type"]

    if detected_type == "t2i_adapter":
        # T2I-Adapter Format
        return f"{base_model}_T2I-{control_type}_{precision}_{version}.safetensors"
    else:
        # ControlNet Format
        return f"{base_model}_CN-{control_type}_{precision}_{version}.safetensors"


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_file(file_path):
    """Komplette Analyse einer Datei"""

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    size_gb = file_size / (1024**3)

    # 1. Module Boundary Check
    is_mine, detected_type, reason = is_controlnet_or_adapter(file_path)

    if not is_mine:
        return {
            "status": "SKIP",
            "reason": reason,
            "filename": filename,
            "size_gb": size_gb
        }

    # 2. Read Keys & Metadata
    keys = read_safetensors_keys(file_path)
    metadata = read_safetensors_metadata(file_path)

    # 3. Detection (nach Priorität!)
    base_model = detect_base_model(keys, filename)
    control_type = detect_control_type(keys, metadata, filename)
    precision = detect_precision(metadata, filename)
    version = detect_version(filename)

    # 4. Target Folder
    target_folder = "t2i_adapter" if detected_type == "t2i_adapter" else "controlnet"

    file_info = {
        "status": "PROCESSED",
        "detected_type": detected_type,
        "base_model": base_model,
        "control_type": control_type,
        "precision": precision,
        "version": version,
        "target_folder": target_folder,
        "size_gb": size_gb
    }

    file_info["proper_name"] = generate_proper_name(file_info)
    return file_info


# ============================================================================
# EXPORTABLE FUNCTIONS (for all_modules.py batch processing)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads and return file list (for batch processing)

    Returns:
        dict with 'files' (list of file info dicts) and 'skipped' (list)
    """
    files_to_install = []
    skipped = []

    # Scan downloads (recursive)
    all_files = []
    for root, dirs, files in os.walk(downloads_path):
        for item in files:
            if item.endswith(".safetensors") or item.endswith(".pth"):
                file_path = os.path.join(root, item)
                all_files.append(file_path)

    # Analyze each file
    for file_path in all_files:
        filename = os.path.basename(file_path)
        size_mb = os.path.getsize(file_path) / (1024**2)
        size_gb = size_mb / 1024

        result = analyze_file(file_path)

        if result["status"] == "SKIP":
            skipped.append({
                'filename': filename,
                'reason': result['reason'],
                'size_gb': size_gb
            })
        else:
            files_to_install.append({
                'path': file_path,
                'filename': filename,
                'size_gb': size_gb,
                'result': result
            })

    return {
        'module_name': 'ControlNet & T2I-Adapter',
        'files': files_to_install,
        'skipped': skipped
    }


# ============================================================================
# MODUS A - INSTALLATION (Standalone mode)
# ============================================================================

def modus_a_installation(downloads_path):
    """Modus A: Installation von downloads/ nach models/controlnet/ + models/t2i_adapter/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="CONTROLNET & T2I-ADAPTER (INSTALLATION)",
        downloads_path=downloads_path,
        extensions="*.safetensors, *.pth",
        module_type="ControlNet/T2I-Adapter",
        target_folders="controlnet/, t2i_adapter/"
    )

    # Scanne downloads/ (recursive)
    files = []
    for root, dirs, filenames in os.walk(downloads_path):
        for item in filenames:
            if item.endswith(".safetensors") or item.endswith(".pth"):
                file_path = os.path.join(root, item)
                files.append(file_path)

    if not files:
        print_no_files_found("ControlNet/T2I-Adapter files")
        return

    # ========================================================================
    # PHASE 1: PREVIEW - Analyze all files (silently collect)
    # ========================================================================
    files_to_install = []
    skipped = []

    for file_path in files:
        filename = os.path.basename(file_path)
        size_mb = os.path.getsize(file_path) / (1024**2)
        size_gb = size_mb / 1024

        result = analyze_file(file_path)

        if result["status"] == "SKIP":
            skipped.append({'filename': filename, 'size_mb': size_mb})
        else:
            type_display = "T2I-Adapter" if result['detected_type'] == "t2i_adapter" else "ControlNet"
            files_to_install.append({
                'path': file_path,
                'filename': filename,
                'result': result,
                'detected_str': f"{type_display}, {result['base_model']}, {result['control_type']}, {result['precision']}",
                'size_mb': size_mb,
                'size_gb': size_gb
            })

    # ========================================================================
    # ANALYSIS (using shared helper with colors)
    # ========================================================================
    print_analysis(len(files), len(files_to_install), len(skipped))

    # ========================================================================
    # SKIPPED SECTION (using shared helper)
    # ========================================================================
    print_skipped_section(
        skipped_files=skipped,
        skip_reason="Not a ControlNet/T2I-Adapter (no matching keys found)",
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
        r = file_info['result']
        print_preview_item(
            index=i,
            filename=file_info['filename'],
            size_mb=file_info['size_mb'],
            detected_info=file_info['detected_str'],
            target_path=f"{r['target_folder']}/{r['proper_name']}"
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
        result = file_info['result']

        target_dir = T2I_ADAPTER_PATH if result['target_folder'] == "t2i_adapter" else CONTROLNET_PATH
        target_path = target_dir / result['proper_name']

        success, final_path, msg = handle_duplicate_move(
            file_path,
            target_path,
            expected_target_name=result['proper_name'],
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

def modus_b_reinstall(controlnet_path, t2i_adapter_path, scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Reinstall/Check von models/controlnet/ und models/t2i_adapter/

    Args:
        controlnet_path: Path to controlnet/
        t2i_adapter_path: Path to t2i_adapter/
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="CONTROLNET & T2I-ADAPTER",
        folders="controlnet/, t2i_adapter/",
        extensions="*.safetensors, *.pth",
        module_type="ControlNet/T2I-Adapter",
        target_folders="controlnet/, t2i_adapter/",
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
                result = analyze_file(str(file_path))

                if result["status"] != "SKIP":
                    filename = file_path.name
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    type_display = "T2I-Adapter" if result['detected_type'] == "t2i_adapter" else "ControlNet"
                    print(f"[RESCUE] Found misplaced {type_display}: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                    target_folder = result["target_folder"]
                    proper_name = result["proper_name"]
                    target_dir = T2I_ADAPTER_PATH if target_folder == "t2i_adapter" else CONTROLNET_PATH
                    target_path = target_dir / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        file_path,
                        target_path,
                        expected_target_name=proper_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    if success:
                        print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: {target_folder}/{final_path.name}")
                        remove_misplaced_file(file_path)
                        rescued += 1
                    else:
                        print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                    print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced file(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    files = []

    if os.path.exists(controlnet_path):
        for file_path in controlnet_path.rglob("*.safetensors"):
            if file_path.is_file():
                files.append(("controlnet", file_path))

    if os.path.exists(t2i_adapter_path):
        for file_path in t2i_adapter_path.rglob("*.safetensors"):
            if file_path.is_file():
                files.append(("t2i_adapter", file_path))

    if not files:
        print_no_files_found("ControlNet/T2I-Adapter files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []
    renamed = 0

    for current_folder, file_path in files:
        filename = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        result = analyze_file(file_path)

        if result["status"] == "SKIP":
            if scan_only:
                add_misplaced_file(Path(file_path))
            else:
                problems_list.append(('misplaced', filename, file_size_mb, result['reason'], current_folder))
            continue

        if scan_only:
            continue

        detected_type = result["detected_type"]
        target_folder = result["target_folder"]
        proper_name = result["proper_name"]
        current_name = os.path.basename(file_path)

        type_display = "T2I-Adapter" if detected_type == "t2i_adapter" else "ControlNet"
        detected_str = f"{type_display}, {result['base_model']}, {result['control_type']}, {result['precision']}"

        if current_folder != target_folder:
            problems_list.append(('wrong_folder', filename, file_size_mb, detected_str, current_folder, target_folder, proper_name, file_path))
        elif current_name != proper_name:
            problems_list.append(('wrong_name', filename, file_size_mb, detected_str, current_folder, proper_name, file_path))
        else:
            correct_files.append((current_folder, filename, file_size_mb))

    # ========================================================================
    # SCAN-ONLY MODE: Just return after building queue
    # ========================================================================
    if scan_only:
        return

    # ========================================================================
    # ANALYSIS (using shared helper with colors)
    # ========================================================================
    print_analysis_b(len(files), len(correct_files), len(problems_list))

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
                    warning=f"Not a ControlNet/T2I-Adapter: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, detected_str, curr_folder, target_folder, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=detected_str,
                    target_path=f"{target_folder}/{proper_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    target_dir = T2I_ADAPTER_PATH if target_folder == "t2i_adapter" else CONTROLNET_PATH
                    target_path = target_dir / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=proper_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    print_fix_result(success, "Moved and renamed to correct location" if success else msg)
                    if success:
                        renamed += 1

            elif problem_type == 'wrong_name':
                _, fname, size_mb, detected_str, folder, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=detected_str,
                    target_path=f"{folder}/{proper_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    old_path = Path(fpath)
                    new_path = old_path.parent / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        old_path,
                        new_path,
                        expected_target_name=proper_name,
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Modul 5 - ControlNet & T2I-Adapter")
        print("=" * 80)
        print()
        print("Usage:")
        print("  python modul5_controlnet.py A              # Modus A: Installation")
        print("  python modul5_controlnet.py B [--scan-only] # Modus B: Reinstall/Check")
        print()
        print("Options:")
        print("  --scan-only  PASS 1: Nur Queue aufbauen (nur Modus B)")
        print()
        print("Modus B - 2-Pass System:")
        print("  PASS 1: python modul5_controlnet.py B --scan-only")
        print("          Scannt alle Dateien, baut Queue auf, keine User-Fragen")
        print("  PASS 2: python modul5_controlnet.py B")
        print("          Zeigt Probleme, fragt User, führt Änderungen SOFORT aus")
        print()
        sys.exit(1)

    mode = sys.argv[1].upper()
    scan_only = "--scan-only" in sys.argv
    preview = "--preview" in sys.argv
    batch = "--batch" in sys.argv

    if mode == "A":
        modus_a_installation(DOWNLOADS_PATH)
    elif mode == "B":
        modus_b_reinstall(CONTROLNET_PATH, T2I_ADAPTER_PATH, scan_only=scan_only, batch_mode=batch, preview_mode=preview)
    else:
        print(f"Unbekannter Modus: {mode}")
        print("Nutze 'A' für Installation oder 'B' für Reinstall/Check")
        sys.exit(1)
