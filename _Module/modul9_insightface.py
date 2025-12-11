#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 9 - InsightFace (Face Recognition & Face Swap)

Organisiert InsightFace ONNX Models nach Hybrid-Struktur.

Unterstützt:
- Modus A: Installation aus downloads/
- Modus B: Check/Fix bestehender InsightFace Models

WICHTIG: HYBRID STRUKTUR (Original-Namen beibehalten!)
- models/insightface/                 → Standalone Models (inswapper, simswap)
- models/insightface/models/buffalo_l/ → Model Pack (5 Files)
- models/insightface/models/antelopev2/ → Model Pack (5 Files)

Namenskonvention:
    KEINE UMBENENNUNG! Original-Namen beibehalten.
    ComfyUI Custom Nodes haben hardcoded Filenames.

Beispiele (Original):
    inswapper_128.onnx
    inswapper_128.fp16.onnx
    simswap_512_beta.onnx
    buffalo_l/det_10g.onnx
    buffalo_l/w600k_r50.onnx
    antelopev2/scrfd_10g_bnkps.onnx
"""

import sys
from pathlib import Path

# ============================================================================
# PATH SETUP - Für Unterordner-Struktur
# ============================================================================
_SCRIPT_DIR = Path(__file__).parent
_SHARED_DIR = _SCRIPT_DIR.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

# ============================================================================
# IMPORTS FROM SHARED_UTILS
# ============================================================================

from shared_utils import (
    DOWNLOADS_DIR,              # Source für Mode A
    MODELS_DIR,                 # Base models/
    INSIGHTFACE_DIR,            # Target: models/insightface/
    handle_duplicate_move,      # Duplicate Handling
    read_misplaced_files,       # Queue System
    add_misplaced_file,
    remove_misplaced_file,
    ask_keep_or_delete,         # User Input (Mode A)
    ask_confirm_installation,   # User Input (Mode A)
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
# KNOWN INSIGHTFACE MODELS (für Module Boundary)
# ============================================================================

# Standalone Models (im Root)
STANDALONE_PATTERNS = ['inswapper', 'simswap']

# Model Pack Files (in models/<pack_name>/)
PACK_FILENAMES = [
    # Buffalo Packs (buffalo_l, buffalo_m, buffalo_s, buffalo_sc)
    'det_10g.onnx',
    '1k3d68.onnx',
    '2d106det.onnx',
    'genderage.onnx',
    'w600k_r50.onnx',

    # Antelopev2 Pack (unique files)
    'scrfd_10g_bnkps.onnx',
    'glintr100.onnx',
]

# Exclusion Patterns (andere ONNX Models)
EXCLUSION_PATTERNS = ['yolox', 'gpen', 'grounding', 'sam_']

# ============================================================================
# MODULE BOUNDARY CHECK
# ============================================================================

def is_insightface(file_path):
    """Prüft ob ONNX Datei zu InsightFace gehört (Module Boundary)

    Detection basiert auf:
    - Extension (.onnx)
    - Folder Context (bereits in insightface/)
    - Known Filename Patterns
    - Exclusion Patterns (YOLOX, Face Restore, etc.)

    Returns:
        tuple: (is_match, reason)
            is_match: bool - True wenn InsightFace
            reason: str - Grund (für Logging)
    """
    # Extension Check
    if file_path.suffix.lower() != '.onnx':
        return False, "Not .onnx file"

    # Folder Context Check (bereits in insightface/)
    if 'insightface' in str(file_path).lower():
        return True, "Already in insightface folder"

    filename_lower = file_path.name.lower()

    # Exclusion Patterns (YOLOX, Face Restore, etc.)
    if any(pattern in filename_lower for pattern in EXCLUSION_PATTERNS):
        return False, f"Excluded pattern detected: {filename_lower}"

    # Known Standalone Patterns
    if any(pattern in filename_lower for pattern in STANDALONE_PATTERNS):
        return True, f"Known Standalone pattern: {filename_lower}"

    # Known Pack Filenames
    if filename_lower in PACK_FILENAMES:
        return True, f"Known Pack file: {filename_lower}"

    # Unknown ONNX file
    return False, "Unknown ONNX file - not InsightFace"

# ============================================================================
# MODEL TYPE DETECTION
# ============================================================================

def detect_model_type(file_path):
    """Erkennt ob Standalone oder Pack-Teil

    Returns:
        str: "Standalone", "Pack", or "Unknown"
    """
    filename_lower = file_path.name.lower()

    # Standalone Models
    if any(p in filename_lower for p in STANDALONE_PATTERNS):
        return "Standalone"

    # Pack Models
    if filename_lower in PACK_FILENAMES:
        return "Pack"

    return "Unknown"

# ============================================================================
# PACK NAME DETECTION (für Pack-Dateien)
# ============================================================================

def detect_pack_name(file_path):
    """Erkennt zu welchem Pack eine Datei gehört

    Heuristik:
    - Folder Context (wenn bereits installiert)
    - Unique Filenames (nur in antelopev2)
    - Default zu buffalo_l (häufigster Pack)

    Returns:
        str: Pack name ("buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2", or "unknown")
    """
    # Priorität 1: Folder Context (wenn bereits in models/<pack_name>/)
    parent = file_path.parent.name.lower()
    if parent in ['buffalo_l', 'buffalo_m', 'buffalo_s', 'buffalo_sc', 'antelopev2']:
        return parent

    # Priorität 2: Unique Filenames
    filename_lower = file_path.name.lower()

    # Antelopev2 unique files
    if filename_lower in ['scrfd_10g_bnkps.onnx', 'glintr100.onnx']:
        return 'antelopev2'

    # Buffalo_* common files (ambiguous) → Default zu buffalo_l
    if filename_lower in ['det_10g.onnx', '1k3d68.onnx', '2d106det.onnx', 'genderage.onnx', 'w600k_r50.onnx']:
        return 'buffalo_l'  # DEFAULT (häufigster Pack)

    return 'unknown'

# ============================================================================
# PRECISION DETECTION (nur für Logging)
# ============================================================================

def detect_precision(file_path):
    """Erkennt FP16 vs FP32 (nur für Logging, KEINE Umbenennung)

    Returns:
        str: "FP16", "FP32", or "Unknown"
    """
    filename_lower = file_path.name.lower()

    # Explicit FP16
    if 'fp16' in filename_lower or 'float16' in filename_lower:
        return "FP16"

    # Explicit FP32
    if 'fp32' in filename_lower or 'float32' in filename_lower:
        return "FP32"

    # Size Heuristic (inswapper example)
    size_mb = file_path.stat().st_size / (1024 * 1024)

    if 'inswapper' in filename_lower:
        if size_mb < 300:  # ~264 MB = FP16
            return "FP16"
        else:              # ~529 MB = FP32
            return "FP32"

    # Default: Unknown (likely FP32)
    return "Unknown"

# ============================================================================
# ANALYZE FILE (für Modus A + B)
# ============================================================================

def analyze_file(file_path):
    """Analysiert eine InsightFace Datei

    Returns:
        dict mit:
        - status: "OK", "SKIP", etc.
        - model_type: "Standalone" or "Pack"
        - pack_name: Pack name (wenn Pack)
        - precision: "FP16", "FP32", "Unknown"
        - target_folder: Wo soll die Datei hin
        - proper_name: Original Name (keine Umbenennung!)
        - reason: Grund (bei SKIP)
        - ambiguous_pack: bool (bei Pack-Zuordnung unsicher?)
    """
    filename = file_path.name

    # Module Boundary Check
    is_mine, reason = is_insightface(file_path)
    if not is_mine:
        return {
            'status': 'SKIP',
            'reason': reason,
            'proper_name': filename
        }

    # Model Type Detection
    model_type = detect_model_type(file_path)

    if model_type == "Unknown":
        return {
            'status': 'SKIP',
            'reason': 'Unknown InsightFace model type',
            'proper_name': filename
        }

    # Precision Detection (nur für Info)
    precision = detect_precision(file_path)

    # Target Folder bestimmen
    ambiguous_pack = False

    if model_type == "Standalone":
        # Standalone → Root
        target_folder = INSIGHTFACE_DIR
        pack_name = None

    elif model_type == "Pack":
        # Pack File → models/<pack_name>/
        pack_name = detect_pack_name(file_path)

        # Check if ambiguous (defaulted to buffalo_l)
        filename_lower = file_path.name.lower()
        if pack_name == 'buffalo_l' and filename_lower in ['det_10g.onnx', '1k3d68.onnx', '2d106det.onnx', 'genderage.onnx', 'w600k_r50.onnx']:
            # Diese Dateien gibt es in ALLEN Buffalo Packs
            ambiguous_pack = True

        target_folder = INSIGHTFACE_DIR / "models" / pack_name

    # KEINE UMBENENNUNG! Original Name beibehalten
    proper_name = filename

    return {
        'status': 'OK',
        'model_type': model_type,
        'pack_name': pack_name,
        'precision': precision,
        'target_folder': target_folder,
        'proper_name': proper_name,
        'ambiguous_pack': ambiguous_pack
    }

# ============================================================================
# MODUS A - INSTALLATION
# ============================================================================

def modus_a(dry_run=False):
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="INSIGHTFACE (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.onnx",
        module_type="InsightFace",
        target_folders="insightface/, insightface/models/"
    )

    all_files = sorted(DOWNLOADS_DIR.glob("*.onnx"))

    if not all_files:
        print_no_files_found("InsightFace files")
        return

    # ========================================================================
    # PHASE 1: PREVIEW - Analyze all files (silently collect)
    # ========================================================================
    files_to_install = []
    skipped = []

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024**2)
        size_gb = size_mb / 1024

        result = analyze_file(file_path)

        if result['status'] == 'SKIP':
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # Build detected string
        if result['model_type'] == "Standalone":
            detected_str = f"Standalone, {result['precision']}"
        elif result['model_type'] == "Pack":
            pack_info = result['pack_name']
            if result['ambiguous_pack']:
                pack_info += " (assumed)"
            detected_str = f"Pack ({pack_info}), {result['precision']}"
        else:
            detected_str = result['model_type']

        try:
            target_relative = result['target_folder'].relative_to(MODELS_DIR)
        except ValueError:
            target_relative = result['target_folder']

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'result': result,
            'detected_str': detected_str,
            'target_relative': target_relative,
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
        skip_reason="Not an InsightFace model",
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
            target_path=f"{file_info['target_relative']}/{r['proper_name']}"
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
        source_path = file_info['path']
        result = file_info['result']
        target_path = result['target_folder'] / result['proper_name']

        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        success, final_path, msg = handle_duplicate_move(
            source_path,
            target_path,
            mode="A",
            keep_source_option=keep_source,
            dry_run=dry_run
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
# MODUS B - CHECK/FIX
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check bestehende InsightFace Dateien

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes) - used by all_modules.py
        preview_mode: Show problems only, no execution, no prompts - used by all_modules.py
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="INSIGHTFACE",
        folders="insightface/, insightface/models/",
        extensions="*.onnx",
        module_type="InsightFace",
        target_folders="insightface/, insightface/models/",
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
                is_mine, reason = is_insightface(file_path)
                if not is_mine:
                    continue

                filename = file_path.name
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                print(f"[RESCUE] Found misplaced InsightFace: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                result = analyze_file(file_path)

                if result['status'] == 'SKIP':
                    print(f"         {Colors.YELLOW}WARNING{Colors.RESET} {result['reason']}")
                    print()
                    continue

                target_path = result['target_folder'] / result['proper_name']
                target_path.parent.mkdir(parents=True, exist_ok=True)

                success, final_path, msg = handle_duplicate_move(
                    file_path,
                    target_path,
                    mode="B",
                    keep_source_option=False,
                    dry_run=False
                )

                if success:
                    try:
                        target_rel = result['target_folder'].relative_to(MODELS_DIR)
                    except ValueError:
                        target_rel = result['target_folder']
                    print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: {target_rel}/{final_path.name}")
                    remove_misplaced_file(file_path)
                    rescued += 1
                else:
                    print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced InsightFace file(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    root_files = list(INSIGHTFACE_DIR.glob("*.onnx")) if INSIGHTFACE_DIR.exists() else []
    models_subdir = INSIGHTFACE_DIR / "models"
    pack_files = list(models_subdir.rglob("*.onnx")) if models_subdir.exists() else []
    all_files = root_files + pack_files

    if not all_files:
        print_no_files_found("InsightFace files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []
    moved = 0

    for file_path in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        is_mine, reason = is_insightface(file_path)
        if not is_mine:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('not_insightface', filename, file_size_mb, reason))
            continue

        result = analyze_file(file_path)

        if result['status'] == 'SKIP':
            if not scan_only:
                problems_list.append(('unknown_type', filename, file_size_mb, result['reason']))
            continue

        if scan_only:
            continue

        expected_folder = result['target_folder']
        current_folder = file_path.parent

        if current_folder != expected_folder:
            try:
                curr_rel = current_folder.relative_to(INSIGHTFACE_DIR)
            except ValueError:
                curr_rel = current_folder
            try:
                exp_rel = expected_folder.relative_to(INSIGHTFACE_DIR)
            except ValueError:
                exp_rel = expected_folder
            problems_list.append(('wrong_folder', filename, file_size_mb, str(curr_rel), str(exp_rel), expected_folder, file_path))
        else:
            try:
                relative_path = file_path.relative_to(INSIGHTFACE_DIR)
            except ValueError:
                relative_path = file_path
            correct_files.append((str(relative_path.parent), filename, file_size_mb))

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

            if problem_type == 'not_insightface':
                _, fname, size_mb, reason = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=None,
                    target_path="Added to rescue queue for other modules",
                    warning=f"Not an InsightFace model: {reason}"
                )

            elif problem_type == 'unknown_type':
                _, fname, size_mb, reason = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=None,
                    target_path="Manual check recommended",
                    warning=f"Unknown model type: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, curr_rel, exp_rel, exp_folder, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"Current: {curr_rel}/",
                    target_path=f"{exp_rel}/{fname}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    new_path = exp_folder / fname
                    new_path.parent.mkdir(parents=True, exist_ok=True)

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        new_path,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    print_fix_result(success, "Moved to correct location" if success else msg)
                    if success:
                        moved += 1

# ============================================================================
# BATCH INTEGRATION (for all_modules.py)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scannt downloads/ für InsightFace Dateien (für Batch Mode)

    Args:
        downloads_path: Path object zu downloads/

    Returns:
        dict: {
            'module_name': 'InsightFace',
            'files': [{'path': Path, 'filename': str, 'size_gb': float, 'result': dict}, ...],
            'skipped': int
        }
    """
    result = {
        'module_name': 'InsightFace',
        'files': [],
        'skipped': 0
    }

    # Scan for .onnx files
    all_files = list(downloads_path.glob("*.onnx"))

    for file_path in all_files:
        is_mine, reason = is_insightface(file_path)
        if not is_mine:
            result['skipped'] += 1
            continue

        # Analyze
        analysis = analyze_file(file_path)
        if analysis['status'] == 'SKIP':
            result['skipped'] += 1
            continue

        # Add to results
        size_gb = file_path.stat().st_size / (1024**3)
        result['files'].append({
            'path': file_path,
            'filename': file_path.name,
            'size_gb': size_gb,
            'result': analysis
        })

    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modul 11 - InsightFace")
    parser.add_argument('mode', choices=['A', 'B'], help='Modus: A (Install) oder B (Check/Fix)')
    parser.add_argument('--dry-run', action='store_true', help='Nur simulieren (Modus A)')
    parser.add_argument('--scan-only', action='store_true', help='PASS 1: Nur Queue aufbauen (Modus B)')
    parser.add_argument('--batch', action='store_true', help='Batch Mode: Skip prompts, execute fixes (Modus B)')
    parser.add_argument('--preview', action='store_true', help='Preview Mode: Show problems only (Modus B)')

    args = parser.parse_args()

    if args.mode == 'A':
        modus_a(dry_run=args.dry_run)
    elif args.mode == 'B':
        modus_b(scan_only=args.scan_only, batch_mode=args.batch, preview_mode=args.preview)
