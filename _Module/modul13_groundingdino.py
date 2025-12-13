#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 13 - Grounding DINO

Zweck:
- Installiert Grounding DINO Models (Object Detection) aus downloads/
- Prüft und korrigiert bestehende Grounding DINO Models in models/grounding-dino/

Module Boundary:
- Grounding DINO (IDEA Research): groundingdino_swint_ogc, groundingdino_swinb_cogcoor
- Extensions: .pth, .pt (Checkpoints) + .cfg.py (Config Files)
- Größen: ~662 MB (SwinT) - ~895 MB (SwinB)

WICHTIG:
- Nur ~4 offizielle Grounding DINO-Dateien existieren (v1.0)!
- .pth + .cfg.py PAARE (Config ist nötig für Model Loading!)
- Namen sind standardisiert (IDEA Research)
- Filename-Detection ausreichend
- KEINE Umbenennung (Original-Namen beibehalten!)

Dokumentation: 19_Modul19_GroundingDINO.md
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

# Import shared utilities
from shared_utils import (
    DOWNLOADS_DIR,
    GROUNDING_DINO_DIR,
    handle_duplicate_move,
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
# CONSTANTS
# ============================================================================

# Bekannte Grounding DINO Dateinamen-Patterns (offizielle Releases)
KNOWN_GROUNDINGDINO_PATTERNS = [
    'groundingdino_swint',     # SwinT variants
    'groundingdino_swinb',     # SwinB variants
    'groundingdino_swinl',     # SwinL variants (hypothetisch)
]

# Config File Patterns
CONFIG_FILE_PATTERNS = [
    'GroundingDINO_SwinT',
    'GroundingDINO_SwinB',
    'GroundingDINO_SwinL',
]

# ============================================================================
# MODULE BOUNDARY
# ============================================================================

def is_grounding_dino(file_path):
    """
    Prüft ob Datei ein Grounding DINO Model oder Config ist (Module Boundary)

    Args:
        file_path: Path zur Datei

    Returns:
        tuple: (is_grounding_dino: bool, reason: str)
    """
    # 1. Extension Check
    is_checkpoint = file_path.suffix.lower() in ['.pth', '.pt']
    is_config = file_path.name.endswith('.cfg.py')

    if not (is_checkpoint or is_config):
        return False, f"Wrong extension: {file_path.suffix}"

    # 2. Handle Config Files (.cfg.py)
    if is_config:
        filename_lower = file_path.name.lower()

        # Check bekannte Config-Patterns
        for pattern in CONFIG_FILE_PATTERNS:
            if pattern.lower() in filename_lower:
                return True, f"Grounding DINO config: {pattern}"

        # Zusätzlich: Prüfe Datei-Inhalt (optional, aber zuverlässig)
        try:
            content = file_path.read_text(encoding='utf-8')
            if 'modelname = "groundingdino"' in content or "modelname = 'groundingdino'" in content:
                return True, "Grounding DINO config (content verified)"
        except:
            pass

        return False, "No known Grounding DINO config pattern"

    # 3. Handle Checkpoint Files (.pth/.pt)
    # Size Check (Grob-Filter)
    size_mb = file_path.stat().st_size / (1024 * 1024)

    # Grounding DINO Größen:
    # SwinT: ~662 MB
    # SwinB: ~895 MB
    # SwinL: >1000 MB (hypothetisch)

    if size_mb < 500:
        return False, f"Too small ({size_mb:.0f} MB) for Grounding DINO"

    if size_mb > 2000:
        return False, f"Too large ({size_mb:.0f} MB) for Grounding DINO"

    # 4. Filename Check (sehr zuverlässig bei Grounding DINO!)
    filename_lower = file_path.name.lower()

    # MUSS "groundingdino" enthalten
    if "groundingdino" not in filename_lower:
        return False, "Missing 'groundingdino' in filename"

    # MUSS "swin" enthalten (Backbone)
    if "swin" not in filename_lower:
        return False, "Missing 'swin' in filename"

    # Check bekannte Grounding DINO-Patterns
    for pattern in KNOWN_GROUNDINGDINO_PATTERNS:
        if pattern in filename_lower:
            return True, f"Grounding DINO pattern matched: {pattern}"

    # Filename hat groundingdino + swin → akzeptieren
    return True, "Grounding DINO (groundingdino + swin in filename)"

# ============================================================================
# MODUS A - INSTALLATION
# ============================================================================

def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="GROUNDING DINO (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.pth, *.pt",
        module_type="Grounding DINO",
        target_folders="grounding-dino/"
    )

    all_files = list(DOWNLOADS_DIR.glob("**/*.pth"))  # recursive
    all_files += list(DOWNLOADS_DIR.glob("**/*.pt"))  # recursive
    all_files += list(DOWNLOADS_DIR.glob("**/*.cfg.py"))  # recursive

    if not all_files:
        print_no_files_found("Grounding DINO files")
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

        is_gdino, reason = is_grounding_dino(file_path)
        if not is_gdino:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'detected_str': reason,
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
        skip_reason="Not a Grounding DINO model",
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
            target_path=f"grounding-dino/{file_info['filename']}"
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

    GROUNDING_DINO_DIR.mkdir(parents=True, exist_ok=True)

    installed = 0
    errors = 0
    collisions = 0

    for idx, file_info in enumerate(files_to_install, 1):
        source_path = file_info['path']
        filename = file_info['filename']
        target_path = GROUNDING_DINO_DIR / filename

        success, final_path, msg = handle_duplicate_move(
            source_path,
            target_path,
            expected_target_name=filename,
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
    """Modus B: Check bestehender Grounding DINO Files"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="GROUNDING DINO",
        folders="grounding-dino/",
        extensions="*.pth, *.pt",
        module_type="Grounding DINO",
        target_folders="grounding-dino/",
        preview_mode=preview_mode
    )

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode, not preview/scan)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        for file_path in misplaced:
            is_gdino, reason = is_grounding_dino(file_path)
            if not is_gdino:
                continue

            filename = file_path.name
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            print(f"[RESCUE] Found misplaced Grounding DINO: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

            target_path = GROUNDING_DINO_DIR / file_path.name

            success, final_path, msg = handle_duplicate_move(
                file_path,
                target_path,
                expected_target_name=file_path.name,
                mode="B",
                keep_source_option=False,
                dry_run=False
            )

            if success:
                print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: grounding-dino/{final_path.name}")
                remove_misplaced_file(file_path)
                rescued += 1
            else:
                print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
            print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced Grounding DINO file(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDER
    # ========================================================================
    all_files = list(GROUNDING_DINO_DIR.glob("**/*.pth"))  # recursive
    all_files += list(GROUNDING_DINO_DIR.glob("**/*.pt"))  # recursive

    if not all_files:
        print_no_files_found("Grounding DINO files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []

    for file_path in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        is_gdino, reason = is_grounding_dino(file_path)
        if not is_gdino:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason))
            continue

        if scan_only:
            continue

        correct_files.append(("grounding-dino", filename, file_size_mb))

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
            _, fname, size_mb, reason = problem
            print_problem_item(
                index=idx,
                total=len(problems_list),
                filename=fname,
                size_mb=size_mb,
                detected_info=None,
                target_path="Added to rescue queue for other modules",
                warning=f"Not a Grounding DINO: {reason}"
            )

# ============================================================================
# BATCH INTEGRATION (for all_modules.py)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scannt downloads/ für Grounding DINO Dateien (für Batch Mode)

    Args:
        downloads_path: Path object zu downloads/

    Returns:
        dict: {
            'module_name': 'Grounding DINO',
            'files': [{'path': Path, 'filename': str, 'size_gb': float, 'new_name': str, 'target_folder': Path}, ...],
            'skipped': int
        }
    """
    result = {
        'module_name': 'Grounding DINO',
        'files': [],
        'skipped': 0
    }

    # Scan for .pth, .pt, and .cfg.py files (recursive)
    all_files = (
        list(downloads_path.glob("**/*.pth")) +  # recursive
        list(downloads_path.glob("**/*.pt")) +  # recursive
        list(downloads_path.glob("**/*.cfg.py"))  # recursive
    )

    for file_path in all_files:
        is_mine, reason = is_grounding_dino(file_path)
        if not is_mine:
            result['skipped'] += 1
            continue

        # Grounding DINO keeps original names - no analysis needed
        size_gb = file_path.stat().st_size / (1024**3)
        result['files'].append({
            'path': file_path,
            'filename': file_path.name,
            'size_gb': size_gb,
            'new_name': file_path.name,  # Keep original name
            'target_folder': GROUNDING_DINO_DIR
        })

    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    scan_only = "--scan-only" in sys.argv
    batch_mode = "--batch" in sys.argv
    preview_mode = "--preview" in sys.argv

    if mode.upper() == "A":
        modus_a()
    elif mode.upper() == "B":
        modus_b(scan_only=scan_only, batch_mode=batch_mode, preview_mode=preview_mode)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python modul19_groundingdino.py [A|B] [--scan-only]")
        sys.exit(1)
