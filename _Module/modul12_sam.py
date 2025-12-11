#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 12 - SAM (Segment Anything Model)

Zweck:
- Installiert SAM Models (Segment Anything) aus downloads/
- Prüft und korrigiert bestehende SAM Models in models/sams/

Module Boundary:
- SAM (Meta AI): sam_vit_h, sam_vit_l, sam_vit_b, mobile_sam
- SAM-HQ (SysCV): sam_hq_vit_h, sam_hq_vit_l, sam_hq_vit_b
- Extensions: .pth, .pt
- Größen: 39 MB (Mobile) - 2.6 GB (ViT-H)

WICHTIG:
- Nur ~7 offizielle SAM-Dateien existieren!
- Namen sind standardisiert (Meta AI / SysCV)
- Filename-Detection ausreichend
- KEINE Umbenennung (Original-Namen beibehalten!)

Dokumentation: 18_Modul18_SAM.md
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

# Import shared utilities (includes torch with warning suppression)
from shared_utils import (
    DOWNLOADS_DIR,
    SAMS_DIR,
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

# Bekannte SAM-Dateinamen (offizielle Releases)
KNOWN_SAM_PATTERNS = [
    'sam_vit_h',       # SAM ViT-Huge
    'sam_vit_l',       # SAM ViT-Large
    'sam_vit_b',       # SAM ViT-Base
    'mobile_sam',      # SAM Mobile
    'sam_hq_vit_h',    # SAM-HQ ViT-Huge
    'sam_hq_vit_l',    # SAM-HQ ViT-Large
    'sam_hq_vit_b',    # SAM-HQ ViT-Base
]

# ============================================================================
# MODULE BOUNDARY
# ============================================================================

def is_sam_model(file_path):
    """
    Prüft ob Datei ein SAM-Model ist (Module Boundary)

    Args:
        file_path: Path zur Datei

    Returns:
        tuple: (is_sam: bool, reason: str)
    """
    # 1. Extension Check
    if file_path.suffix.lower() not in ['.pth', '.pt']:
        return False, f"Wrong extension: {file_path.suffix}"

    # 2. Size Check (Grob-Filter)
    size_mb = file_path.stat().st_size / (1024 * 1024)

    # SAM Größen:
    # Mobile: ~39 MB
    # ViT-B: ~358 MB
    # ViT-L: ~1250 MB
    # ViT-H: ~2400 MB

    if size_mb < 30:
        return False, f"Too small ({size_mb:.0f} MB) for SAM"

    if size_mb > 3000:
        return False, f"Too large ({size_mb:.0f} MB) for SAM"

    # 3. Filename Check (sehr zuverlässig bei SAM!)
    filename_lower = file_path.name.lower()

    # Check bekannte SAM-Patterns
    for pattern in KNOWN_SAM_PATTERNS:
        if pattern in filename_lower:
            return True, f"SAM pattern matched: {pattern}"

    # Kein bekanntes Pattern → Nicht SAM
    return False, "No known SAM pattern in filename"

# ============================================================================
# MODUS A - INSTALLATION
# ============================================================================

def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="SAM (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.pth, *.pt",
        module_type="SAM",
        target_folders="sams/"
    )

    all_files = list(DOWNLOADS_DIR.glob("*.pth"))
    all_files += list(DOWNLOADS_DIR.glob("*.pt"))

    if not all_files:
        print_no_files_found("SAM files")
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

        is_sam, reason = is_sam_model(file_path)
        if not is_sam:
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
        skip_reason="Not a SAM model",
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
            target_path=f"sams/{file_info['filename']}"
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

    SAMS_DIR.mkdir(parents=True, exist_ok=True)

    installed = 0
    errors = 0
    collisions = 0

    for idx, file_info in enumerate(files_to_install, 1):
        source_path = file_info['path']
        filename = file_info['filename']
        target_path = SAMS_DIR / filename

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
    """Modus B: Check bestehender SAM Models"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="SAM",
        folders="sams/",
        extensions="*.pth, *.pt",
        module_type="SAM",
        target_folders="sams/",
        preview_mode=preview_mode
    )

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode, not preview/scan)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        for file_path in misplaced:
            is_sam, reason = is_sam_model(file_path)
            if not is_sam:
                continue

            filename = file_path.name
            file_size_mb = file_path.stat().st_size / (1024 * 1024)

            print(f"[RESCUE] Found misplaced SAM: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

            target_path = SAMS_DIR / file_path.name

            success, final_path, msg = handle_duplicate_move(
                file_path,
                target_path,
                expected_target_name=file_path.name,
                mode="B",
                keep_source_option=False,
                dry_run=False
            )

            if success:
                print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: sams/{final_path.name}")
                remove_misplaced_file(file_path)
                rescued += 1
            else:
                print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
            print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced SAM model(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDER
    # ========================================================================
    all_files = list(SAMS_DIR.glob("*.pth"))
    all_files += list(SAMS_DIR.glob("*.pt"))

    if not all_files:
        print_no_files_found("SAM files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []

    for file_path in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        is_sam, reason = is_sam_model(file_path)
        if not is_sam:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason))
            continue

        if scan_only:
            continue

        # SAM files don't need renaming - always correct
        correct_files.append(("sams", filename, file_size_mb))

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
                warning=f"Not a SAM model: {reason}"
            )

# ============================================================================
# BATCH INTEGRATION (for all_modules.py)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scannt downloads/ für SAM Dateien (für Batch Mode)

    Args:
        downloads_path: Path object zu downloads/

    Returns:
        dict: {
            'module_name': 'SAM',
            'files': [{'path': Path, 'filename': str, 'size_gb': float, 'new_name': str, 'target_folder': Path}, ...],
            'skipped': int
        }
    """
    result = {
        'module_name': 'SAM',
        'files': [],
        'skipped': 0
    }

    # Scan for .pth and .pt files
    all_files = list(downloads_path.glob("*.pth")) + list(downloads_path.glob("*.pt"))

    for file_path in all_files:
        is_mine, reason = is_sam_model(file_path)
        if not is_mine:
            result['skipped'] += 1
            continue

        # SAM models keep original names - no analysis needed
        size_gb = file_path.stat().st_size / (1024**3)
        result['files'].append({
            'path': file_path,
            'filename': file_path.name,
            'size_gb': size_gb,
            'new_name': file_path.name,  # Keep original name
            'target_folder': SAM_DIR
        })

    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    scan_only = "--scan-only" in sys.argv

    if mode.upper() == "A":
        modus_a()
    elif mode.upper() == "B":
        modus_b(scan_only=scan_only)
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python modul18_sam.py [A|B] [--scan-only]")
        sys.exit(1)
