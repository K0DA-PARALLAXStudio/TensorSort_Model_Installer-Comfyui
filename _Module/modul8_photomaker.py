#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 8 - PhotoMaker
Face Identity Preservation Models für SDXL

PhotoMaker ermöglicht zero-shot personalization:
- User gibt 1+ Fotos einer Person als Input
- PhotoMaker lernt die Gesichtsidentität ohne Training
- Person kann dann in beliebigen Prompts verwendet werden

Base Model: SDXL-exclusive (nicht für SD1.5 oder Flux)
Format: PyTorch .bin (ZIP Archive mit ~1585 Dateien)
Ordner: models/photomaker/

Versionen:
- V1: photomaker-v1.bin (Dez 2023, älter)
- V2: photomaker-v2.bin (Juli 2024, neueste, verbesserte ID Fidelity)

Detection-Strategie:
- ROBUST: ZIP-Struktur Analyse (Datei-Count)
- PhotoMaker: ~1585 Dateien im Archive
- IP-Adapter: ~124 Dateien im Archive
- Unabhängig vom Filename!
"""

import sys
import zipfile
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
    PHOTOMAKER_DIR,
    MODELS_DIR,
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


def is_photomaker(file_path):
    """Prüft ob Datei ein PhotoMaker Model ist (Module Boundary)

    ROBUSTE Detection durch interne ZIP-Struktur Analyse!
    - .bin ist ZIP Archive mit PyTorch Tensors
    - PhotoMaker: ~1585 Dateien im Archive
    - IP-Adapter: ~124 Dateien im Archive
    - Unterscheidung durch Datei-Count (unabhängig vom Filename!)

    Args:
        file_path: Path object zur Datei

    Returns:
        tuple: (bool, str, dict)
            - is_match: True/False
            - reason: Type oder Skip-Grund
            - details: {'num_files': int, 'root_folder': str}
    """
    # 1. Extension Check
    if not file_path.name.endswith('.bin'):
        return False, "SKIP: Not .bin extension", {}

    # 2. Ist es ein ZIP Archive?
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            num_files = len(files)

            # 3. HAUPT-CHECK: Anzahl Dateien im Archive
            # PhotoMaker: ~1585 Dateien
            # IP-Adapter: ~124 Dateien
            if num_files < 500:
                return False, "SKIP: Too few files for PhotoMaker (likely IP-Adapter or other)", {}

            # 4. Root Folder Name Check (zusätzliche Sicherheit)
            if files:
                root_folder = files[0].split('/')[0].lower()

                # IP-Adapter hat "ip-adapter" als root folder
                if 'ip-adapter' in root_folder or 'ipadapter' in root_folder:
                    return False, "SKIP: IP-Adapter (Modul 12)", {}

                # PhotoMaker hat Training-Namen mit "insightface" oder "hynoid"
                has_photomaker_markers = ('insightface' in root_folder or
                                         'hynoid' in root_folder or
                                         'photomaker' in root_folder)

                if not has_photomaker_markers:
                    # Fallback: Filename check
                    if 'photomaker' not in file_path.name.lower():
                        return False, "SKIP: Not PhotoMaker structure", {}

            # ✅ Alle Checks bestanden - ist PhotoMaker!
            return True, "PhotoMaker", {'num_files': num_files, 'root_folder': root_folder if files else ''}

    except zipfile.BadZipFile:
        return False, "SKIP: Not a valid ZIP file", {}
    except Exception as e:
        return False, f"SKIP: Error reading file ({str(e)})", {}


def detect_version(filename):
    """Erkennt PhotoMaker Version aus Filename

    Nur 2 offizielle Versionen:
    - V1 (photomaker-v1.bin) - Dez 2023
    - V2 (photomaker-v2.bin) - Juli 2024

    Args:
        filename: Filename string

    Returns:
        str: "V1" oder "V2"
    """
    filename_lower = filename.lower()

    # V2 erkennen
    if 'v2' in filename_lower or '-v2' in filename_lower:
        return 'V2'

    # V1 erkennen
    if 'v1' in filename_lower or '-v1' in filename_lower:
        return 'V1'

    # Fallback: Wenn keine Version im Namen
    # Assume V2 (neueste Version, da V1 veraltet)
    return 'V2'


def analyze_file(file_path):
    """Analysiert PhotoMaker Datei und generiert Zielname

    Args:
        file_path: Path object zur Datei

    Returns:
        dict: {
            'status': 'OK' oder 'SKIP',
            'reason': Skip-Grund (bei SKIP),
            'version': 'V1' oder 'V2' (bei OK),
            'proper_name': Standardisierter Filename (bei OK),
            'target_folder': PHOTOMAKER_DIR (bei OK)
        }
    """
    # Module Boundary Check
    is_mine, reason, details = is_photomaker(file_path)

    if not is_mine:
        return {
            'status': 'SKIP',
            'reason': reason
        }

    # Version Detection
    version = detect_version(file_path.name)

    # Name Generation
    # Format: SDXL_PhotoMaker_V2.bin
    proper_name = f"SDXL_PhotoMaker_{version}.bin"

    return {
        'status': 'OK',
        'version': version,
        'proper_name': proper_name,
        'target_folder': PHOTOMAKER_DIR,
        'num_files': details.get('num_files', 0)
    }


def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="PHOTOMAKER (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.bin, *.safetensors",
        module_type="PhotoMaker",
        target_folders="photomaker/"
    )

    # ========================================================================
    # PHASE 1: PREVIEW - Analyze all files (silently collect)
    # ========================================================================
    all_files = list(DOWNLOADS_DIR.glob("**/*.bin"))  # recursive
    files_to_install = []
    skipped = []

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024**2)
        size_gb = size_mb / 1024

        is_mine, reason, details = is_photomaker(file_path)
        if not is_mine:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        result = analyze_file(file_path)
        if result['status'] == 'SKIP':
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'result': result,
            'detected_str': f"PhotoMaker {result['version']} (SDXL-only)",
            'size_mb': size_mb,
            'size_gb': size_gb
        })

    if not all_files:
        print_no_files_found("PhotoMaker files")
        return

    # ========================================================================
    # ANALYSIS (using shared helper with colors)
    # ========================================================================
    print_analysis(len(all_files), len(files_to_install), len(skipped))

    # ========================================================================
    # SKIPPED SECTION (using shared helper)
    # ========================================================================
    print_skipped_section(
        skipped_files=skipped,
        skip_reason="Not a PhotoMaker model",
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
            target_path=f"photomaker/{r['proper_name']}"
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

        success, final_path, msg = handle_duplicate_move(
            source_path,
            target_path,
            expected_target_name=result['proper_name'],
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

    # Final Summary (using shared helper)
    print_summary(len(files_to_install), installed, collisions, errors, keep_source)


def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check bestehender PhotoMaker Dateien

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes) - used by all_modules.py
        preview_mode: Show problems only, no execution, no prompts - used by all_modules.py
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="PHOTOMAKER",
        folders="photomaker/",
        extensions="*.bin, *.safetensors",
        module_type="PhotoMaker",
        target_folders="photomaker/",
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
                is_mine, reason, details = is_photomaker(file_path)
                if is_mine:
                    result = analyze_file(file_path)
                    if result['status'] == 'OK':
                        filename = file_path.name
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)

                        print(f"[RESCUE] Found misplaced PhotoMaker: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                        target_path = result['target_folder'] / result['proper_name']

                        success, final_path, msg = handle_duplicate_move(
                            file_path,
                            target_path,
                            expected_target_name=result['proper_name'],
                            mode="B",
                            keep_source_option=False
                        )

                        if success:
                            print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: photomaker/{result['proper_name']}")
                            remove_misplaced_file(file_path)
                            rescued += 1
                        else:
                            print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                        print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced PhotoMaker file(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDER
    # ========================================================================
    all_files = list(PHOTOMAKER_DIR.glob("**/*.bin"))  # recursive

    if not all_files:
        print_no_files_found("PhotoMaker files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []
    renamed = 0

    for file_path in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        is_mine, reason, details = is_photomaker(file_path)

        if not is_mine:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason))
            continue

        if scan_only:
            continue

        result = analyze_file(file_path)
        if result['status'] == 'SKIP':
            continue

        current_name = file_path.name
        proper_name = result['proper_name']

        if current_name != proper_name:
            problems_list.append(('wrong_name', filename, file_size_mb, result['version'], proper_name, file_path))
        else:
            correct_files.append(("photomaker", filename, file_size_mb))

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
                    warning=reason
                )

            elif problem_type == 'wrong_name':
                _, fname, size_mb, version, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"PhotoMaker {version}",
                    target_path=f"photomaker/{proper_name}",
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


def scan_for_batch(downloads_path):
    """Scannt downloads/ für PhotoMaker Dateien (für Batch Mode)

    Args:
        downloads_path: Path object zu downloads/

    Returns:
        dict: {
            'module_name': 'PhotoMaker',
            'files': [{'path': Path, 'filename': str, 'size_gb': float, 'result': dict}, ...],
            'skipped': int
        }
    """
    result = {
        'module_name': 'PhotoMaker',
        'files': [],
        'skipped': 0
    }

    all_files = list(downloads_path.glob("**/*.bin"))  # recursive

    for file_path in all_files:
        is_mine, reason, details = is_photomaker(file_path)
        if not is_mine:
            result['skipped'] += 1
            continue

        # Analyze
        analysis = analyze_file(file_path)
        if analysis['status'] == 'SKIP':
            result['skipped'] += 1
            continue

        size_gb = file_path.stat().st_size / (1024**3)

        result['files'].append({
            'path': file_path,
            'filename': file_path.name,
            'size_gb': size_gb,
            'result': {
                'version': analysis['version'],
                'proper_name': analysis['proper_name'],
                'target_folder': analysis['target_folder']
            }
        })

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python modul10_photomaker.py [A|B] [--preview]")
        sys.exit(1)

    mode = sys.argv[1].upper()

    if mode == "A":
        modus_a()
    elif mode == "B":
        # Check for --preview flag
        preview = "--preview" in sys.argv
        modus_b(scan_only=False, batch_mode=False, preview_mode=preview)
    else:
        print("Invalid mode. Use A or B.")
        sys.exit(1)
