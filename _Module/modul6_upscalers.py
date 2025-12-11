#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 6 - Upscalers

Organisiert und benennt Upscaler Models (.pth, .pt) nach standardisiertem Schema.

Unterstützt:
- Modus A: Installation aus downloads/
- Modus B: Check/Fix bestehender Upscaler in upscale_models/

Namenskonvention:
    Upscaler_{Scale}_{Types}_{Name}_{Version}.{ext}

Beispiele:
    Upscaler_4x_Anime_AnimeSharp.pth
    Upscaler_8x_Photo_NMKDSuperscale.pth
    Upscaler_4x_Photo-Face_GFPGan_v1.pt
"""

import sys
import struct
import json
import re
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

# Target folder
UPSCALE_MODELS_DIR = MODELS_DIR / "upscale_models"

# ============================================================================
# KNOWN UPSCALERS DATABASE (from known_upscalers.py)
# ============================================================================
# Integrated directly into script for later single-file distribution

KNOWN_UPSCALERS = {
    # Anime/Manga Upscalers
    'realesrgan_x4plus_anime': {'type': 'Anime', 'arch': 'RealESRGAN'},
    'realesrgan-x4-plus-anime': {'type': 'Anime', 'arch': 'RealESRGAN'},
    'animesharp': {'type': 'Anime-Detail', 'arch': 'ESRGAN'},
    '4xanimesharp': {'type': 'Anime-Detail', 'arch': 'ESRGAN'},
    'fatal_anime': {'type': 'Anime', 'arch': 'ESRGAN'},
    'fatalanime': {'type': 'Anime', 'arch': 'ESRGAN'},
    'nmkd-ultrayandere': {'type': 'Anime', 'arch': 'NMKD'},
    'hfa2k': {'type': 'Anime', 'arch': 'ESRGAN'},
    'anime4k': {'type': 'Anime', 'arch': 'Anime4K'},
    'anijapan': {'type': 'Anime', 'arch': 'ESRGAN'},
    'aniscale': {'type': 'Anime', 'arch': 'ESRGAN'},

    # Photo/Realistic Upscalers
    'nmkd-superscale': {'type': 'Photo', 'arch': 'NMKD'},
    'ultrasharp': {'type': 'Photo-Detail', 'arch': 'ESRGAN'},
    '4xultrasharp': {'type': 'Photo-Detail', 'arch': 'ESRGAN'},
    'realesrgan-x4plus': {'type': 'Photo', 'arch': 'RealESRGAN'},
    'realesrgan_x4plus': {'type': 'Photo', 'arch': 'RealESRGAN'},
    'foolhardy': {'type': 'Photo-Detail', 'arch': 'ESRGAN'},
    'remacri': {'type': 'Photo-Detail', 'arch': 'ESRGAN'},
    'nomos8k': {'type': 'Photo-Web', 'arch': 'ESRGAN'},
    'nomos2': {'type': 'Photo', 'arch': 'ESRGAN'},
    'nomos8kdat': {'type': 'Photo', 'arch': 'DAT'},
    'artfaces': {'type': 'Photo-Face', 'arch': 'ESRGAN'},
    'gfpgan': {'type': 'Photo-Face', 'arch': 'GAN'},
    'codeformer': {'type': 'Photo-Face', 'arch': 'Transformer'},
    'supir': {'type': 'Photo-Realistic', 'arch': 'Diffusion'},

    # AI-Generated Content
    'aigcsmooth': {'type': 'AI-Smooth', 'arch': 'ESRGAN'},
    'smooth_diff': {'type': 'AI-Smooth', 'arch': 'ESRGAN'},
    'smoothdiff': {'type': 'AI-Smooth', 'arch': 'ESRGAN'},

    # Special Purpose
    'text2hd': {'type': 'Text', 'arch': 'RealPLKSR'},
    'typescale': {'type': 'Text', 'arch': 'NMKD'},
    'pixelart': {'type': 'PixelArt', 'arch': 'ESRGAN'},
    'gamescreenshot': {'type': 'GameScreenshot', 'arch': 'ESRGAN'},

    # General/Modern Architectures
    'swinir': {'type': 'General', 'arch': 'SwinIR'},
    'dat': {'type': 'General', 'arch': 'DAT'},
    'realplksr': {'type': 'General', 'arch': 'RealPLKSR'},
    'plksr': {'type': 'General', 'arch': 'RealPLKSR'},
    'bsrgan': {'type': 'General', 'arch': 'BSRGAN'},
    'esrgan': {'type': 'General', 'arch': 'ESRGAN'},
    'realesrgan': {'type': 'General', 'arch': 'RealESRGAN'},
    'realesrnet': {'type': 'General', 'arch': 'RealESRGAN'},
    'hat': {'type': 'General', 'arch': 'HAT'},
}

def lookup_known_upscaler(filename):
    """Prüft ob Upscaler in Knowledge Base bekannt ist"""
    filename_lower = filename.lower()

    # Entferne Extension für besseres Matching
    if filename_lower.endswith(('.pth', '.pt', '.safetensors')):
        filename_lower = filename_lower.rsplit('.', 1)[0]

    # Check gegen alle bekannten Patterns
    for pattern, info in KNOWN_UPSCALERS.items():
        if pattern in filename_lower:
            return info

    return None  # Unknown

# ============================================================================
# MODULE BOUNDARY CHECK
# ============================================================================

def is_detection_model(file_path):
    """Prüft ob Datei ein YOLO Detection Model ist (ADetailer)

    Detection Models haben spezifische Torch-Struktur:
    - dict mit Keys: 'epoch', 'model', 'ema', 'optimizer', 'train_args'
    - Upscaler haben OrderedDict mit Model Weights

    Returns:
        bool: True wenn Detection Model (NICHT Upscaler)
    """
    try:
        import torch

        # Lade nur die Keys, nicht die kompletten Weights
        data = torch.load(file_path, map_location='cpu', weights_only=False)

        # Detection Model hat dict mit diesen Keys
        if isinstance(data, dict):
            keys = set(data.keys())
            # YOLO Training Checkpoint Format
            yolo_keys = {'epoch', 'model', 'ema', 'optimizer', 'train_args'}
            if yolo_keys.issubset(keys):
                return True  # Das ist ein Detection Model!

        return False
    except Exception as e:
        # ZIP-Fallback für alte Ultralytics Modelle (ultralytics.yolo Import fehlt)
        if 'ultralytics' in str(e):
            # Versuche ZIP-basierte Analyse
            import zipfile
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    pkl_files = [n for n in zf.namelist() if n.endswith('data.pkl')]
                    if pkl_files:
                        with zf.open(pkl_files[0]) as f:
                            content = f.read().decode('latin-1', errors='replace')
                        # Check für YOLO Patterns
                        if 'ultralytics' in content and ('DetectionModel' in content or 'SegmentationModel' in content):
                            return True  # Das ist ein YOLO Detection Model!
            except:
                pass
        # Bei sonstigem Fehler → konservativ als Upscaler behandeln
        return False


def is_upscaler(file_path):
    """Prüft ob Datei ein Upscaler ist (Module Boundary)

    Detection basiert auf:
    - Extension (.pth, .pt)
    - Dateigröße (3-200 MB typical range)
    - NEGATIVE Detection (kein Detection Model, Base Model, LoRA, etc.)

    Args:
        file_path (Path): Pfad zur Datei

    Returns:
        bool: True wenn Upscaler, False sonst
    """
    # Extension Check
    if file_path.suffix.lower() not in ['.pth', '.pt']:
        return False

    # Size Check
    size_mb = file_path.stat().st_size / (1024 * 1024)

    # Zu klein für Upscaler (unter 3 MB)
    if size_mb < 3:
        return False

    # Zu groß für Upscaler (über 200 MB)
    # Base Models sind 2-24 GB, LoRAs 10-500 MB
    if size_mb > 200:
        return False

    # WICHTIG: Prüfe ob es ein Detection Model ist (gehört zu Modul 22)
    if is_detection_model(file_path):
        return False  # NICHT für dieses Modul!

    # Falls in upscale_models/ → vermutlich Upscaler
    if 'upscale_models' in str(file_path):
        return True

    # In downloads/: Akzeptieren (andere Module haben bereits geskippt)
    return True

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_scale_factor(filename):
    """Erkennt Scale Factor aus Filename

    Returns:
        str: Scale Factor (2x, 4x, 6x, 8x) oder "UnknownX"
    """
    filename_lower = filename.lower()

    # Explizit im Namen
    if '8x' in filename_lower:
        return '8x'
    elif '6x' in filename_lower:
        return '6x'
    elif '4x' in filename_lower:
        return '4x'
    elif '2x' in filename_lower:
        return '2x'

    # Fallback: UnknownX (ehrlich statt falsche Annahme)
    return 'UnknownX'


def detect_types_from_filename(filename):
    """Erkennt Types aus Filename (Keyword-basiert)

    Returns:
        list: Liste der Types (z.B. ['Anime', 'Detail'])
    """
    filename_lower = filename.lower()
    types = []

    # Type Keywords (Order matters - check specific before general)
    type_map = {
        'Anime': ['anime', 'manga', 'cartoon'],
        'Photo': ['photo', 'real', 'realistic'],
        'Face': ['face', 'facial', 'portrait', 'gfpgan', 'codeformer'],
        'Detail': ['detail', 'sharp', 'ultrasharp'],
        'Web': ['web', 'jpg', 'jpeg'],
        'Text': ['text', 'typescale'],
        'PixelArt': ['pixelart', 'pixel'],
        'GameScreenshot': ['gamescreenshot', 'game'],
        'AI-Smooth': ['aigc', 'smooth'],
    }

    for type_name, keywords in type_map.items():
        if any(kw in filename_lower for kw in keywords):
            if type_name not in types:  # Avoid duplicates
                types.append(type_name)

    # Fallback: General
    if not types:
        types = ['General']

    return types


def detect_types_and_arch(filename):
    """Erkennt Types + Architecture aus Filename

    Uses Knowledge Base first, then falls back to filename keyword detection.

    Returns:
        tuple: (types_list, architecture or None)
    """
    # Check Knowledge Base first
    known_info = lookup_known_upscaler(filename)

    if known_info:
        # Parse type (kann "-" enthalten wie "Anime-Detail")
        type_str = known_info['type']
        types = type_str.split('-')  # "Anime-Detail" → ["Anime", "Detail"]
        arch = known_info['arch']
        return types, arch

    # Fallback: Filename-basierte Detection
    types = detect_types_from_filename(filename)

    # Architecture aus Filename (wenn vorhanden)
    filename_lower = filename.lower()

    arch_map = {
        'RealESRGAN': ['realesrgan'],
        'NMKD': ['nmkd'],
        'SwinIR': ['swinir'],
        'DAT': ['dat'],
        'RealPLKSR': ['realplksr', 'plksr'],
        'HAT': ['hat'],
        'BSRGAN': ['bsrgan'],
        'ESRGAN': ['esrgan'],  # Generic, last
    }

    arch = None
    for arch_name, keywords in arch_map.items():
        if any(kw in filename_lower for kw in keywords):
            arch = arch_name
            break

    # Architecture Fallback: Weglassen (None) wenn nicht erkennbar
    return types, arch


def detect_version(filename):
    """Erkennt Version aus Filename

    Returns:
        str or None: Version (z.B. "v2") oder None
    """
    # Pattern: _v1, _v2, _v10, -v1, etc.
    version_match = re.search(r'[_-]v(\d+)', filename.lower())

    if version_match:
        return f"v{version_match.group(1)}"

    return None


def extract_clean_name(filename, scale, types, arch, version):
    """Extrahiert sauberen Namen aus Filename - BEHÄLT Original-Infos

    Neues Konzept (2025-12):
    - Entfernt Prefix (Upscaler_, Scale am Anfang)
    - Entfernt Types die SEPARAT stehen (mit _/- getrennt), nicht wenn Teil eines Wortes
    - BEHÄLT Training-Artefakte wie _500000_G, _150000_G
    - Verhindert Duplikate im finalen Namen

    Returns:
        str: Bereinigter Name mit Original-Infos
    """
    name = filename

    # Entferne Extension
    if name.lower().endswith(('.pth', '.pt', '.safetensors')):
        name = name.rsplit('.', 1)[0]

    # Entferne "Upscaler_" Prefix falls vorhanden
    if name.lower().startswith('upscaler_'):
        name = name[9:]  # Remove "Upscaler_"

    # Entferne Scale Factor NUR am Anfang (um Duplikate zu vermeiden)
    # z.B. "4x_fatal_Anime" -> "fatal_Anime" (da 4x bereits im neuen Namen ist)
    name = re.sub(r'^[2468]x[_-]?', '', name, flags=re.IGNORECASE)

    # Entferne Types die als SEPARATE Wörter stehen (um Duplikate zu vermeiden)
    # z.B. "fatal_Anime_500000_G" -> "fatal_500000_G" (da Anime bereits im Type-Teil ist)
    # ABER: "AnimeSharp" bleibt "AnimeSharp" (Teil eines Wortes)
    for type_name in types:
        # Entferne Type wenn es als separates Wort steht (umgeben von _/- oder am Rand)
        # Pattern: _Anime_ oder _Anime$ oder ^Anime_
        name = re.sub(rf'[_-]{type_name}[_-]', '_', name, flags=re.IGNORECASE)
        name = re.sub(rf'[_-]{type_name}$', '', name, flags=re.IGNORECASE)
        name = re.sub(rf'^{type_name}[_-]', '', name, flags=re.IGNORECASE)

    # Architecture BEHALTEN - ist wertvolle Info!
    # (NMKD, RealESRGAN, etc. bleiben im Namen)

    # Entferne Version am Ende (wird separat hinzugefügt)
    if version:
        name = re.sub(rf'[_-]?{version}$', '', name, flags=re.IGNORECASE)

    # Entferne bestimmte Artefakte die keine nützliche Info haben
    # (atd = "all the details", jpg/png = file format hints)
    name = re.sub(r'[_-]?(atd|jpg|png|jpeg|paired)[_-]?', '', name, flags=re.IGNORECASE)

    # BEHALTE Training-Artefakte wie _500000_G, _150000_G (nützliche Info!)

    # Cleanup: Multiple underscores/dashes
    name = re.sub(r'^[_-]+|[_-]+$', '', name)
    name = re.sub(r'[_-]+', '_', name)  # Multiple -> Single

    # Capitalize first letter of each word (but keep underscores)
    parts = name.split('_')
    parts = [p.capitalize() if p else p for p in parts]
    name = '_'.join(parts)

    # Fallback wenn leer
    if not name:
        name = "Unknown"

    return name


def generate_proper_name(filename, scale, types, arch, version):
    """Generiert standardisierten Namen

    Format: Upscaler_{Scale}_{Types}_{Name}_{Version}.{ext}

    Returns:
        str: Generierter Filename
    """
    # Clean Name extrahieren
    clean_name = extract_clean_name(filename, scale, types, arch, version)

    # Types zusammenfügen (max 3)
    types_limited = types[:3]
    types_str = '-'.join(types_limited)

    # Extension beibehalten (Original .pt oder .pth)
    ext = '.pth' if filename.lower().endswith('.pth') else '.pt'

    # Build Name: Upscaler_{Scale}_{Types}_{Name}_{Version}.{ext}
    parts = ['Upscaler', scale, types_str, clean_name]

    if version:
        parts.append(version)

    proper_name = '_'.join(parts) + ext

    return proper_name

# ============================================================================
# MODUS A - INSTALLATION
# ============================================================================

def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="UPSCALERS (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.pth, *.pt, *.safetensors",
        module_type="Upscalers",
        target_folders="upscale_models/"
    )

    # Finde alle .pth und .pt Dateien
    all_files = []
    all_files += list(DOWNLOADS_DIR.glob("*.pth"))
    all_files += list(DOWNLOADS_DIR.glob("*.pt"))

    if not all_files:
        print_no_files_found("upscaler files")
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

        # Module Boundary Check
        if not is_upscaler(file_path):
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # Detection
        scale = detect_scale_factor(filename)
        types, arch = detect_types_and_arch(filename)
        version = detect_version(filename)

        # Generate proper name
        proper_name = generate_proper_name(filename, scale, types, arch, version)

        types_str = '-'.join(types)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'proper_name': proper_name,
            'detected_str': f"{scale}, {types_str}, {arch if arch else 'Unknown'}",
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
        skip_reason="Not an upscaler model",
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
            target_path=f"upscale_models/{file_info['proper_name']}"
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
        proper_name = file_info['proper_name']
        target_path = UPSCALE_MODELS_DIR / proper_name

        success, final_path, msg = handle_duplicate_move(
            source_path,
            target_path,
            expected_target_name=proper_name,
            mode="A",
            keep_source_option=keep_source
        )

        if success:
            if "collision" in msg.lower() or '_alt' in str(final_path.name):
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
    """Modus B: Check bestehender Upscaler

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes) - used by all_modules.py
        preview_mode: Show problems only, no execution, no prompts - used by all_modules.py
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="UPSCALERS",
        folders="upscale_models/",
        extensions="*.pth, *.pt",
        module_type="Upscalers",
        target_folders="upscale_models/",
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
                if not is_upscaler(file_path):
                    continue

                filename = file_path.name
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                scale = detect_scale_factor(filename)
                types, arch = detect_types_and_arch(filename)
                version = detect_version(filename)
                proper_name = generate_proper_name(filename, scale, types, arch, version)

                target_path = UPSCALE_MODELS_DIR / proper_name

                print(f"[RESCUE] Found misplaced upscaler: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                success, final_path, msg = handle_duplicate_move(
                    file_path,
                    target_path,
                    expected_target_name=proper_name,
                    mode="B",
                    keep_source_option=False
                )

                if success:
                    print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: upscale_models/{final_path.name}")
                    remove_misplaced_file(file_path)
                    rescued += 1
                else:
                    print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced upscaler(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDER
    # ========================================================================
    all_files = []
    all_files += list(UPSCALE_MODELS_DIR.glob("*.pth"))
    all_files += list(UPSCALE_MODELS_DIR.glob("*.pt"))

    if not all_files:
        print_no_files_found("upscaler files")
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

        if not is_upscaler(file_path):
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, "Not an upscaler"))
            continue

        if scan_only:
            continue

        scale = detect_scale_factor(filename)
        types, arch = detect_types_and_arch(filename)
        version = detect_version(filename)
        proper_name = generate_proper_name(filename, scale, types, arch, version)

        current_name = file_path.name

        if current_name != proper_name:
            types_str = '-'.join(types)
            problems_list.append(('wrong_name', filename, file_size_mb, scale, types_str, arch, proper_name, file_path))
        else:
            correct_files.append(("upscale_models", filename, file_size_mb))

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
                    warning=f"Not an upscaler: {reason}"
                )

            elif problem_type == 'wrong_name':
                _, fname, size_mb, scale, types_str, arch, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{scale}, {types_str}, {arch if arch else 'Unknown'}",
                    target_path=f"upscale_models/{proper_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    target_path = fpath.parent / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=proper_name,
                        mode="B",
                        keep_source_option=False
                    )

                    print_fix_result(success, "Renamed to standard format" if success else msg)
                    if success:
                        renamed += 1

# ============================================================================
# BATCH MODE HELPER (for all_modules.py)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scannt downloads/ für Upscaler (für all_modules.py Preview)

    Returns:
        list: Liste von dicts mit file info
    """
    results = []

    # Finde alle .pth und .pt Dateien
    all_files = []
    all_files += list(downloads_path.glob("*.pth"))
    all_files += list(downloads_path.glob("*.pt"))

    for file_path in all_files:
        # Module Boundary Check
        if not is_upscaler(file_path):
            continue

        filename = file_path.name

        # Detection
        scale = detect_scale_factor(filename)
        types, arch = detect_types_and_arch(filename)
        version = detect_version(filename)

        # Generate proper name
        proper_name = generate_proper_name(filename, scale, types, arch, version)

        # Size
        size_gb = file_path.stat().st_size / (1024**3)

        results.append({
            'file_path': file_path,
            'size_gb': size_gb,
            'result': {
                'scale': scale,
                'types': types,
                'arch': arch,
                'version': version,
                'proper_name': proper_name,
                'target_folder': UPSCALE_MODELS_DIR
            }
        })

    return {
        'module_name': 'Upscale Models',
        'files': results,
        'skipped': []
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python modul6_upscalers.py A          - Modus A (Installation)")
        print("  python modul6_upscalers.py B          - Modus B (Reinstall/Check)")
        print("  python modul6_upscalers.py B --scan-only    - Modus B PASS 1")
        print("  python modul6_upscalers.py B --preview      - Modus B Preview")
        sys.exit(0)

    mode = sys.argv[1].upper()

    if mode == "A":
        modus_a()
    elif mode == "B":
        # Check for flags
        scan_only = "--scan-only" in sys.argv
        preview_mode = "--preview" in sys.argv
        batch_mode = "--batch" in sys.argv

        modus_b(scan_only=scan_only, batch_mode=batch_mode, preview_mode=preview_mode)
    else:
        print(f"Unknown mode: {mode}")
        print("Use A or B")
        sys.exit(1)

if __name__ == "__main__":
    main()
