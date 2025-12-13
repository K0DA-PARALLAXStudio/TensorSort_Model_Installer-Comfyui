#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 10 - IP-Adapter (Image Prompt Adapter)

Organisiert und benennt IP-Adapter Models (.safetensors, .bin) nach standardisiertem Schema.

Unterstützt:
- Modus A: Installation aus downloads/
- Modus B: Check/Fix bestehender IP-Adapter in ipadapter/, ipadapter-flux/, xlabs/ipadapters/

WICHTIG: 3 SEPARATE ORDNER für inkompatible Formate!
- models/ipadapter/         → SD1.5 & SDXL (alle Varianten)
- models/ipadapter-flux/    → Flux InstantX/Shakker (nested dict .bin, 5GB)
- models/xlabs/ipadapters/  → Flux XLabs (flat keys .safetensors, 936MB)

Namenskonvention:
    IPA_{BaseModel}{Variant}_{Encoder}[_vX].{safetensors|bin}

Beispiele:
    IPA_SD15Plus_ViT-H.safetensors
    IPA_SDXLPlus_ViT-H.safetensors
    IPA_FluxBase_SigLIP.bin
"""

import sys
import struct
import json
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
    IPADAPTER_DIR,              # SD1.5 & SDXL
    IPADAPTER_FLUX_DIR,         # Flux InstantX/Shakker
    XLABS_IPADAPTER_DIR,        # Flux XLabs
    XLABS_DIR,                  # xlabs/ parent
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
# HELPER FUNCTIONS - SAFETENSORS
# ============================================================================

def read_safetensors_info(file_path):
    """Liest Keys, Metadata und Shapes aus Safetensors

    Returns:
        tuple: (keys, metadata, shapes)
            keys: list of str - Alle Keys
            metadata: dict - __metadata__ section
            shapes: dict - {key: shape_tuple}
    """
    try:
        with open(file_path, 'rb') as f:
            # Read header size
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]

            # Read header
            metadata_bytes = f.read(header_size)
            header = json.loads(metadata_bytes.decode('utf-8'))

            # Extract metadata
            metadata = header.get('__metadata__', {})

            # Extract keys and shapes
            keys = []
            shapes = {}
            for key, value in header.items():
                if key == '__metadata__':
                    continue
                keys.append(key)
                if isinstance(value, dict) and 'shape' in value:
                    shapes[key] = tuple(value['shape'])

            return keys, metadata, shapes

    except Exception as e:
        print(f"[ERROR] Cannot read safetensors: {file_path.name}: {e}")
        return [], {}, {}


def read_bin_info(file_path):
    """Liest Info aus .bin File (PyTorch format)

    Lädt nur Structure, nicht die kompletten Weights.

    Returns:
        tuple: (keys, metadata, shapes, total_params)
            keys: list - Top-level und nested keys (flattened)
            metadata: dict - Empty (bin hat normalerweise keine metadata)
            shapes: dict - {key: shape_tuple}
            total_params: int - Geschätzte Parameter-Anzahl
    """
    try:
        import torch

        # Load checkpoint
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

        keys = []
        shapes = {}
        total_params = 0

        def extract_from_dict(d, prefix=''):
            """Recursively extract keys from nested dicts"""
            nonlocal total_params

            if isinstance(d, dict):
                for key, value in d.items():
                    full_key = f"{prefix}{key}" if prefix else key

                    if isinstance(value, dict):
                        # Nested dict - recurse
                        extract_from_dict(value, f"{full_key}.")
                    elif hasattr(value, 'shape'):
                        # Tensor
                        keys.append(full_key)
                        shapes[full_key] = tuple(value.shape)
                        total_params += value.numel() if hasattr(value, 'numel') else 0
                    else:
                        # Other value (int, str, etc.)
                        keys.append(full_key)

        extract_from_dict(checkpoint)

        return keys, {}, shapes, total_params

    except Exception as e:
        print(f"[ERROR] Cannot read .bin file: {file_path.name}: {e}")
        return [], {}, {}, 0


# ============================================================================
# MODULE BOUNDARY CHECK
# ============================================================================

def is_ipadapter(file_path):
    """Prüft ob Datei ein IP-Adapter ist (Module Boundary)

    Detection basiert auf:
    - Extension (.safetensors, .bin)
    - Tensor Patterns (image_proj, ip_adapter, double_blocks.*.processor.ip_adapter_*)
    - Size Range (22 MB - 5 GB)
    - Negative: NICHT Base Model, LoRA, ControlNet, VAE

    Returns:
        tuple: (is_match, reason)
            is_match: bool - True wenn IP-Adapter
            reason: str - Grund (für Logging)
    """
    # Extension Check
    if file_path.suffix.lower() not in ['.safetensors', '.bin']:
        return False, "Not .safetensors or .bin"

    # Size Check
    size_mb = file_path.stat().st_size / (1024 * 1024)

    # Zu klein für IP-Adapter (unter 10 MB)
    if size_mb < 10:
        return False, f"Too small ({size_mb:.1f} MB)"

    # Zu groß für IP-Adapter (über 6 GB)
    # Base Models sind 2-24 GB, aber Flux InstantX IP-Adapter ist 5 GB
    if size_mb > 6000:
        return False, f"Too large ({size_mb:.1f} MB, likely Base Model)"

    # Read Keys
    if file_path.suffix.lower() == '.safetensors':
        keys, metadata, shapes = read_safetensors_info(file_path)
    else:  # .bin
        keys, metadata, shapes, _ = read_bin_info(file_path)

    if not keys:
        return False, "Cannot read keys"

    # IP-Adapter Tensor Pattern Check
    has_image_proj = any('image_proj' in k for k in keys)
    has_ip_adapter = any('ip_adapter' in k for k in keys)
    has_to_k_ip = any('to_k_ip' in k for k in keys)
    has_to_v_ip = any('to_v_ip' in k for k in keys)

    # SD/SDXL IP-Adapter Pattern
    if has_image_proj or (has_ip_adapter and (has_to_k_ip or has_to_v_ip)):
        return True, "IP-Adapter detected (SD/SDXL pattern)"

    # Flux IP-Adapter Pattern (XLabs or InstantX)
    has_double_blocks_ip = any('double_blocks' in k and 'ip_adapter' in k for k in keys)
    if has_double_blocks_ip:
        return True, "IP-Adapter detected (Flux pattern)"

    # Filename Check (Fallback)
    filename_lower = file_path.name.lower()
    if 'ip-adapter' in filename_lower or 'ipadapter' in filename_lower:
        # Plausibility: Size range ok?
        if 20 < size_mb < 6000:
            return True, "IP-Adapter detected (filename + plausible size)"

    return False, "No IP-Adapter pattern detected"


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_base_model(keys, metadata, filename, file_path, shapes):
    """Erkennt Base Model (SD1.5, SDXL, Flux)

    Priorität:
    1. Tensor Dimensions (latents, norm, to_k_ip input dims)
    2. Tensor Pattern (double_blocks = Flux)
    3. Metadata
    4. Filename
    5. Folder Location

    Returns:
        tuple: (base_model, confidence)
            base_model: str - 'SD1.5', 'SDXL', 'Flux', 'Unknown'
            confidence: str - 'High', 'Medium', 'Low'
    """

    # ========================================================================
    # PRIORITÄT 1: TENSOR DIMENSIONS (Input dims, not output!)
    # ========================================================================

    # Check image_proj.latents dimension (Perceiver latents)
    if 'image_proj.latents' in shapes:
        latent_dim = shapes['image_proj.latents'][-1]  # Last dimension
        if latent_dim == 768:
            return 'SD1.5', 'High'
        elif latent_dim == 1280:
            return 'SDXL', 'High'
        elif latent_dim == 2048:
            return 'SDXL', 'High'

    # Check image_proj.norm dimension
    if 'image_proj.norm.weight' in shapes:
        norm_dim = shapes['image_proj.norm.weight'][0]
        if norm_dim == 768:
            return 'SD1.5', 'High'
        elif norm_dim == 2048:
            return 'SDXL', 'High'
        elif norm_dim == 1280:
            return 'SDXL', 'High'

    # Check ip_adapter input dimension (to_k_ip/to_v_ip)
    for key in ['ip_adapter.1.to_k_ip.weight', 'ip_adapter.0.to_k_ip.weight']:
        if key in shapes:
            input_dim = shapes[key][1]  # Input dimension
            if input_dim == 768:
                return 'SD1.5', 'High'
            elif input_dim == 2048:
                return 'SDXL', 'High'
            elif input_dim == 1280:
                return 'SDXL', 'High'

    # ========================================================================
    # PRIORITÄT 2: TENSOR PATTERN
    # ========================================================================

    # Flux has unique double_blocks pattern
    if any('double_blocks' in k and 'ip_adapter' in k for k in keys):
        return 'Flux', 'High'

    # ========================================================================
    # PRIORITÄT 3: METADATA
    # ========================================================================

    if metadata:
        arch = metadata.get('modelspec.architecture', '').lower()
        if 'sdxl' in arch:
            return 'SDXL', 'Medium'
        elif 'sd15' in arch or 'sd-v1' in arch or 'sd_v1' in arch:
            return 'SD1.5', 'Medium'
        elif 'flux' in arch:
            return 'Flux', 'Medium'

    # ========================================================================
    # PRIORITÄT 4: FILENAME
    # ========================================================================

    filename_lower = filename.lower()

    if 'sdxl' in filename_lower:
        return 'SDXL', 'Medium'
    elif 'sd15' in filename_lower or 'sd1.5' in filename_lower or 'sd_1_5' in filename_lower:
        return 'SD1.5', 'Medium'
    elif 'flux' in filename_lower:
        return 'Flux', 'Medium'

    # ========================================================================
    # PRIORITÄT 5: FOLDER LOCATION
    # ========================================================================

    parent_lower = str(file_path.parent).lower()

    if 'flux' in parent_lower:
        return 'Flux', 'Low'
    elif 'sdxl' in parent_lower:
        return 'SDXL', 'Low'
    elif 'sd15' in parent_lower or 'sd1.5' in parent_lower:
        return 'SD1.5', 'Low'

    # Fallback
    return 'Unknown', 'Low'


def detect_variant(keys, filename, num_keys, size_mb):
    """Erkennt IP-Adapter Variante (Plus, Light, Face, etc.)

    Priorität:
    1. Perceiver-Resampler Pattern (nur Plus)
    2. Tensor Count + Size
    3. Filename

    Returns:
        str: Variant ('Base', 'Plus', 'Light', 'Face', 'Plus-Face', 'Full-Face', 'FaceID', 'FaceID-Plus')
    """

    # ========================================================================
    # PRIORITÄT 1: PERCEIVER-RESAMPLER PATTERN (nur Plus)
    # ========================================================================

    has_perceiver = any('perceiver' in k.lower() for k in keys)
    has_latents = any('latents' in k and 'image_proj' in k for k in keys)

    if has_perceiver or has_latents:
        # Plus Variant hat Perceiver-Resampler
        filename_lower = filename.lower()
        if 'face' in filename_lower:
            return 'Plus-Face'
        else:
            return 'Plus'

    # ========================================================================
    # PRIORITÄT 2: TENSOR COUNT + SIZE
    # ========================================================================

    # Plus: ~150-200 keys, ~800 MB (SDXL) or ~100 MB (SD1.5)
    # Base: ~80-120 keys, ~600-700 MB (SDXL) or ~90 MB (SD1.5)
    # Light: ~30-50 keys, ~22-50 MB
    # Face: ~83-296 keys, ~43-100 MB
    # FaceID Plus: 346-1318 keys, ~150-1400 MB (sehr groß!)

    filename_lower = filename.lower()

    # Light
    if num_keys < 60 and size_mb < 100:
        if 'light' in filename_lower:
            return 'Light'

    # Plus (Large key count OR large size)
    if num_keys > 150 or size_mb > 500:
        if 'face' in filename_lower:
            if 'full' in filename_lower:
                return 'Full-Face'
            else:
                return 'Plus-Face'
        else:
            return 'Plus'

    # ========================================================================
    # PRIORITÄT 3: FILENAME
    # ========================================================================

    if 'plus' in filename_lower:
        if 'face' in filename_lower:
            return 'Plus-Face'
        else:
            return 'Plus'

    elif 'light' in filename_lower:
        return 'Light'

    elif 'full-face' in filename_lower or 'fullface' in filename_lower:
        return 'Full-Face'

    elif 'faceid' in filename_lower:
        if 'plus' in filename_lower:
            return 'FaceID-Plus'
        else:
            return 'FaceID'

    elif 'face' in filename_lower:
        return 'Face'

    # Default: Base
    return 'Base'


def detect_encoder(shapes, filename, base_model):
    """Erkennt CLIP Vision Encoder (ViT-H, ViT-G, SigLIP)

    Priorität:
    1. Tensor Dimensions (image_proj, perceiver input dims)
    2. Filename

    Returns:
        str: Encoder ('ViT-H', 'ViT-G', 'SigLIP-SO400M', 'Unknown')
    """

    # Flux nutzt immer SigLIP
    if base_model == 'Flux':
        return 'SigLIP-SO400M'

    # SD1.5 nutzt immer ViT-H
    if base_model == 'SD1.5':
        return 'ViT-H'

    # ========================================================================
    # SDXL: ViT-H oder ViT-G
    # ========================================================================

    # PRIORITÄT 1: Tensor Dimensions
    # Suche nach image_proj oder perceiver Eingangsdimensionen

    for key, shape in shapes.items():
        if 'image_proj' in key or 'perceiver' in key:
            # Check für typische Dimensionen
            if 1280 in shape:  # ViT-G/bigG = 1280 dim
                return 'ViT-G'
            elif 1024 in shape:  # ViT-H = 1024 dim
                return 'ViT-H'

    # PRIORITÄT 2: Filename
    filename_lower = filename.lower()

    if 'vit-h' in filename_lower or 'vit_h' in filename_lower:
        return 'ViT-H'
    elif 'vit-g' in filename_lower or 'vit_g' in filename_lower or 'vitg' in filename_lower or 'bigg' in filename_lower:
        return 'ViT-G'

    # Fallback: SDXL meist ViT-H
    if base_model == 'SDXL':
        return 'ViT-H'

    return 'Unknown'


def detect_version(filename):
    """Erkennt Version aus Filename

    Returns:
        str or None: Version (z.B. "v2", "v11") oder None
    """
    import re

    # Pattern: _v1, _v2, _v10, v11, etc.
    version_match = re.search(r'[_-]?v(\d+)', filename.lower())

    if version_match:
        return f"v{version_match.group(1)}"

    return None


def is_xlabs_pattern(keys):
    """Prüft ob Flux IP-Adapter das XLabs Format hat

    XLabs Format:
        - Flat keys: double_blocks.0.processor.ip_adapter_double_stream_k_proj.weight
        - File Size: ~936 MB

    InstantX Format:
        - Nested dict: {'image_proj': {...}, 'ip_adapter': {...}}
        - File Size: ~5 GB

    Returns:
        bool: True wenn XLabs Format
    """
    # XLabs hat "processor.ip_adapter_double_stream_" pattern
    has_xlabs_pattern = any('processor.ip_adapter_double_stream_' in k for k in keys)

    return has_xlabs_pattern


def get_target_folder(base_model, file_path, keys):
    """Bestimmt Ziel-Ordner basierend auf Base Model und Format

    Returns:
        Path or None: Ziel-Ordner oder None (→ Queue für User)
    """

    # SD1.5 & SDXL → ipadapter/
    if base_model in ['SD1.5', 'SDXL']:
        return IPADAPTER_DIR

    # Flux: Unterscheide XLabs vs InstantX
    if base_model == 'Flux':
        # Prüfe ob XLabs Pattern
        if is_xlabs_pattern(keys):
            # XLabs Format → braucht xlabs/ folder
            if XLABS_DIR.exists():
                return XLABS_IPADAPTER_DIR
            else:
                # XLabs Extension nicht installiert
                # File NICHT nach ipadapter-flux/ (würde crashen!)
                return None  # → Queue für User
        else:
            # InstantX Format → ipadapter-flux/
            return IPADAPTER_FLUX_DIR

    # Unknown Base Model
    return IPADAPTER_DIR  # Fallback


def generate_proper_name(base_model, variant, encoder, version, ext):
    """Generiert standardisierten Namen

    Format: IPA_{BaseModel}{Variant}_{Encoder}[_vX].{ext}

    Returns:
        str: Generierter Filename
    """
    # BaseModel abbreviations
    base_abbr = {
        'SD1.5': 'SD15',
        'SDXL': 'SDXL',
        'Flux': 'Flux',
        'Unknown': 'Unknown'
    }

    base_str = base_abbr.get(base_model, base_model)

    # Variant (remove spaces and dashes)
    variant_str = variant.replace('-', '').replace(' ', '')

    # Build parts
    parts = ['IPA', f"{base_str}{variant_str}"]

    # Encoder (nur wenn nicht Unknown)
    if encoder and encoder != 'Unknown':
        parts.append(encoder)

    # Version (optional)
    if version:
        parts.append(version)

    # Join with underscore
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
        module_name_caps="IP-ADAPTER (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.safetensors, *.bin",
        module_type="IP-Adapter",
        target_folders="ipadapter/, ipadapter-flux/, xlabs/ipadapters/"
    )

    # Finde alle .safetensors und .bin Dateien (recursive)
    all_files = []
    all_files += list(DOWNLOADS_DIR.glob("**/*.safetensors"))  # recursive
    all_files += list(DOWNLOADS_DIR.glob("**/*.bin"))  # recursive

    if not all_files:
        print_no_files_found("IP-Adapter files")
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
        ext = file_path.suffix

        is_match, reason = is_ipadapter(file_path)
        if not is_match:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        if ext == '.safetensors':
            keys, metadata, shapes = read_safetensors_info(file_path)
        else:
            keys, metadata, shapes, _ = read_bin_info(file_path)

        if not keys:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        base_model, confidence = detect_base_model(keys, metadata, filename, file_path, shapes)
        variant = detect_variant(keys, filename, len(keys), size_mb)
        encoder = detect_encoder(shapes, filename, base_model)
        version = detect_version(filename)

        target_folder = get_target_folder(base_model, file_path, keys)

        if target_folder is None:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        proper_name = generate_proper_name(base_model, variant, encoder, version, ext)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'proper_name': proper_name,
            'target_folder': target_folder,
            'detected_str': f"{base_model}, {variant}, {encoder}",
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
        skip_reason="Not an IP-Adapter (no matching keys found)",
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
            target_path=f"{file_info['target_folder'].name}/{file_info['proper_name']}"
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
        proper_name = file_info['proper_name']
        target_folder = file_info['target_folder']
        target_path = target_folder / proper_name

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

    print_summary(len(files_to_install), installed, collisions, errors, keep_source)


# ============================================================================
# MODUS B - REINSTALL/CHECK
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check bestehender IP-Adapter"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="IP-ADAPTER",
        folders="ipadapter/, ipadapter-flux/, xlabs/ipadapters/",
        extensions="*.safetensors, *.bin",
        module_type="IP-Adapter",
        target_folders="ipadapter/, ipadapter-flux/, xlabs/ipadapters/",
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
                is_match, reason = is_ipadapter(file_path)
                if not is_match:
                    continue

                filename = file_path.name
                ext = file_path.suffix
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                if ext == '.safetensors':
                    keys, metadata, shapes = read_safetensors_info(file_path)
                else:
                    keys, metadata, shapes, _ = read_bin_info(file_path)

                if not keys:
                    continue

                base_model, confidence = detect_base_model(keys, metadata, filename, file_path, shapes)
                variant = detect_variant(keys, filename, len(keys), file_size_mb)
                encoder = detect_encoder(shapes, filename, base_model)
                version = detect_version(filename)

                target_folder = get_target_folder(base_model, file_path, keys)

                if target_folder is None:
                    continue

                proper_name = generate_proper_name(base_model, variant, encoder, version, ext)
                target_path = target_folder / proper_name

                print(f"[RESCUE] Found misplaced IP-Adapter: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

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
            print(f"[SUCCESS] Rescued {rescued} misplaced IP-Adapter(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    folders_to_scan = [
        (IPADAPTER_DIR, "ipadapter"),
        (IPADAPTER_FLUX_DIR, "ipadapter-flux"),
        (XLABS_IPADAPTER_DIR, "xlabs/ipadapters")
    ]

    all_files = []
    for folder, folder_name in folders_to_scan:
        if folder.exists():
            all_files += [(f, folder_name) for f in folder.glob("**/*.safetensors")]  # recursive
            all_files += [(f, folder_name) for f in folder.glob("**/*.bin")]  # recursive

    if not all_files:
        print_no_files_found("IP-Adapter files")
        return

    # ========================================================================
    # ANALYZE ALL FILES (silently collect)
    # ========================================================================
    correct_files = []
    problems_list = []
    renamed = 0
    moved = 0

    for file_path, current_folder_name in all_files:
        filename = file_path.name
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        ext = file_path.suffix

        is_match, reason = is_ipadapter(file_path)
        if not is_match:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason, current_folder_name))
            continue

        if scan_only:
            continue

        if ext == '.safetensors':
            keys, metadata, shapes = read_safetensors_info(file_path)
        else:
            keys, metadata, shapes, _ = read_bin_info(file_path)

        if not keys:
            continue

        base_model, confidence = detect_base_model(keys, metadata, filename, file_path, shapes)
        variant = detect_variant(keys, filename, len(keys), file_size_mb)
        encoder = detect_encoder(shapes, filename, base_model)
        version = detect_version(filename)

        target_folder = get_target_folder(base_model, file_path, keys)

        if target_folder is None:
            problems_list.append(('xlabs_missing', filename, file_size_mb, current_folder_name))
            continue

        proper_name = generate_proper_name(base_model, variant, encoder, version, ext)
        detected_str = f"{base_model}, {variant}, {encoder}"

        if file_path.parent != target_folder:
            problems_list.append(('wrong_folder', filename, file_size_mb, detected_str, current_folder_name, target_folder.name, proper_name, target_folder, file_path))
        elif file_path.name != proper_name:
            problems_list.append(('wrong_name', filename, file_size_mb, detected_str, current_folder_name, proper_name, file_path))
        else:
            correct_files.append((current_folder_name, filename, file_size_mb))

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
                    warning=f"Not an IP-Adapter: {reason}"
                )

            elif problem_type == 'xlabs_missing':
                _, fname, size_mb, folder = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=None,
                    target_path="Install x-flux-comfyui extension",
                    warning="XLabs Flux IP-Adapter but xlabs/ folder missing"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, detected_str, curr_folder, expected_folder, proper_name, target_folder, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=detected_str,
                    target_path=f"{expected_folder}/{proper_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    target_path = target_folder / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=proper_name,
                        mode="B",
                        keep_source_option=False
                    )

                    print_fix_result(success, "Moved and renamed to correct location" if success else msg)
                    if success:
                        moved += 1

            elif problem_type == 'wrong_name':
                _, fname, size_mb, detected_str, folder_name, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=detected_str,
                    target_path=f"{folder_name}/{proper_name}",
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
    """Scannt downloads/ für IP-Adapter (für all_modules.py Preview)

    Returns:
        list: Liste von dicts mit file info
    """
    results = []

    # Finde alle .safetensors und .bin Dateien (recursive)
    all_files = []
    all_files += list(downloads_path.glob("**/*.safetensors"))  # recursive
    all_files += list(downloads_path.glob("**/*.bin"))  # recursive

    for file_path in all_files:
        # Module Boundary Check
        is_match, reason = is_ipadapter(file_path)
        if not is_match:
            continue

        filename = file_path.name
        ext = file_path.suffix

        # Read Info
        if ext == '.safetensors':
            keys, metadata, shapes = read_safetensors_info(file_path)
        else:  # .bin
            keys, metadata, shapes, _ = read_bin_info(file_path)

        if not keys:
            continue

        # Detection
        base_model, confidence = detect_base_model(keys, metadata, filename, file_path, shapes)
        variant = detect_variant(keys, filename, len(keys), file_path.stat().st_size / (1024 * 1024))
        encoder = detect_encoder(shapes, filename, base_model)
        version = detect_version(filename)

        # Target Folder
        target_folder = get_target_folder(base_model, file_path, keys)

        if target_folder is None:
            # XLabs ohne xlabs/ folder - skip
            continue

        # Generate proper name
        proper_name = generate_proper_name(base_model, variant, encoder, version, ext)

        # Size
        size_gb = file_path.stat().st_size / (1024**3)

        results.append({
            'path': file_path,
            'filename': file_path.name,
            'size_gb': size_gb,
            'result': {
                'base_model': base_model,
                'variant': variant,
                'encoder': encoder,
                'version': version,
                'proper_name': proper_name,
                'target_folder': target_folder
            }
        })

    return {
        'module_name': 'IP-Adapter',
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
        print("  python modul12_ipadapter.py A          - Modus A (Installation)")
        print("  python modul12_ipadapter.py B          - Modus B (Reinstall/Check)")
        print("  python modul12_ipadapter.py B --scan-only    - Modus B PASS 1")
        print("  python modul12_ipadapter.py B --preview      - Modus B Preview")
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
