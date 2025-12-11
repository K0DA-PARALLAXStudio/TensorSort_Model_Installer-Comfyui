#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 2 - VAE (Variational AutoEncoder) + VAE Approx
Status: IM TEST

This module handles TWO types of VAE files:
1. Regular VAE (Variational AutoEncoder) - 100-700 MB, .safetensors
2. VAE Approx (TAESD) - 200 KB - 5 MB, .pth/.pt/.safetensors
"""

import os
import sys
import json
import struct
import shutil
import re
from pathlib import Path
from collections import defaultdict, Counter

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
    VAE_DIR,
    VAE_APPROX_DIR,
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
VAE_PATH = VAE_DIR

# Pfad zur Known VAEs Liste
KNOWN_VAES_FILE = _SHARED_DIR / "known_vaes.txt"

# ============================================================================
# KNOWN VAEs LISTE LADEN
# ============================================================================

def load_known_vaes():
    """Lädt bekannte VAE-Namen aus known_vaes.txt"""
    known_vaes = {}

    if not KNOWN_VAES_FILE.exists():
        print(f"[WARNING] known_vaes.txt nicht gefunden: {KNOWN_VAES_FILE}")
        return known_vaes

    try:
        with open(KNOWN_VAES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip Kommentare und leere Zeilen
                if not line or line.startswith('#'):
                    continue

                # Parse: filename_pattern|BaseModel|Description
                parts = line.split('|')
                if len(parts) >= 2:
                    filename_pattern = parts[0].strip()
                    base_model = parts[1].strip()
                    known_vaes[filename_pattern.lower()] = base_model

    except Exception as e:
        print(f"[ERROR] Kann known_vaes.txt nicht laden: {e}")

    return known_vaes

# Lade Liste beim Start
KNOWN_VAES = load_known_vaes()

# ============================================================================
# PHASE 1: BASIS-FUNKTIONEN
# ============================================================================

def read_safetensors_keys(file_path):
    """Liest Keys aus Safetensors"""
    try:
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            metadata_bytes = f.read(header_size)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            return list(metadata.keys())
    except Exception as e:
        print(f"[ERROR] Kann Keys nicht lesen von {file_path}: {e}")
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
        print(f"[ERROR] Kann Metadata nicht lesen von {file_path}: {e}")
        return {}


def is_vae(file_path):
    """Prüft ob Datei ein VAE ist (Module Boundary)"""

    # Convert Path to string if needed
    file_path = str(file_path)

    # 1. Extension Check
    if not file_path.endswith(".safetensors"):
        return False

    # 2. Keys lesen
    keys = read_safetensors_keys(file_path)
    if not keys or len(keys) < 10:
        return False

    # 3. MUSS HABEN: VAE Keys
    vae_required = ["decoder", "encoder"]
    has_vae_keys = all(any(p in k for k in keys) for p in vae_required)

    if not has_vae_keys:
        return False

    # 4. DARF NICHT HABEN: UNET
    # WICHTIG: up_blocks/down_blocks gehören auch zu VAE decoder/encoder!
    # Nur als UNET werten wenn NICHT unter decoder/encoder
    unet_patterns = [
        "model.diffusion_model", "double_blocks", "single_blocks"
    ]
    has_unet = any(any(p in k for p in unet_patterns) for k in keys)

    # Check up_blocks/down_blocks nur wenn nicht Teil von decoder/encoder
    for k in keys:
        if any(p in k for p in ["up_blocks", "down_blocks", "input_blocks", "output_blocks"]):
            # Nur UNET wenn nicht decoder/encoder
            if not any(prefix in k for prefix in ["decoder.", "encoder."]):
                has_unet = True
                break

    # 5. DARF NICHT HABEN: CLIP
    clip_patterns = [
        "conditioner.embedders", "text_model.encoder",
        "clip_l", "clip_g"
    ]
    has_clip = any(any(p in k for p in clip_patterns) for k in keys)

    # 6. DARF NICHT HABEN: LoRA
    lora_patterns = ["lora_", ".alpha", "lora.down", "lora.up"]
    has_lora = any(any(p in k for p in lora_patterns) for k in keys)

    # Nur VAE, nichts anderes
    return has_vae_keys and not has_unet and not has_clip and not has_lora


# ============================================================================
# PHASE 2: BASE MODEL & TYPE DETECTION
# ============================================================================

def detect_base_model_from_metadata(metadata):
    """Erkennt Base Model aus Metadata"""

    # Check __metadata__
    meta_info = metadata.get("__metadata__", {})

    # modelspec.architecture
    arch = meta_info.get("modelspec.architecture", "").lower()

    if "flux" in arch:
        return "Flux"
    elif "sdxl" in arch:
        return "SDXL"
    elif "sd15" in arch or "sd_v1" in arch:
        return "SD15"
    elif "sd21" in arch or "sd_v2" in arch:
        return "SD21"

    return None


def detect_base_model_from_filename(filename):
    """Erkennt Base Model aus Filename"""

    filename_lower = filename.lower()

    # 1. PRIO: Check Known VAEs Liste (exakte Namen)
    if filename_lower in KNOWN_VAES:
        return KNOWN_VAES[filename_lower]

    # 2. PRIO: Keyword-basierte Erkennung

    # Flux
    if "flux" in filename_lower:
        return "Flux"

    # SDXL (inkl. Pony, Illustrious)
    if any(kw in filename_lower for kw in ["sdxl", "pony", "illustrious"]):
        return "SDXL"

    # SD 1.5 / 2.1
    if any(kw in filename_lower for kw in ["sd15", "sd_v1", "840000"]):
        return "SD15"
    if any(kw in filename_lower for kw in ["sd21", "sd_v2"]):
        return "SD21"

    # Qwen
    if "qwen" in filename_lower:
        return "Qwen"

    # WAN (Video)
    if "wan" in filename_lower:
        return "WAN"

    return "Unknown"


def detect_precision_from_metadata(metadata):
    """Erkennt Precision (dtype) aus Metadata"""

    dtype_counts = Counter()

    for key, value in metadata.items():
        if key == "__metadata__":
            continue

        if isinstance(value, dict) and "dtype" in value:
            dtype = value["dtype"]
            dtype_counts[dtype] += 1

    if not dtype_counts:
        return "Unknown"

    # Häufigster dtype
    most_common_dtype = dtype_counts.most_common(1)[0][0]

    # Map dtype names
    dtype_map = {
        "F32": "FP32",
        "F16": "FP16",
        "BF16": "BF16",
        "F64": "F64",
        "I32": "INT32",
        "I16": "INT16"
    }

    return dtype_map.get(most_common_dtype.upper(), most_common_dtype.upper())


def extract_version_from_filename(filename):
    """Extrahiert Version aus Filename (z.B. 2.1, v1.0, etc.)"""
    # Patterns für Versionen: 2.1, v2.1, -2.1-, _2.1_
    version_patterns = [
        r'[-_](\d+\.\d+)[-_]',      # -2.1- oder _2.1_
        r'[-_]v?(\d+\.\d+)',         # -2.1 oder -v2.1 am Ende
        r'v(\d+\.\d+)',              # v2.1
    ]

    for pattern in version_patterns:
        match = re.search(pattern, filename.lower())
        if match:
            return match.group(1)
    return None


def extract_name_from_filename(filename, base_model):
    """Extrahiert Original-Namen aus Filename (nur VAE und Base Model entfernen)"""

    # Entferne Extension
    name = filename.replace(".safetensors", "")
    name_lower = name.lower()

    # =========================================================================
    # SPEZIALFALL: WAN VAEs mit Version (wan-2.1-vae → WAN2.1)
    # =========================================================================
    if base_model == "WAN":
        version = extract_version_from_filename(filename)
        if version:
            # Version gefunden - direkt ans BaseModel anhängen
            # generate_proper_name wird dann: VAE_WAN2.1_BF16.safetensors
            return version  # Wird zu WAN + version = WAN2.1

    # =========================================================================
    # SPEZIALFALL: Qwen VAEs mit Variante (qwen-image-vae → QwenImage)
    # =========================================================================
    if base_model == "Qwen":
        if "image" in name_lower:
            return "Image"  # Wird zu Qwen + Image = QwenImage

    # =========================================================================
    # STANDARD: Entferne Keywords und behalte Rest
    # =========================================================================

    # Liste der zu entfernenden Keywords (case-insensitive)
    keywords_to_remove = [
        "vae",
        base_model.lower(),
        # Base Models
        "flux", "sdxl", "sd15", "sd21", "sd_v1", "sd_v2",
        "pony", "illustrious", "qwen", "wan",
        # Precision (wichtig: sonst wird "fp32" im Namen behalten!)
        "fp32", "fp16", "bf16", "f64", "f32", "f16",
        # Variants (nicht mehr hier - werden oben speziell behandelt)
    ]

    # WICHTIG: Erst Separatoren durch Leerzeichen ersetzen, damit Wortgrenzen funktionieren
    name_lower = re.sub(r'[-_]+', ' ', name_lower)

    # Entferne Keywords
    for keyword in keywords_to_remove:
        # Ersetze mit Leerzeichen um Wortgrenzen zu respektieren
        name_lower = re.sub(rf'\b{re.escape(keyword)}\b', ' ', name_lower)

    # Cleanup: Mehrfache Leerzeichen entfernen und zu - konvertieren
    name_clean = re.sub(r'\s+', '-', name_lower).strip('-_')

    # Wenn nichts übrig bleibt, nutze leeren String (wird in generate_proper_name behandelt)
    if not name_clean or len(name_clean) < 2:
        return ""

    return name_clean


def generate_proper_name(file_info):
    """Generiert korrekten Namen nach Konvention

    Format: VAE_{BaseModel}[{Suffix}|_{OriginalName}]_{Precision}.safetensors

    Regeln:
    - Version (Zahl wie 2.1) → direkt ans BaseModel: WAN2.1
    - Variante (Image) → direkt ans BaseModel: QwenImage
    - Original-Name → mit Unterstrich: Flux_ae, SDXL_hdreffectvae-v10

    Beispiele:
    - wan-2.1-vae.safetensors → VAE_WAN2.1_BF16.safetensors
    - qwen-image-vae.safetensors → VAE_QwenImage_BF16.safetensors
    - flux-ae.safetensors → VAE_Flux_ae_FP32.safetensors
    - sdxl-hdreffect-vae.safetensors → VAE_SDXL_hdreffect_FP16.safetensors
    """

    base_model = file_info["base_model"]
    precision = file_info.get("precision", "Unknown")
    original_name = file_info["original_name"]

    # Wenn original_name leer ist oder nur das BaseModel → einfaches Format
    if not original_name or original_name == base_model:
        return f"VAE_{base_model}_{precision}.safetensors"

    # Spezialfälle: Version (Zahl) oder Variante (Image) → direkt zusammen
    # z.B. WAN + 2.1 = WAN2.1, Qwen + Image = QwenImage
    if re.match(r'^\d+\.?\d*$', original_name) or original_name == "Image":
        return f"VAE_{base_model}{original_name}_{precision}.safetensors"

    # Standard: Original-Name mit Unterstrich
    # z.B. Flux + ae = Flux_ae, SDXL + hdreffectvae-v10 = SDXL_hdreffectvae-v10
    return f"VAE_{base_model}_{original_name}_{precision}.safetensors"


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_file(file_path):
    """Komplette Analyse einer VAE-Datei"""

    # Convert Path to string if needed
    file_path = str(file_path)

    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    size_mb = file_size / (1024**2)

    # 1. Ist VAE?
    if not is_vae(file_path):
        return {
            "status": "SKIP",
            "reason": "Kein VAE (oder hat UNET/CLIP/LoRA)",
            "filename": filename,
            "size_mb": size_mb
        }

    # 2. Metadata lesen
    metadata = read_safetensors_metadata(file_path)

    # 3. Base Model Detection
    base_model = detect_base_model_from_metadata(metadata)
    if not base_model:
        base_model = detect_base_model_from_filename(filename)

    if base_model == "Unknown":
        return {
            "status": "SKIP",
            "reason": "Base Model nicht erkannt",
            "filename": filename,
            "size_mb": size_mb
        }

    # 4. Precision Detection
    precision = detect_precision_from_metadata(metadata)

    # 5. Name Extraction (Original Name bereinigt)
    original_name = extract_name_from_filename(filename, base_model)

    # 6. Generate proper name
    file_info = {
        "base_model": base_model,
        "precision": precision,
        "original_name": original_name
    }

    proper_name = generate_proper_name(file_info)

    # Return result
    return {
        "status": "PROCESSED",
        "current_path": file_path,
        "current_name": filename,
        "proper_name": proper_name,
        "base_model": base_model,
        "precision": precision,
        "original_name": original_name,
        "size_mb": size_mb,
        "target_folder": "vae"
    }


# ============================================================================
# EXPORTABLE FUNCTIONS (for all_modules.py batch processing)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads for Regular VAE + VAE Approx (for batch processing)"""
    files_to_install = []
    skipped = []

    # Scan for both types: .safetensors (VAE) + .pth/.pt (VAE Approx)
    all_files = []
    all_files.extend(downloads_path.glob("*.safetensors"))
    all_files.extend(downloads_path.glob("*.pth"))
    all_files.extend(downloads_path.glob("*.pt"))

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024**2)
        size_gb = size_mb / 1024

        # Try VAE Approx first
        is_approx, vae_type, approx_details = is_vae_approx(file_path)

        if is_approx:
            # VAE Approx detected
            files_to_install.append({
                'file_path': file_path,
                'filename': filename,
                'size_gb': size_gb,
                'vae_type': vae_type,  # "TAESD" or "Custom"
                'result': approx_details,
                'target_folder': VAE_APPROX_DIR,
                'type': 'vae_approx'  # Mark as VAE Approx for install_for_batch
            })
        else:
            # Try Regular VAE
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
                    'result': result,
                    'target_folder': VAE_DIR,
                    'type': 'vae'  # Mark as Regular VAE
                })

    return {
        'module_name': 'VAE (Regular + Approx)',
        'files': files_to_install,
        'skipped': skipped
    }

# ============================================================================
# VAE APPROX (TAESD) - DETECTION & PROCESSING
# ============================================================================

# Official TAESD patterns (strict naming required for ComfyUI auto-loading)
TAESD_PATTERNS = [
    # .pth (original format)
    'taesd_decoder.pth', 'taesd_encoder.pth',          # SD1/SD2
    'taesdxl_decoder.pth', 'taesdxl_encoder.pth',      # SDXL
    'taesd3_decoder.pth', 'taesd3_encoder.pth',        # SD3
    'taef1_decoder.pth', 'taef1_encoder.pth',          # Flux
    # .safetensors (HuggingFace mirrors)
    'taesd_decoder.safetensors', 'taesd_encoder.safetensors',
    'taesdxl_decoder.safetensors', 'taesdxl_encoder.safetensors',
    'taesd3_decoder.safetensors', 'taesd3_encoder.safetensors',
    'taef1_decoder.safetensors', 'taef1_encoder.safetensors',
]

def is_vae_approx(file_path):
    """Prüft ob Datei ein VAE Approx Model ist"""
    filename = file_path.name.lower()
    file_size = file_path.stat().st_size / (1024 * 1024)  # MB

    # 1. Size Check (< 10 MB)
    if file_size >= 10:
        return False, None, {}

    # 2. Extension Check
    if file_path.suffix.lower() not in ['.pth', '.pt', '.safetensors']:
        return False, None, {}

    # 3. TAESD Pattern (Official - Strict Naming)
    if filename in TAESD_PATTERNS:
        base_model = detect_taesd_base_model(filename)
        taesd_type = "Decoder" if "_decoder" in filename else "Encoder"

        return True, "TAESD", {
            'base_model': base_model,
            'type': taesd_type,
            'proper_name': file_path.name  # Keep original name
        }

    # 4. Custom VAE Approx (Filename Keywords)
    approx_keywords = ['vaeapprox', 'approx', 'taesd']
    if any(kw in filename for kw in approx_keywords):
        base_model = detect_approx_base_model(filename)
        source = extract_approx_source(filename, base_model)
        proper_name = f"vaeapprox_{base_model.lower()}_{source}{file_path.suffix}"

        return True, "Custom", {
            'base_model': base_model,
            'source': source,
            'proper_name': proper_name
        }

    return False, None, {}

def detect_taesd_base_model(filename):
    """Detects base model from TAESD filename"""
    if 'taesdxl' in filename:
        return 'SDXL'
    elif 'taesd3' in filename:
        return 'SD3'
    elif 'taef1' in filename:
        return 'Flux'
    elif 'taesd' in filename:
        return 'SD15'
    return 'Unknown'

def detect_approx_base_model(filename):
    """Detects base model from custom VAE Approx filename"""
    filename_lower = filename.lower()

    if 'flux' in filename_lower:
        return 'Flux'
    elif 'sdxl' in filename_lower:
        return 'SDXL'
    elif 'sd3' in filename_lower:
        return 'SD3'
    elif any(kw in filename_lower for kw in ['sd15', 'sd_v1', 'sd1']):
        return 'SD15'
    elif 'sd21' in filename_lower or 'sd_v2' in filename_lower:
        return 'SD21'
    return 'Unknown'

def extract_approx_source(filename, base_model):
    """Extracts source name from custom VAE Approx filename"""
    name = Path(filename).stem
    keywords_to_remove = [
        'vaeapprox', 'approx', 'vae',
        base_model.lower(),
        'flux', 'sdxl', 'sd15', 'sd3', 'sd21', 'sd_v1', 'sd_v2'
    ]

    name_lower = name.lower()
    name_lower = re.sub(r'[-_]+', ' ', name_lower)

    for keyword in keywords_to_remove:
        name_lower = re.sub(rf'\b{re.escape(keyword)}\b', ' ', name_lower)

    name_clean = re.sub(r'\s+', '-', name_lower).strip('-_')

    if not name_clean or len(name_clean) < 2:
        return 'custom'
    return name_clean

def scan_vae_approx_for_batch(downloads_path):
    """Scans downloads/ for VAE Approx files"""
    result_files = []

    for ext in ['*.pth', '*.pt', '*.safetensors']:
        for file_path in downloads_path.glob(ext):
            is_approx, vae_type, details = is_vae_approx(file_path)

            if is_approx:
                size_gb = file_path.stat().st_size / (1024 * 1024 * 1024)
                result_files.append({
                    'file_path': file_path,
                    'filename': file_path.name,
                    'size_gb': size_gb,
                    'vae_type': vae_type,
                    'result': details,
                    'target_folder': VAE_APPROX_DIR
                })

    return {
        'module_name': 'VAE Approx',
        'files': result_files
    }

def install_vae_approx_for_batch(files_list, keep_source):
    """Installs VAE Approx files"""
    installed = 0
    errors = 0
    collisions = 0

    VAE_APPROX_DIR.mkdir(parents=True, exist_ok=True)

    for file_info in files_list:
        source = file_info['file_path']
        proper_name = file_info['result']['proper_name']
        target = VAE_APPROX_DIR / proper_name

        success, final_path, msg = handle_duplicate_move(
            source=source,
            target=target,
            strategy="suffix",
            delete_source=(not keep_source),
            dry_run=False
        )

        if success:
            installed += 1
            if "collision" in msg.lower():
                collisions += 1
        else:
            errors += 1

    return {
        'installed': installed,
        'errors': errors,
        'collisions': collisions
    }

# ============================================================================
# MODUS A: INSTALLATION (Standalone mode)
# ============================================================================

def modus_a():
    """Modus A: Installiert VAEs + VAE Approx von downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="VAE (INSTALLATION)",
        downloads_path=DOWNLOADS_PATH,
        extensions="*.safetensors, *.pth, *.pt",
        module_type="VAE",
        target_folders="vae/, vae_approx/"
    )

    if not DOWNLOADS_PATH.exists():
        print(f"\n[ERROR] Downloads-Ordner nicht gefunden: {DOWNLOADS_PATH}")
        return

    # Scanne downloads/ für BEIDE Typen
    all_files = []
    all_files.extend(DOWNLOADS_PATH.glob("*.safetensors"))
    all_files.extend(DOWNLOADS_PATH.glob("*.pth"))
    all_files.extend(DOWNLOADS_PATH.glob("*.pt"))

    if not all_files:
        print_no_files_found("VAE files")
        return

    # ========================================================================
    # PHASE 1: ANALYZE (silently collect)
    # ========================================================================
    files_to_install = []
    skipped = []

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024**2)
        size_gb = size_mb / 1024

        # Try VAE Approx first (faster check)
        is_approx, vae_type, approx_details = is_vae_approx(file_path)

        if is_approx:
            files_to_install.append({
                'path': file_path,
                'filename': filename,
                'type': 'vae_approx',
                'vae_type': vae_type,
                'result': approx_details,
                'size_mb': size_mb,
                'size_gb': size_gb
            })
        else:
            # Try Regular VAE
            result = analyze_file(file_path)

            if result["status"] == "SKIP":
                skipped.append({'filename': filename, 'size_mb': size_mb})
            else:
                files_to_install.append({
                    'path': file_path,
                    'filename': filename,
                    'type': 'vae',
                    'result': result,
                    'size_mb': size_mb,
                    'size_gb': size_gb
                })

    # ========================================================================
    # ANALYSIS (using shared helper)
    # ========================================================================
    print_analysis(len(all_files), len(files_to_install), len(skipped))

    # ========================================================================
    # SKIPPED SECTION
    # ========================================================================
    print_skipped_section(
        skipped_files=skipped,
        skip_reason="Not a VAE file (no VAE keys found)",
        max_show=5
    )

    # ========================================================================
    # PREVIEW SECTION
    # ========================================================================
    if not files_to_install:
        print_no_files_to_install()
        return

    print_preview_header(len(files_to_install))

    for i, file_info in enumerate(files_to_install, 1):
        filename = file_info['filename']
        size_mb = file_info['size_mb']
        result = file_info['result']
        file_type = file_info['type']

        if file_type == 'vae_approx':
            vae_type = file_info.get('vae_type', 'Custom')
            if vae_type == "TAESD":
                detected = f"TAESD (Official), {result['base_model']}, {result['type']}"
            else:
                detected = f"VAE Approx (Custom), {result['base_model']}"
            target = f"vae_approx/{result['proper_name']}"
        else:
            detected = f"{result['base_model']}, {result['precision']}"
            target = f"vae/{result['proper_name']}"

        print_preview_item(
            index=i,
            filename=filename,
            size_mb=size_mb,
            detected_info=detected,
            target_path=target
        )

    # Calculate total size
    total_size = sum(f['size_gb'] for f in files_to_install)
    print_total(len(files_to_install), total_size)

    # ========================================================================
    # PHASE 2: DELETE/KEEP
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

    # Ensure target folders exist
    VAE_PATH.mkdir(parents=True, exist_ok=True)
    VAE_APPROX_DIR.mkdir(parents=True, exist_ok=True)

    installed = 0
    errors = 0
    collisions = 0

    for idx, file_info in enumerate(files_to_install, 1):
        file_path = file_info['path']
        result = file_info['result']
        file_type = file_info['type']
        filename = file_path.name

        # Choose target folder based on type
        if file_type == 'vae_approx':
            target_path = VAE_APPROX_DIR / result['proper_name']
        else:  # Regular VAE
            target_path = VAE_PATH / result['proper_name']

        success, final_path, msg = handle_duplicate_move(
            source=file_path,
            target=target_path,
            strategy="suffix",
            delete_source=(not keep_source),
            dry_run=False
        )

        if success:
            if "collision" in msg.lower():
                collisions += 1
            installed += 1
        else:
            errors += 1

        print_install_item(idx, len(files_to_install), filename, success, msg)

    # Final Summary (using shared helper)
    print_summary(len(files_to_install), installed, collisions, errors, keep_source)


# ============================================================================
# MODUS B: REINSTALL/CHECK
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check/Korrektur in vae/

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="VAE",
        folders="vae/",
        extensions="*.safetensors",
        module_type="VAE",
        target_folders="vae/",
        preview_mode=preview_mode
    )

    if not VAE_PATH.exists():
        print(f"[ERROR] VAE folder not found: {VAE_PATH}")
        return

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        if misplaced:
            for file_path in misplaced:
                if is_vae(file_path):
                    filename = file_path.name
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"[RESCUE] Found misplaced VAE: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                    result = analyze_file(file_path)

                    if result["status"] != "SKIP":
                        target_path = VAE_PATH / result['proper_name']

                        success, final_path, msg = handle_duplicate_move(
                            file_path,
                            target_path,
                            expected_target_name=result['proper_name'],
                            mode="B",
                            keep_source_option=False,
                            dry_run=False
                        )

                        if success:
                            print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: vae/{final_path.name}")
                            remove_misplaced_file(file_path)
                            rescued += 1
                        else:
                            print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                    print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced VAE(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDER
    # ========================================================================
    all_files = list(VAE_PATH.glob("*.safetensors"))

    if not all_files:
        print_no_files_found("VAE files")
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

        result = analyze_file(file_path)

        if result["status"] == "SKIP":
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, result['reason']))
            continue

        base_model = result["base_model"]
        precision = result["precision"]
        current_name = result["current_name"]
        proper_name = result["proper_name"]

        if scan_only:
            continue

        if current_name != proper_name:
            problems_list.append(('wrong_name', filename, file_size_mb, base_model, precision, proper_name, file_path))
        else:
            correct_files.append(("vae", filename, file_size_mb))

    # ========================================================================
    # SCAN-ONLY MODE: Just return after building queue
    # ========================================================================
    if scan_only:
        return

    # ========================================================================
    # ANALYSIS (using shared helper)
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
                    warning=f"Not a VAE file: {reason}"
                )

            elif problem_type == 'wrong_name':
                _, fname, size_mb, model, prec, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{model}, {prec}",
                    target_path=f"vae/{proper_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    new_path = VAE_PATH / proper_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python modul2_vae.py <A|B> [--run] [--scan-only]")
        print("  A           = Modus A (Installation)")
        print("  B           = Modus B (Reinstall/Check)")
        print("  --run       = Echt ausfuehren (ohne --run = DRY RUN)")
        print("  --scan-only = PASS 1: Nur Queue aufbauen (nur Modus B)")
        sys.exit(1)

    mode = sys.argv[1].upper()
    scan_only = "--scan-only" in sys.argv

    if mode == "A":
        modus_a()
    elif mode == "B":
        preview = "--preview" in sys.argv
        batch = "--batch" in sys.argv
        modus_b(scan_only=scan_only, batch_mode=batch, preview_mode=preview)
    else:
        print(f"[ERROR] Unbekannter Modus: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
