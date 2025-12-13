#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 1 - Base Models (Checkpoints/UNET/GGUF)
Vollständige Implementation mit Modus A und B
"""

import os
import sys
import json
import struct
import shutil
from pathlib import Path
from collections import defaultdict

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
    CHECKPOINTS_DIR,
    UNET_DIR,
    DIFFUSION_MODELS_DIR,
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
CHECKPOINTS_PATH = CHECKPOINTS_DIR
UNET_PATH = UNET_DIR
DIFFUSION_MODELS_PATH = DIFFUSION_MODELS_DIR

# Größen-Grenze für Base Models
MIN_BASE_MODEL_SIZE = 1_000_000_000  # 1GB

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


def analyze_keys(keys):
    """Analysiert Keys für UNET/CLIP/VAE/WAN/Z-Image/Qwen-Edit/Lotus Detection"""

    unet_patterns = [
        "model.diffusion_model", "down_blocks", "up_blocks", "mid_block",
        "double_blocks", "single_blocks", "input_blocks", "output_blocks",
        "middle_block"
    ]

    clip_patterns = [
        "cond_stage_model", "text_model",
        "clip_l", "clip_g", "conditioner.embedders"
    ]

    # Z-Image specific: Qwen3 Text Encoder (NICHT CLIP!)
    # Pattern: text_encoders.qwen3_4b.* → NICHT als CLIP zählen!
    qwen_patterns = ["text_encoders.qwen", "qwen3_4b"]

    vae_patterns = [
        "first_stage_model", "decoder.conv_in", "encoder.conv_in", "vae"
    ]

    # WAN Video Model patterns (blocks.N.cross_attn)
    # WAN hat "blocks.0.cross_attn" OHNE "double_blocks" oder "single_blocks"
    has_wan = any("blocks." in k and "cross_attn" in k for k in keys)
    has_flux_blocks = any("double_blocks" in k or "single_blocks" in k for k in keys)

    # WAN = hat blocks.*.cross_attn aber NICHT Flux-style blocks
    is_wan = has_wan and not has_flux_blocks

    # Z-Image Detection (NextDiT Architektur)
    # Unique Keys: context_refiner, noise_refiner, cap_embedder
    is_zimage = any("context_refiner" in k or "noise_refiner" in k or "cap_embedder" in k for k in keys)

    # Qwen-Image-Edit Detection
    # Unique Key: model.diffusion_model.txt_in (hat auch img_in)
    is_qwen_edit = any("model.diffusion_model.txt_in" in k for k in keys)

    # Lotus-Depth Detection
    # Unique Keys: class_embedding + down_blocks.*.attentions (UNet-style aber für Depth)
    has_class_embedding = any("class_embedding" in k for k in keys)
    has_down_attentions = any("down_blocks." in k and ".attentions." in k for k in keys)
    is_lotus = has_class_embedding and has_down_attentions

    has_unet = any(any(p in k for p in unet_patterns) for k in keys)

    # WAN, Z-Image, Qwen-Edit, Lotus zählen auch als "hat UNET" für is_main_model()
    if is_wan or is_zimage or is_qwen_edit or is_lotus:
        has_unet = True

    # CLIP Detection: NUR wenn NICHT Qwen3!
    has_qwen = any(any(p in k for p in qwen_patterns) for k in keys)
    if has_qwen:
        # Z-Image hat Qwen3 statt CLIP!
        has_clip = False
    else:
        # Standard CLIP detection (inkl. text_encoder Pattern)
        has_clip = any(any(p in k for p in clip_patterns) for k in keys)
        # Legacy: auch text_encoder (aber nur wenn NICHT Qwen!)
        if not has_clip:
            has_clip = any("text_encoder" in k for k in keys)

    has_vae = any(any(p in k for p in vae_patterns) for k in keys)

    return has_unet, has_clip, has_vae


def is_main_model(file_path):
    """Prüft ob Datei ein Base Model ist (Module Boundary)"""

    # 1. Größe < 1GB? → Kann kein Base Model sein
    file_size = os.path.getsize(file_path)
    if file_size < MIN_BASE_MODEL_SIZE:
        return False

    # 2. GGUF? → Immer Base Model
    if file_path.endswith(".gguf"):
        return True

    # 3. Safetensors: Hat UNET?
    if file_path.endswith(".safetensors"):
        keys = read_safetensors_keys(file_path)
        has_unet, _, _ = analyze_keys(keys)
        return has_unet

    # 4. Andere Extensions → kein Base Model
    return False


# ============================================================================
# PHASE 2: BASE MODEL & COMPONENT DETECTION
# ============================================================================

def detect_base_model_from_metadata(metadata):
    """Erkennt Base Model aus Metadata"""

    # Check verschiedene Metadata-Felder
    fields = [
        "ss_base_model_version",
        "ss_sd_model_name",
        "modelspec.architecture",
        "base_model"
    ]

    for field in fields:
        value = metadata.get(field, "").lower()

        # Z-Image (NEU!)
        if "zimage" in value or "z-image" in value or "z_image" in value:
            if "turbo" in value:
                return "ZImageTurbo"
            elif "edit" in value:
                return "ZImageEdit"
            elif "base" in value:
                return "ZImageBase"
            else:
                return "ZImageTurbo"  # Default

        # Flux
        if "flux" in value:
            if "schnell" in value:
                return "FluxS"
            elif "dev" in value:
                return "FluxD"
            elif "pro" in value:
                return "FluxP"
            else:
                return "FluxD"  # Default

        # SDXL
        if "sdxl" in value or "xl" in value:
            if "turbo" in value:
                return "SDXLTurbo"
            elif "lightning" in value:
                return "SDXLLightning"
            elif "refiner" in value:
                return "SDXLRefiner"
            else:
                return "SDXL"

        # Pony
        if "pony" in value:
            return "Pony"

        # Illustrious
        if "illustrious" in value:
            return "Illustrious"

        # NoobAI
        if "noob" in value:
            return "NoobAI"

        # Animagine
        if "animagine" in value:
            return "Animagine"

        # SD Classic
        if "sd15" in value or "v1-5" in value or "sd_v1" in value:
            return "SD15"
        if "sd21" in value or "v2-1" in value or "sd_v2" in value:
            return "SD21"
        if "sd3" in value or "sd35" in value:
            return "SD3"

    return None


def detect_base_model_from_filename(filename):
    """Erkennt Base Model aus Filename"""
    filename_lower = filename.lower()

    # Z-Image (NEU!)
    if "zimage" in filename_lower or "z-image" in filename_lower or "z_image" in filename_lower:
        if "turbo" in filename_lower:
            return "ZImageTurbo"
        elif "edit" in filename_lower:
            return "ZImageEdit"
        elif "base" in filename_lower:
            return "ZImageBase"
        else:
            return "ZImageTurbo"  # Default

    # Flux
    if "flux" in filename_lower:
        if "schnell" in filename_lower:
            return "FluxS"
        elif "dev" in filename_lower:
            return "FluxD"
        elif "pro" in filename_lower:
            return "FluxP"
        elif "fill" in filename_lower:
            return "FluxF"
        elif "ultra" in filename_lower:
            return "FluxU"
        else:
            return "FluxD"  # Default

    # SDXL
    if "sdxl" in filename_lower or "xl" in filename_lower:
        if "turbo" in filename_lower:
            return "SDXLTurbo"
        elif "lightning" in filename_lower:
            return "SDXLLightning"
        elif "refiner" in filename_lower:
            return "SDXLRefiner"
        else:
            return "SDXL"

    # Pony
    if "pony" in filename_lower:
        return "Pony"

    # Illustrious
    if "illustrious" in filename_lower:
        return "Illustrious"

    # NoobAI
    if "noob" in filename_lower:
        return "NoobAI"

    # Animagine
    if "animagine" in filename_lower:
        return "Animagine"

    # SD Classic
    if "sd15" in filename_lower or "sd_15" in filename_lower or "sd-15" in filename_lower:
        return "SD15"
    if "sd21" in filename_lower or "sd_21" in filename_lower or "sd-21" in filename_lower:
        return "SD21"
    if "sd3" in filename_lower or "sd35" in filename_lower:
        return "SD3"

    return None


def detect_base_model_from_keys(keys):
    """Erkennt Base Model aus Key Patterns (Fallback)"""

    # Z-Image: context_refiner ODER cap_embedder (UNIQUE! HÖCHSTE PRIORITÄT!)
    if any("context_refiner" in k for k in keys):
        return "ZImage"  # Basis-Erkennung, Variante aus Filename

    if any("cap_embedder" in k for k in keys):
        return "ZImage"

    # Alternative: Qwen3 Text Encoder Check
    if any("qwen3_4b" in k for k in keys):
        return "ZImage"

    # Qwen-Image-Edit: txt_in OHNE double_blocks/single_blocks (die hat Flux!)
    # WICHTIG: Flux hat auch txt_in, aber zusätzlich double_blocks/single_blocks
    has_txt_in = any("model.diffusion_model.txt_in" in k for k in keys)
    has_flux_structure = any("double_blocks" in k or "single_blocks" in k for k in keys)
    if has_txt_in and not has_flux_structure:
        return "QwenImageEdit"

    # Lotus-Depth: class_embedding + down_blocks.*.attentions (UNIQUE Kombination!)
    # UNet-ähnliche Struktur aber für Depth Estimation
    has_class_embedding = any("class_embedding" in k for k in keys)
    has_down_attentions = any("down_blocks." in k and ".attentions." in k for k in keys)
    if has_class_embedding and has_down_attentions:
        return "LotusDepth"

    # WAN Video Models: blocks.*.cross_attn OHNE double_blocks/single_blocks
    # WICHTIG: VOR Flux prüfen! WAN hat auch "blocks." aber andere Struktur
    has_wan_blocks = any("blocks." in k and "cross_attn" in k for k in keys)
    has_flux_blocks = any("double_blocks" in k or "single_blocks" in k for k in keys)
    if has_wan_blocks and not has_flux_blocks:
        # WAN erkannt - Typ (I2V vs T2V) später bestimmen
        return "WAN"

    # Flux: double_blocks, single_blocks
    if any("double_blocks" in k or "single_blocks" in k for k in keys):
        return "FluxD"  # Default to dev

    # SDXL/Pony: conditioner.embedders
    if any("conditioner.embedders" in k for k in keys):
        return "SDXL"  # Kann nicht zwischen SDXL/Pony unterscheiden

    # SD15/SD21: cond_stage_model.transformer
    if any("cond_stage_model.transformer" in k for k in keys):
        return "SD15"  # Default

    return None


def detect_component_variant(has_unet, has_clip, has_vae):
    """Bestimmt Component-Variante"""

    if has_unet and has_clip and has_vae:
        return "Full"
    elif has_unet and has_clip and not has_vae:
        return "NoVAE"
    elif has_unet and not has_clip and has_vae:
        return "NoCLIP"
    elif has_unet and not has_clip and not has_vae:
        return "UNET-only"
    else:
        return "Unknown"


def detect_precision_from_tensors(metadata):
    """Erkennt Precision aus Tensor dtypes"""

    dtypes = {}
    for key, value in metadata.items():
        if isinstance(value, dict) and "dtype" in value:
            dtype = value["dtype"]
            dtypes[dtype] = dtypes.get(dtype, 0) + 1

    if not dtypes:
        return None

    # FP8 hat Priorität - wenn vorhanden, ist es FP8 quantized
    # (auch wenn gemischt mit F32 scale factors)
    for dtype in dtypes.keys():
        if "F8" in dtype or "E4M3" in dtype or "E5M2" in dtype:
            return "FP8"

    # Dominanten dtype finden (>50%)
    total = sum(dtypes.values())
    for dtype, count in dtypes.items():
        if count / total > 0.5:
            # Mapping (WICHTIG: Längere Strings zuerst prüfen!)
            if "BF16" in dtype or "bfloat16" in dtype:
                return "BF16"
            elif "F32" in dtype or "float32" in dtype:
                return "FP32"
            elif "F16" in dtype or "float16" in dtype:
                return "FP16"

    return "Mixed"


# ============================================================================
# WAN VIDEO MODEL DETECTION
# ============================================================================

def detect_wan_type(keys):
    """Erkennt WAN Model Type: I2V (Image-to-Video) vs T2V (Text-to-Video)

    Returns:
        str: 'I2V' oder 'T2V'
    """
    # I2V hat img_emb oder k_img/v_img keys
    has_img_emb = any("img_emb" in k for k in keys)
    has_k_img = any("k_img" in k or "v_img" in k for k in keys)

    if has_img_emb or has_k_img:
        return "I2V"

    # T2V hat fps_ keys
    has_fps = any("fps_" in k for k in keys)
    if has_fps:
        return "T2V"

    # Default: I2V (häufiger)
    return "I2V"


def detect_wan_version(filename, metadata):
    """Erkennt WAN Version aus Metadata oder Filename

    Priorität: Metadata > Filename > None

    Returns:
        str or None: '2.1', '2.2', '2.5', '3.0' oder None
    """
    filename_lower = filename.lower()

    # 1. METADATA (höchste Priorität - wenn vorhanden)
    if metadata:
        meta_str = str(metadata).lower()
        if 'wan2.2' in meta_str or 'wan22' in meta_str or 'wan_2.2' in meta_str:
            return '2.2'
        if 'wan2.1' in meta_str or 'wan21' in meta_str or 'wan_2.1' in meta_str:
            return '2.1'
        if 'wan2.5' in meta_str or 'wan25' in meta_str or 'wan_2.5' in meta_str:
            return '2.5'
        if 'wan3.0' in meta_str or 'wan30' in meta_str or 'wan_3.0' in meta_str:
            return '3.0'

    # 2. FILENAME (sichere explizite Patterns)
    if any(x in filename_lower for x in ['wan2.1', 'wan21', 'wan2_1']):
        return '2.1'
    if any(x in filename_lower for x in ['wan2.2', 'wan22', 'wan2_2']):
        return '2.2'
    if any(x in filename_lower for x in ['wan2.5', 'wan25', 'wan2_5']):
        return '2.5'
    if any(x in filename_lower for x in ['wan3.0', 'wan30', 'wan3_0']):
        return '3.0'

    # 3. FALLBACK - keine Version erkannt
    return None


def detect_wan_size(metadata):
    """Erkennt WAN Model Size aus Parameter Count

    Berechnet Total Parameters aus Tensor Shapes.

    Returns:
        str: '14B', '1.3B' oder 'Unknown'
    """
    total_params = 0

    for key, value in metadata.items():
        if key == "__metadata__":
            continue
        if isinstance(value, dict) and "shape" in value:
            shape = value["shape"]
            params = 1
            for dim in shape:
                params *= dim
            total_params += params

    if total_params == 0:
        return "Unknown"

    # Size Ranges (basierend auf Analyse)
    # 14B: ~14-17B params
    # 1.3B: ~1.3-1.5B params
    if total_params > 10_000_000_000:  # >10B
        return "14B"
    elif total_params > 1_000_000_000:  # >1B
        return "1.3B"
    else:
        return "Unknown"


def extract_wan_variant_name(filename):
    """Extrahiert Varianten-Namen aus WAN Filename

    Neue Strategie: Entferne alle technischen Teile, behalte den Rest
    So funktioniert es für JEDEN Namen - egal wie kreativ der User war.

    Args:
        filename: Original filename

    Returns:
        tuple: (variant_name, version, resolution)
    """
    import re

    # Entferne Extension
    name = os.path.splitext(filename)[0]
    original_name = name  # Für CamelCase-Extraktion
    name_lower = name.lower()

    # Version extrahieren - NUR echte Versions-Marker (v2, V2, nicht 14B!)
    # Auch V2 am Ende ohne Trennzeichen (wan22...V2FP8)
    version_match = re.search(r'(?:^|[_\-\s])v(\d+(?:\.\d+)?)(?:[_\-\s\.]|fp|$)', name_lower)
    if not version_match:
        # Auch "V2" direkt vor FP8 etc.
        version_match = re.search(r'v(\d+)(?:fp|$)', name_lower)
    version = f"v{version_match.group(1)}" if version_match else None

    # Auflösung extrahieren (480p, 540p, 720p, 1080p) - VOR dem Entfernen
    resolution_match = re.search(r'(480|540|720|1080)p', name_lower)
    resolution = f"{resolution_match.group(1)}P" if resolution_match else None

    # Technische Teile die ENTFERNT werden (werden zu Leerzeichen)
    tech_patterns = [
        r'wan2?[._]?[12]?[._]?[0-9]?',  # wan, wan2, wan21, wan2.1, wan22, wan2.2
        r'wanai',
        r'\bai\b',  # "AI" alleine
        r'i2v|t2v|img2vid|txt2vid',
        r'14b|1[._]?3b|5b',
        r'720p|480p|540p|1080p',
        r'fp32|fp16|bf16|fp8[a-z0-9]*',
        r'chkp|checkpoint',
        r'safetensors|diffusion',
        r'onthefly',
        r'videomodel',  # "videomodel" zusammen
        r'video\s*model',  # "video model"
        r'\bvideo\b',  # "video" als ganzes Wort
        r'\bmodel\b',
        r'20gb|14gb|10gb|[0-9]+gb',  # Größenangaben
        r'v\d+(?:fp|$)',  # Version vor FP (V2FP8)
        r'(?:^|[_\-\s])v\d+(?:\.\d+)?(?:[_\-\s\.]|$)',  # Versions (schon extrahiert)
        # Spezial-Keywords die wir separat extrahieren
        r'nsfw',
        r'fastmove',
        r'enhanced',
        r'cameraprompt',
        r'camera',
        r'prompt',
        r'\bhigh\b',
        r'\bdf\b',
    ]

    clean = name_lower
    for pattern in tech_patterns:
        clean = re.sub(pattern, ' ', clean, flags=re.IGNORECASE)

    # Separatoren normalisieren
    clean = re.sub(r'[_\-\.]+', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # CamelCase aus Original splitten (RadiantCrushHigh -> Radiant Crush High)
    # Aber zusammengesetzte Namen wie SkyReels behalten
    camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', original_name)
    camel_words = re.findall(r'[A-Z][a-z]+', camel_split)
    camel_map = {w.lower(): w for w in camel_words}  # lowercase -> Original

    # Bekannte zusammengesetzte Namen die nicht gesplittet werden sollen
    compound_names = {
        'skyreels': 'SkyReels',
        'radiantcrush': 'RadiantCrush',
        'fastmove': 'FastMove',
        'cameraprompt': 'CameraPrompt',
    }
    camel_map.update(compound_names)

    # Wörter sammeln und filtern
    words = clean.split()
    filtered = []
    skip_words = ['the', 'and', 'for', 'with', 'new', 'nsfw', 'sfw', 'nsfwfastmove', 'nsfwfastmovev2', 'video', 'sky']

    for word in words:
        # Skip wenn zu kurz, nur Zahlen
        if len(word) < 2 or word.isdigit():
            continue
        if word in skip_words:
            continue

        # Versuche CamelCase innerhalb des Wortes zu splitten
        # z.B. "radiantcrushhigh" -> suche nach bekannten CamelCase-Teilen
        found_camel = False
        for camel in sorted(camel_map.keys(), key=len, reverse=True):
            if camel in word and len(camel) > 3:
                # Gefunden! Splitte das Wort
                idx = word.find(camel)
                before = word[:idx]
                after = word[idx + len(camel):]
                parts = []
                if before and len(before) >= 2:
                    parts.append(before.capitalize())
                parts.append(camel_map[camel])
                if after and len(after) >= 2:
                    # Auch den Rest prüfen
                    if after in camel_map:
                        parts.append(camel_map[after])
                    else:
                        parts.append(after.capitalize())
                filtered.extend(parts)
                found_camel = True
                break

        if not found_camel:
            filtered.append(word)

    # Spezielle Keywords die immer drin sein sollen (wenn vorhanden)
    special_keywords = []
    if 'nsfw' in name_lower:
        special_keywords.append('NSFW')
    if 'sfw' in name_lower and 'nsfw' not in name_lower:
        special_keywords.append('SFW')
    if 'fastmove' in name_lower:
        special_keywords.append('FastMove')
    if 'lightning' in name_lower:
        special_keywords.append('Lightning')
    if 'turbo' in name_lower:
        special_keywords.append('Turbo')
    if 'enhanced' in name_lower:
        special_keywords.append('Enhanced')
    if 'cameraprompt' in name_lower or 'camera' in name_lower:
        special_keywords.append('CameraPrompt')
    if re.search(r'\bdf\b', name_lower):
        special_keywords.append('DF')
    if re.search(r'\bhigh\b', name_lower):
        special_keywords.append('High')

    # Formatiere normale Wörter mit Original-CamelCase oder compound_names
    formatted = []
    skip_lower = [k.lower() for k in special_keywords] + ['v2', 'v1', 'v3']  # Versions nicht als Teil des Namens
    for word in filtered:
        # Skip wenn es ein special keyword oder Version ist
        if word.lower() in skip_lower:
            continue
        # Prüfe compound_names und camel_map
        if word.lower() in camel_map:
            formatted.append(camel_map[word.lower()])
        elif word in camel_map:
            formatted.append(camel_map[word])
        else:
            formatted.append(word.capitalize())

    # Kombiniere: erst normale Wörter, dann special keywords
    all_parts = formatted[:3] + special_keywords[:2]

    if all_parts:
        variant_name = "-".join(all_parts)
    else:
        variant_name = "Custom"

    # Kürzen wenn zu lang
    if len(variant_name) > 40:
        variant_name = variant_name[:40].rsplit('-', 1)[0]

    return variant_name, version, resolution


def skip_metadata_value(f, value_type, version=3):
    """
    Überspringt einen Metadata-Value in GGUF Datei

    Args:
        f: File handle
        value_type: Metadata value type (0-10)
        version: GGUF version (3)

    Returns:
        bool: True wenn erfolgreich, False bei Fehler
    """

    TYPE_SIZES = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8
    }

    try:
        # Scalar types
        if value_type in TYPE_SIZES:
            f.read(TYPE_SIZES[value_type])
            return True

        # STRING
        elif value_type == 8:
            str_len_bytes = f.read(8)
            if len(str_len_bytes) < 8:
                return False
            str_len = struct.unpack('<Q', str_len_bytes)[0]

            if str_len > 1_000_000:  # Sanity check
                return False

            f.read(str_len)
            return True

        # ARRAY (RECURSIVE!)
        elif value_type == 9:
            elem_type_bytes = f.read(4)
            if len(elem_type_bytes) < 4:
                return False
            elem_type = struct.unpack('<I', elem_type_bytes)[0]

            arr_len_bytes = f.read(8)
            if len(arr_len_bytes) < 8:
                return False
            arr_len = struct.unpack('<Q', arr_len_bytes)[0]

            if arr_len > 1_000_000:  # Sanity check
                return False

            # Skip each element RECURSIVELY
            for _ in range(arr_len):
                if not skip_metadata_value(f, elem_type, version):
                    return False

            return True

        else:
            return False

    except Exception:
        return False


def read_metadata_and_skip(file_path):
    """
    Liest general.file_type aus Metadata und überspringt Metadata-Sektion

    Returns:
        tuple: (metadata_quant, tensor_info_position)
               metadata_quant: str or None ("Q5_1", "Q8_0", etc.)
               tensor_info_position: int (byte position nach Metadata)
    """

    file_type_map = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    }

    metadata_quant = None

    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return None, None

            version = struct.unpack('<I', f.read(4))[0]

            if version == 3:
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            else:
                tensor_count = struct.unpack('<I', f.read(4))[0]
                metadata_kv_count = struct.unpack('<I', f.read(4))[0]

            # If no metadata, tensor info starts at byte 24
            if metadata_kv_count == 0:
                return None, 24

            # Process each metadata entry
            for i in range(metadata_kv_count):
                key_len_size = 8 if version == 3 else 4
                key_len_bytes = f.read(key_len_size)
                if len(key_len_bytes) < key_len_size:
                    return None, None

                key_len = struct.unpack('<Q' if version == 3 else '<I', key_len_bytes)[0]

                if key_len > 10000:  # Sanity check
                    return None, None

                key = f.read(key_len).decode('utf-8', errors='ignore')

                value_type_bytes = f.read(4)
                if len(value_type_bytes) < 4:
                    return None, None
                value_type = struct.unpack('<I', value_type_bytes)[0]

                # Special case: general.file_type
                if key == "general.file_type" and value_type == 4:
                    value = struct.unpack('<I', f.read(4))[0]
                    metadata_quant = file_type_map.get(value, f"Q{value}")
                else:
                    if not skip_metadata_value(f, value_type, version):
                        return None, None

            tensor_info_position = f.tell()
            return metadata_quant, tensor_info_position

    except Exception:
        return None, None


def analyze_tensor_types_from_position(file_path, start_position, tensor_count):
    """
    Analysiert Tensor-Typen ab gegebener Position

    Args:
        file_path: Path zur GGUF Datei
        start_position: Byte-Position wo Tensor-Info beginnt
        tensor_count: Anzahl der Tensors

    Returns:
        str: Dominant Quantization Type ("Q8_0", "Q5_1", etc.)
    """

    GGML_TYPE_NAMES = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
        6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
        10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    }

    tensor_types = []
    tensor_sizes = []

    try:
        with open(file_path, 'rb') as f:
            f.seek(start_position)

            for i in range(tensor_count):
                name_len_bytes = f.read(8)
                if len(name_len_bytes) < 8:
                    break
                name_len = struct.unpack('<Q', name_len_bytes)[0]

                if name_len > 10000:  # Sanity check
                    break

                f.read(name_len)  # Skip name

                n_dims_bytes = f.read(4)
                if len(n_dims_bytes) < 4:
                    break
                n_dims = struct.unpack('<I', n_dims_bytes)[0]

                if n_dims > 10:  # Sanity check
                    break

                dims = []
                for _ in range(n_dims):
                    dim_bytes = f.read(8)
                    if len(dim_bytes) < 8:
                        break
                    dim = struct.unpack('<Q', dim_bytes)[0]
                    dims.append(dim)

                type_bytes = f.read(4)
                if len(type_bytes) < 4:
                    break
                tensor_type = struct.unpack('<I', type_bytes)[0]

                f.read(8)  # Skip offset

                # Calculate element count
                element_count = 1
                for dim in dims:
                    element_count *= dim

                tensor_types.append(tensor_type)
                tensor_sizes.append(element_count)

        if not tensor_types:
            return None

        # Weight by size (WICHTIG!)
        type_weighted = {}
        total_elements = sum(tensor_sizes)

        for tensor_type, size in zip(tensor_types, tensor_sizes):
            type_weighted[tensor_type] = type_weighted.get(tensor_type, 0) + size

        # Find dominant type
        dominant_type = max(type_weighted.items(), key=lambda x: x[1])[0]
        dominant_percentage = (type_weighted[dominant_type] / total_elements) * 100

        if dominant_percentage > 70:
            return GGML_TYPE_NAMES.get(dominant_type, None)
        elif dominant_percentage > 40:
            return GGML_TYPE_NAMES.get(dominant_type, None)
        else:
            return "Mixed"

    except Exception:
        return None


def detect_gguf_quantization(file_path):
    """
    Erkennt GGUF Quantization aus Tensor-Analyse

    Returns aktuelle Quantisierung aus Tensors (Ground Truth).
    Ignoriert Metadata wenn es nicht mit Tensors übereinstimmt.

    Returns:
        str: Quantization ("Q8_0", "Q3_K_M", "Unknown")
    """

    # 1. Lese Metadata (für Fallback)
    metadata_quant, tensor_info_pos = read_metadata_and_skip(file_path)

    # 2. Lese Header für tensor_count
    try:
        with open(file_path, 'rb') as f:
            f.read(4)  # magic
            version = struct.unpack('<I', f.read(4))[0]
            if version == 3:
                tensor_count = struct.unpack('<Q', f.read(8))[0]
            else:
                tensor_count = struct.unpack('<I', f.read(4))[0]
    except:
        return "Unknown"

    # 3. Analysiere Tensors (PRIORITÄT - Ground Truth!)
    if tensor_info_pos:
        tensor_quant = analyze_tensor_types_from_position(
            file_path,
            tensor_info_pos,
            tensor_count
        )
        if tensor_quant:
            return tensor_quant  # Einfach Wahrheit zurückgeben!

    # 4. Fallback zu Metadata (wenn Tensor-Analyse fehlschlägt)
    if metadata_quant:
        return metadata_quant

    return "Unknown"



# ============================================================================
# PHASE 3: NAMING
# ============================================================================

def extract_category(filename):
    """Extrahiert Category aus Filename (vereinfacht)"""

    filename_lower = filename.lower()

    # NSFW Keywords
    nsfw_keywords = ["nsfw", "porn", "hentai", "xxx", "nude", "sex"]
    if any(kw in filename_lower for kw in nsfw_keywords):
        return "NSFW"

    # Realism Keywords
    real_keywords = ["real", "photo", "realistic"]
    if any(kw in filename_lower for kw in real_keywords):
        return "Realism"

    # Anime Keywords
    anime_keywords = ["anime", "cartoon", "2d"]
    if any(kw in filename_lower for kw in anime_keywords):
        return "Anime"

    return "General"


def get_zimage_semantic_name(filename, base_model, has_unet, has_clip, has_vae):
    """Generiert semantischen Namen für Z-Image Models (Option A)

    Args:
        filename: Original filename
        base_model: Detected base model (ZImageTurbo, ZImageBase, ZImageEdit)
        has_unet: Has UNET component
        has_clip: Has CLIP component
        has_vae: Has VAE component

    Returns:
        str: Semantic name or None (use standard extraction)
    """

    filename_lower = filename.lower()

    # AIO Detection: "aio" im Filename
    has_aio = 'aio' in filename_lower

    # UNET-only: UNET=True, CLIP=False, VAE=False
    is_unet_only = has_unet and not has_clip and not has_vae

    # Z-Image Turbo
    if base_model == 'ZImageTurbo':
        if has_aio:
            return 'TurboAIO'
        elif is_unet_only:
            return 'Turbo'
        # Else: Finetuned/Community → Standard extraction

    # Z-Image Base
    elif base_model == 'ZImageBase':
        if has_aio:
            return 'Foundation'
        else:
            return 'Base'

    # Z-Image Edit
    elif base_model == 'ZImageEdit':
        if has_aio:
            return 'ImageEdit'
        else:
            return 'Edit'

    # Fallback: Use standard extraction
    return None


def extract_name_version(filename):
    """Extrahiert Name und Version aus Filename"""
    import re

    # Entferne Extension
    name = os.path.splitext(filename)[0]
    name_lower = name.lower()

    # 1. GRÖSSE extrahieren (z.B. 23GB, 17GB, 12GByte)
    size_match = re.search(r'(\d+)\s*(gb|gbyte)', name_lower, re.IGNORECASE)
    size_part = None
    if size_match:
        size_part = f"{size_match.group(1)}GB"  # Normalisiert zu "23GB"
        name_lower = name_lower.replace(size_match.group(0), " ")

    # 2. VERSION extrahieren (z.B. v10, v2.5)
    version_match = re.search(r'v(\d+(?:\.\d+)?)', name_lower, re.IGNORECASE)
    if version_match:
        version = f"v{version_match.group(1)}"
        name_lower = name_lower.replace(version_match.group(0), " ")
    else:
        version = "v1"

    # 3. Alle Separatoren zu Spaces machen
    name_lower = re.sub(r'[_\-]+', ' ', name_lower)

    # 4. KEYWORDS liste (werden komplett entfernt)
    keywords_to_remove = [
        # Z-Image (NEU!) - WICHTIG: Längere zuerst!
        'zimageturbo', 'zimageedit', 'zimagebase',
        'zimage', 'zimg', 'z-image', 'z_image',
        # Base Models - WICHTIG: Längere zuerst!
        'fluxdev', 'fluxschnell', 'fluxpro', 'fluxfill', 'fluxultra',
        'fluxd', 'fluxs', 'fluxp', 'fluxf', 'fluxu',  # Kurzformen
        'flux',
        'dev', 'schnell', 'pro', 'fill', 'ultra',
        'sdxlturbo', 'sdxllightning', 'sdxlrefiner', 'sdxl', 'xl',
        'pony', 'illustrious', 'noobai', 'noob', 'animagine',
        'sd15', 'sd21', 'sd3',
        # Components
        'full', 'unet', 'novae', 'noclip', 'gguf',
        'checkpoint', 'checkpoints', 'chkp', 'ckpt',
        # Precision
        'fp32', 'fp16', 'bf16', 'fp8', 'mixed',
        'q8', 'q4', 'q5', 'q6',
        # Category
        'nsfw', 'porn', 'hentai', 'xxx', 'nude', 'sex',
        'realism', 'realistic', 'real', 'photo',
        'anime', 'cartoon', '2d',
        'general', 'base',
        # Andere
        'unknown',  # Entfernt "UnknownUnknownunknown..." aus Namen
        'model', 'safetensors', 'only', 'diffusion', 'turbo', 'lightning', 'refiner'
    ]

    # Präfixe/Suffixe die von Wörtern entfernt werden (z.B. "fluxdpersephone" -> "persephone", "persephone-flux" -> "persephone")
    # WICHTIG: Längere zuerst, damit "fluxd" vor "flux" geprüft wird!
    prefixes_suffixes_to_remove = [
        'zimageturbo', 'zimageedit', 'zimagebase', 'zimage', 'zimg',  # Z-Image
        'fluxdev', 'fluxschnell', 'fluxpro', 'fluxd', 'fluxs', 'fluxp', 'fluxu', 'flux',  # Flux
        'sdxlturbo', 'sdxl', 'pony', 'xl'  # SDXL
    ]

    # 5. Wörter splitten und filtern
    words = name_lower.split()
    filtered_words = []

    for word in words:
        # Skip wenn word ein Keyword ist
        if word in keywords_to_remove:
            continue

        # Entferne Präfixe (BaseModel am Anfang: "fluxdpersephone" -> "persephone")
        # Loop until no more prefixes found (für "fluxdfluxedupnsfw" -> "fluxedupnsfw" -> "edupnsfw")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes_suffixes_to_remove:
                if word.startswith(prefix) and len(word) > len(prefix):
                    word = word[len(prefix):]
                    changed = True
                    break

        # Entferne Suffixe (BaseModel am Ende: "persephone-flux" -> "persephone")
        changed = True
        while changed:
            changed = False
            for suffix in prefixes_suffixes_to_remove:
                if word.endswith(suffix) and len(word) > len(suffix):
                    word = word[:-len(suffix)]
                    changed = True
                    break

        # Entferne "unknown" rekursiv (für "UnknownUnknownmaniaiii" -> "maniaiii")
        while 'unknown' in word:
            idx = word.find('unknown')
            if idx != -1:
                word = word[:idx] + word[idx+7:]
            else:
                break
        
        # Skip wenn word zu kurz (<3 Zeichen)
        if len(word) < 3:
            continue

        # Behalten
        filtered_words.append(word)

    # 6. Name capitalizieren (Erster Buchstabe groß, Rest klein)
    if filtered_words:
        name_final = "".join([w.capitalize() for w in filtered_words])
    else:
        name_final = "Model"

    # 7. Größe anhängen wenn vorhanden
    if size_part:
        name_final = f"{name_final}_{size_part}"

    return name_final, version


def generate_proper_name(file_info):
    """Generiert korrekten Namen nach Konvention"""

    base_model = file_info["base_model"]
    variant = file_info["variant"]
    precision = file_info["precision"]
    category = file_info["category"]
    name = file_info["name"]
    version = file_info["version"]
    extension = file_info["extension"]
    target_folder = file_info["target_folder"]

    # WAN Video Models - Spezielles Naming Schema
    # Format: WAN{Version}_{Type}_{Size}_{Variant}_{Resolution}_{Version}_{Precision}.safetensors
    if target_folder == "diffusion_models" and base_model.startswith("WAN"):
        wan_version = file_info.get("wan_version", "")
        wan_type = file_info.get("wan_type", "I2V")
        wan_size = file_info.get("wan_size", "Unknown")
        wan_resolution = file_info.get("wan_resolution")  # 480P, 720P, etc. or None

        # Base: WAN oder WAN2.1/WAN2.2 etc.
        if wan_version:
            wan_base = f"WAN{wan_version}"
        else:
            wan_base = "WAN"

        # Baue Name zusammen: WAN2.1_I2V_14B_Kijai_720P_v2_FP8
        parts = [wan_base, wan_type, wan_size, name]

        # Resolution hinzufügen wenn vorhanden
        if wan_resolution:
            parts.append(wan_resolution)

        # Version hinzufügen wenn nicht v1 (Default)
        if version and version != "v1":
            parts.append(version)

        parts.append(precision)

        return "_".join(parts) + ".safetensors"

    # Z-Image Models - Spezielles Naming Schema
    # Format: ZImage_{Variant}_{Precision}.safetensors
    # Beispiele: ZImage_Turbo_BF16.safetensors, ZImage_TurboAIO_FP8.safetensors
    if base_model and (base_model.startswith("ZImage") or base_model == "ZImage"):
        # Name enthält bereits die semantische Bedeutung (Turbo, TurboAIO, etc.)
        parts = ["ZImage", name, precision]
        return "_".join(parts) + ".safetensors"

    # Qwen-Image-Edit - Spezielles Naming Schema
    # Format: QwenImageEdit_{Variant}_{Version}_{Precision}.safetensors
    # Beispiel: QwenImageEdit_2509_FP8.safetensors
    if base_model == "QwenImageEdit":
        parts = ["QwenImageEdit", name]
        if version and version != "v1":
            parts.append(version)
        parts.append(precision)
        return "_".join(parts) + ".safetensors"

    # Lotus-Depth - Spezielles Naming Schema
    # Format: Lotus_{Type}_{Variant}_{Version}_{Precision}.safetensors
    # Beispiel: Lotus_Depth-G_v1_FP16.safetensors, Lotus_Depth-G_v2.1-Disparity_FP16.safetensors
    if base_model == "LotusDepth":
        lotus_type = file_info.get("lotus_type", "Depth-G")
        lotus_variant = file_info.get("lotus_variant", "")
        parts = ["Lotus", lotus_type]
        if lotus_variant:
            parts.append(lotus_variant)
        if version and version != "v1":
            parts.append(version)
        parts.append(precision)
        return "_".join(parts) + ".safetensors"

    # GGUF Format
    if extension == ".gguf":
        return f"{base_model}_GGUF-{precision}_{category}_{name}_{version}.gguf"

    # Safetensors in unet/ (nur UNET)
    if target_folder == "unet" and variant == "UNET-only":
        return f"{base_model}_Unet-{precision}_{category}_{name}_{version}.safetensors"

    # Safetensors in checkpoints/ (Full, NoVAE, NoCLIP)
    return f"{base_model}_{variant}-{precision}_{category}_{name}_{version}.safetensors"


# ============================================================================
# PHASE 4: FOLDER LOGIC
# ============================================================================

def determine_target_folder(has_unet, has_clip, has_vae, extension, base_model=None):
    """Bestimmt Zielordner basierend auf Components"""

    # WAN Video Models → immer diffusion_models/
    if base_model and base_model.startswith("WAN"):
        return "diffusion_models"

    # Z-Image (alle Varianten) → immer diffusion_models/
    # Grund: Wird mit "Load Diffusion Model" Node geladen, NICHT CheckpointLoaderSimple
    if base_model and (base_model.startswith("ZImage") or base_model == "ZImage"):
        return "diffusion_models"

    # Qwen-Image-Edit → immer diffusion_models/
    # Grund: Wird mit "Load Diffusion Model" Node geladen
    if base_model and base_model == "QwenImageEdit":
        return "diffusion_models"

    # Lotus-Depth → immer diffusion_models/
    # Grund: Wird mit "Load Lotus Model" Node geladen (sucht in diffusion_models/)
    if base_model and base_model == "LotusDepth":
        return "diffusion_models"

    # GGUF → immer unet/
    if extension == ".gguf":
        return "unet"

    # Safetensors: Hat CLIP oder VAE?
    if has_clip or has_vae:
        return "checkpoints"
    else:
        return "unet"


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_file(file_path):
    """Komplette Analyse einer Datei"""

    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1]
    file_size = os.path.getsize(file_path)
    size_gb = file_size / (1024**3)

    # 1. Ist Base Model?
    if not is_main_model(file_path):
        return {
            "status": "SKIP",
            "reason": "Kein Base Model (< 1GB oder kein UNET)",
            "filename": filename,
            "size_gb": size_gb
        }

    # 2. GGUF?
    if extension == ".gguf":
        base_model = detect_base_model_from_filename(filename)

        if not base_model:
            return {
                "status": "SKIP",
                "reason": "Base Model nicht erkannt",
                "filename": filename,
                "size_gb": size_gb
            }

        quantization = detect_gguf_quantization(file_path)
        name, version = extract_name_version(filename)
        category = extract_category(filename)

        file_info = {
            "status": "PROCESSED",
            "type": "GGUF",
            "base_model": base_model,
            "variant": "GGUF",
            "precision": quantization,
            "category": category,
            "name": name,
            "version": version,
            "extension": ".gguf",
            "target_folder": "unet",
            "size_gb": size_gb
        }

        file_info["proper_name"] = generate_proper_name(file_info)
        return file_info

    # 3. Safetensors Analysis
    keys = read_safetensors_keys(file_path)
    has_unet, has_clip, has_vae = analyze_keys(keys)

    # Base Model erkennen
    # WICHTIG: Keys-First für Z-Image (eindeutige Patterns!)
    base_model = detect_base_model_from_keys(keys)

    # Wenn Keys nichts ergeben haben, versuche Metadata
    if not base_model:
        metadata = read_safetensors_metadata(file_path)
        base_model = detect_base_model_from_metadata(metadata)
    else:
        # Lade Metadata trotzdem (für Precision Detection)
        metadata = read_safetensors_metadata(file_path)

    # Filename als letzter Fallback
    if not base_model:
        base_model = detect_base_model_from_filename(filename)

    # Z-Image Varianten-Check: Wenn Keys "ZImage" ergeben, prüfe Filename für Variante
    if base_model == "ZImage":
        filename_variant = detect_base_model_from_filename(filename)
        if filename_variant in ["ZImageTurbo", "ZImageEdit", "ZImageBase"]:
            base_model = filename_variant

    # Pony-Check: SDXL-basiert aber eigenes Ökosystem
    # WICHTIG: Nur sd_merge_models und explizite Basis-Felder prüfen!
    # NICHT workflow/prompt (das sind nur Referenzen zu anderen Modellen)
    if base_model == "SDXL":
        # Metadata ist unter __metadata__ key (nicht direkt im Header)
        inner_meta = metadata.get('__metadata__', {})

        # 1. sd_merge_models (Modell wurde aus Pony gemerged)
        sd_merge = str(inner_meta.get('sd_merge_models', '')).lower()
        if 'pony' in sd_merge:
            base_model = "Pony"
        else:
            # 2. Explizite Basis-Felder
            for field in ['ss_base_model_version', 'modelspec.architecture', 'base_model', 'ss_sd_model_name']:
                if field in inner_meta and 'pony' in str(inner_meta.get(field, '')).lower():
                    base_model = "Pony"
                    break

        # 3. Filename als zusätzlicher Check
        if base_model == "SDXL" and 'pony' in filename.lower():
            base_model = "Pony"

    if not base_model:
        return {
            "status": "SKIP",
            "reason": "Base Model nicht erkannt",
            "filename": filename,
            "size_gb": size_gb,
            "has_unet": has_unet
        }

    # Component Variant
    variant = detect_component_variant(has_unet, has_clip, has_vae)

    # Precision
    precision = detect_precision_from_tensors(metadata)
    if not precision:
        precision = "FP16"  # Default

    # Target Folder (needs base_model for WAN detection)
    target_folder = determine_target_folder(has_unet, has_clip, has_vae, extension, base_model)

    # WAN-spezifische Verarbeitung
    if base_model == "WAN":
        wan_type = detect_wan_type(keys)
        wan_version = detect_wan_version(filename, metadata)
        wan_size = detect_wan_size(metadata)

        # Name für WAN: Variante aus Filename extrahieren (ohne WAN-Keywords)
        # Gibt jetzt auch resolution zurück
        name, version, wan_resolution = extract_wan_variant_name(filename)

        file_info = {
            "status": "PROCESSED",
            "type": "Safetensors",
            "base_model": base_model,
            "variant": variant,
            "precision": precision,
            "category": "Video",  # WAN ist immer Video
            "name": name,
            "version": version,
            "extension": ".safetensors",
            "target_folder": target_folder,
            "has_unet": has_unet,
            "has_clip": has_clip,
            "has_vae": has_vae,
            "size_gb": size_gb,
            # WAN-spezifische Felder
            "wan_type": wan_type,
            "wan_version": wan_version,
            "wan_size": wan_size,
            "wan_resolution": wan_resolution  # NEU: 480P, 540P, 720P, etc.
        }

        file_info["proper_name"] = generate_proper_name(file_info)
        return file_info

    # Qwen-Image-Edit spezifische Verarbeitung
    if base_model == "QwenImageEdit":
        # Extrahiere Version aus Filename (z.B. "2509" aus "qwen_image_edit_2509_fp8")
        import re
        version_match = re.search(r'(\d{4})', filename)  # 4-digit version like 2509
        name = version_match.group(1) if version_match else "Edit"
        version = "v1"

        file_info = {
            "status": "PROCESSED",
            "type": "Safetensors",
            "base_model": base_model,
            "variant": variant,
            "precision": precision,
            "category": "ImageEdit",
            "name": name,
            "version": version,
            "extension": ".safetensors",
            "target_folder": target_folder,
            "has_unet": has_unet,
            "has_clip": has_clip,
            "has_vae": has_vae,
            "size_gb": size_gb
        }

        file_info["proper_name"] = generate_proper_name(file_info)
        return file_info

    # Lotus-Depth spezifische Verarbeitung
    if base_model == "LotusDepth":
        import re
        filename_lower = filename.lower()

        # Lotus Type: depth-g, depth-d, normal-g, normal-d
        # WICHTIG: Explizit nach "depth-g", "depth-d" etc. suchen!
        # Nicht nur "-d" weil das auch in "depth-g-v2-1" vorkommt!
        lotus_type = "Depth-G"  # Default
        if "normal" in filename_lower:
            # Normal-D oder Normal-G
            if "normal-d" in filename_lower or "normal_d" in filename_lower:
                lotus_type = "Normal-D"
            else:
                lotus_type = "Normal-G"
        elif "depth-d" in filename_lower or "depth_d" in filename_lower:
            # Explizit Depth-D (discriminative)
            lotus_type = "Depth-D"
        # Default bleibt Depth-G (generative)

        # Lotus Variant: Disparity, etc.
        lotus_variant = ""
        if "disparity" in filename_lower:
            lotus_variant = "Disparity"

        # Version extrahieren (v1, v2, v2.1, v2-1, etc.)
        # Auch "v2-1" Format unterstützen (wird zu "v2.1")
        version_match = re.search(r'v(\d+)[-._](\d+)', filename_lower)
        if version_match:
            version = f"v{version_match.group(1)}.{version_match.group(2)}"
        else:
            # Einfache Version (v1, v2)
            version_match = re.search(r'v(\d+)', filename_lower)
            version = f"v{version_match.group(1)}" if version_match else "v1"

        file_info = {
            "status": "PROCESSED",
            "type": "Safetensors",
            "base_model": base_model,
            "variant": variant,
            "precision": precision,
            "category": "Depth",
            "name": lotus_type,
            "version": version,
            "extension": ".safetensors",
            "target_folder": target_folder,
            "has_unet": has_unet,
            "has_clip": has_clip,
            "has_vae": has_vae,
            "size_gb": size_gb,
            # Lotus-spezifische Felder
            "lotus_type": lotus_type,
            "lotus_variant": lotus_variant
        }

        file_info["proper_name"] = generate_proper_name(file_info)
        return file_info

    # Name & Version (Standard Models)
    # Z-Image: Try semantic name first
    if base_model in ["ZImageTurbo", "ZImageBase", "ZImageEdit", "ZImage"]:
        semantic_name = get_zimage_semantic_name(filename, base_model, has_unet, has_clip, has_vae)
        if semantic_name:
            name = semantic_name
            version = "v1"  # Default for semantic names
        else:
            # Finetuned/Community model - standard extraction
            name, version = extract_name_version(filename)
    else:
        # Non-ZImage models - standard extraction
        name, version = extract_name_version(filename)

    # Category
    category = extract_category(filename)

    file_info = {
        "status": "PROCESSED",
        "type": "Safetensors",
        "base_model": base_model,
        "variant": variant,
        "precision": precision,
        "category": category,
        "name": name,
        "version": version,
        "extension": ".safetensors",
        "target_folder": target_folder,
        "has_unet": has_unet,
        "has_clip": has_clip,
        "has_vae": has_vae,
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
    for root, dirs, files_in_dir in os.walk(downloads_path):
        for item in files_in_dir:
            if item.endswith(".safetensors") or item.endswith(".gguf"):
                all_files.append(os.path.join(root, item))

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
        'module_name': 'Stable Diffusion Models',
        'files': files_to_install,
        'skipped': skipped
    }

# ============================================================================
# MODUS A - INSTALLATION (Standalone mode)
# ============================================================================

def modus_a_installation(downloads_path):
    """Modus A: Installation von downloads/ nach models/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="STABLE DIFFUSION MODELS (INSTALLATION)",
        downloads_path=downloads_path,
        extensions="*.safetensors, *.gguf",
        module_type="Base Models",
        target_folders="checkpoints/, unet/"
    )

    # Scanne downloads/ (recursive)
    files = []
    for root, dirs, files_in_dir in os.walk(downloads_path):
        for item in files_in_dir:
            if item.endswith(".safetensors") or item.endswith(".gguf"):
                files.append(os.path.join(root, item))

    if not files:
        print_no_files_found("base model files")
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
            files_to_install.append({
                'path': file_path,
                'filename': filename,
                'result': result,
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
        skip_reason="Not a base model (no UNET keys, too small, or different model type)",
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
        filename = file_info['filename']
        size_mb = file_info['size_mb']
        result = file_info['result']

        print_preview_item(
            index=i,
            filename=filename,
            size_mb=size_mb,
            detected_info=f"{result['base_model']}, {result['variant']}, {result['precision']}",
            target_path=f"{result['target_folder']}/{result['proper_name']}"
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
        sys.exit(1)  # Exit code 1 = Cancelled (no pause in batch script)

    # ========================================================================
    # PHASE 3: CONFIRMATION
    # ========================================================================
    if not ask_confirm_installation():
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Installation cancelled.")
        sys.exit(1)  # Exit code 1 = Cancelled (no pause in batch script)

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
        filename = os.path.basename(file_path)

        # Execute installation - determine target directory
        if result['target_folder'] == "checkpoints":
            target_dir = CHECKPOINTS_PATH
        elif result['target_folder'] == "diffusion_models":
            target_dir = DIFFUSION_MODELS_PATH
        else:
            target_dir = UNET_PATH
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

        print_install_item(idx, len(files_to_install), filename, success, msg)

    # Final Summary (using shared helper)
    print_summary(len(files_to_install), installed, collisions, errors, keep_source)


# ============================================================================
# MODUS B - REINSTALL/CHECK
# ============================================================================

def modus_b_reinstall(checkpoints_path, unet_path, diffusion_models_path=None, scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Reinstall/Check von models/checkpoints/, models/unet/ und models/diffusion_models/

    Args:
        checkpoints_path: Path to checkpoints/
        unet_path: Path to unet/
        diffusion_models_path: Path to diffusion_models/ (WAN Video Models)
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="STABLE DIFFUSION MODELS",
        folders="checkpoints/, unet/, diffusion_models/",
        extensions="*.safetensors, *.gguf",
        module_type="Base Models + WAN Video",
        target_folders="checkpoints/, unet/, diffusion_models/",
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
                    print(f"[RESCUE] Found misplaced Base Model: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                    target_folder = result["target_folder"]
                    proper_name = result["proper_name"]
                    # Determine target directory
                    if target_folder == "checkpoints":
                        target_dir = CHECKPOINTS_PATH
                    elif target_folder == "diffusion_models":
                        target_dir = DIFFUSION_MODELS_PATH
                    else:
                        target_dir = UNET_PATH
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
            print(f"[SUCCESS] Rescued {rescued} misplaced Base Model(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    files = []

    # Recursive scan of checkpoints/
    if os.path.exists(checkpoints_path):
        for root, dirs, files_in_dir in os.walk(checkpoints_path):
            for item in files_in_dir:
                if item.endswith(".safetensors") or item.endswith(".gguf"):
                    files.append(("checkpoints", os.path.join(root, item)))

    # Recursive scan of unet/
    if os.path.exists(unet_path):
        for root, dirs, files_in_dir in os.walk(unet_path):
            for item in files_in_dir:
                if item.endswith(".safetensors") or item.endswith(".gguf"):
                    files.append(("unet", os.path.join(root, item)))

    # Recursive scan of diffusion_models/ (WAN Video Models)
    if diffusion_models_path and os.path.exists(diffusion_models_path):
        for root, dirs, files_in_dir in os.walk(diffusion_models_path):
            for item in files_in_dir:
                if item.endswith(".safetensors"):
                    files.append(("diffusion_models", os.path.join(root, item)))

    if not files:
        print_no_files_found("base model files")
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
            # Datei ist kein Base Model aber in checkpoints/unet/ → misplaced
            if scan_only:
                add_misplaced_file(Path(file_path))
            else:
                problems_list.append(('misplaced', filename, file_size_mb, result['reason'], current_folder))
            continue

        # Get info
        base_model = result["base_model"]
        component = result["variant"]
        precision = result["precision"]
        target_folder = result["target_folder"]
        proper_name = result["proper_name"]
        current_name = os.path.basename(file_path)

        # PASS 1: Skip rename/move operations
        if scan_only:
            continue

        # Check: Falscher Ordner oder Falscher Name?
        if current_folder != target_folder:
            problems_list.append(('wrong_folder', filename, file_size_mb, base_model, component, precision, current_folder, current_name, target_folder, proper_name, file_path))
        elif current_name != proper_name:
            problems_list.append(('wrong_name', filename, file_size_mb, base_model, component, precision, current_folder, current_name, proper_name, file_path))
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
                    warning=f"Not a base model file: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, model, variant, prec, curr_folder, curr_name, target_folder, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{model}, {variant}, {prec}",
                    target_path=f"{target_folder}/{proper_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    # Determine target directory
                    if target_folder == "checkpoints":
                        target_dir = CHECKPOINTS_PATH
                    elif target_folder == "diffusion_models":
                        target_dir = DIFFUSION_MODELS_PATH
                    else:
                        target_dir = UNET_PATH
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
                _, fname, size_mb, model, variant, prec, folder, curr_name, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{model}, {variant}, {prec}",
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
        print("Modul 1 - Base Models")
        print("=" * 80)
        print()
        print("Usage:")
        print("  python modul1_mainmodels.py A              # Modus A: Installation")
        print("  python modul1_mainmodels.py B [--scan-only] # Modus B: Reinstall/Check")
        print()
        print("Options:")
        print("  --scan-only  PASS 1: Nur Queue aufbauen (nur Modus B)")
        print()
        print("Modus B - 2-Pass System:")
        print("  PASS 1: python modul1_mainmodels.py B --scan-only")
        print("          Scannt alle Dateien, baut Queue auf, keine User-Fragen")
        print("  PASS 2: python modul1_mainmodels.py B")
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
        modus_b_reinstall(CHECKPOINTS_PATH, UNET_PATH, DIFFUSION_MODELS_PATH, scan_only=scan_only, batch_mode=batch, preview_mode=preview)
    else:
        print(f"Unbekannter Modus: {mode}")
        print("Nutze 'A' für Installation oder 'B' für Reinstall/Check")
        sys.exit(1)
