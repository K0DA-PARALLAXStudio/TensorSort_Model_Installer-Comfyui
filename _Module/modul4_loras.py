#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 4 - LoRAs & LyCORIS
Status: IN ENTWICKLUNG
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

# Import shared utilities
from shared_utils import (
    DOWNLOADS_DIR,
    LORAS_DIR,
    LORAS_LYCORIS_DIR,
    MODELS_DIR,
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
# SAFETENSORS READING
# ============================================================================

def read_safetensors_info(file_path):
    """Liest Keys und Metadata aus Safetensors

    Returns:
        tuple: (keys, metadata)
    """
    try:
        with open(file_path, 'rb') as f:
            # Header size (8 bytes)
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]

            # Metadata JSON
            metadata_bytes = f.read(header_size)
            metadata = json.loads(metadata_bytes.decode('utf-8'))

            # Keys (exclude __metadata__)
            keys = [k for k in metadata.keys() if k != "__metadata__"]

            return keys, metadata
    except Exception as e:
        print(f"[ERROR] Cannot read safetensors: {e}")
        return [], {}


# ============================================================================
# MODULE BOUNDARY - IS THIS A LORA?
# ============================================================================

def is_lora(file_path):
    """Prüft ob Datei ein LoRA ist (Module Boundary)

    Returns:
        tuple: (is_match, reason, details)
    """

    # 1. Extension Check
    if not str(file_path).endswith(".safetensors"):
        return False, "SKIP: Not a .safetensors file", {}

    # 2. Size Check (LoRAs sind typisch 10-500 MB, manche bis 1GB)
    file_size = file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    # Ausschluss: Zu groß für LoRA (Base Models sind >5GB)
    if file_size_mb > 2000:  # 2GB Grenze
        return False, f"SKIP: Too large for LoRA ({file_size_mb:.0f} MB)", {}

    # 3. Read Keys
    keys, metadata = read_safetensors_info(file_path)
    if not keys:
        return False, "SKIP: Cannot read keys", {}

    # 4. MUSS HABEN: LoRA Keys
    # Standard LoRA Patterns (Modern)
    lora_patterns_modern = [
        "lora_unet",
        "lora_te",
        ".lora_up.weight",
        ".lora_down.weight",
        ".alpha"
    ]

    # Alternative A/B Format (Older/Alternative)
    lora_patterns_alt = [
        ".lora_A.weight",
        ".lora_B.weight"
    ]

    # Old Prefix Patterns (WAN, older trainers)
    lora_patterns_old = [
        "diffusion_model.blocks.",
        "transformer.transformer_blocks."
    ]

    # LyCORIS Patterns (auch LoRAs!)
    lycoris_patterns = [
        "hada_w1",
        "hada_w2",
        "hada_t1",  # Optional LoHa
        "lokr_w1",
        "lokr_w2",
        "lora_mid.weight"  # LoCon
    ]

    # Combine all patterns
    all_lora_patterns = (
        lora_patterns_modern +
        lora_patterns_alt +
        lora_patterns_old +
        lycoris_patterns
    )

    has_lora_keys = any(any(p in k for p in all_lora_patterns) for k in keys)

    if not has_lora_keys:
        return False, "SKIP: No LoRA keys found", {}

    # 5. DARF NICHT HABEN: Standalone Base Model UNET
    # WICHTIG: Flux LoRAs haben "lora_unet_double_blocks" - das ist OK!
    # NUR standalone "double_blocks." (ohne lora_ prefix) ist Base Model

    has_standalone_sdxl_unet = any(k.startswith("model.diffusion_model.") for k in keys)
    has_standalone_flux_unet = any(
        (k.startswith("double_blocks.") or k.startswith("single_blocks."))
        for k in keys
    )

    if has_standalone_sdxl_unet or has_standalone_flux_unet:
        return False, "SKIP: Standalone UNET (Base Model)", {}

    # 6. DARF NICHT HABEN: Standalone VAE (ohne lora_ prefix)
    has_standalone_vae = any(
        (k.startswith("decoder.") or k.startswith("encoder."))
        and "lora" not in k
        for k in keys
    )
    if has_standalone_vae:
        return False, "SKIP: Standalone VAE", {}

    # 7. DARF NICHT HABEN: Standalone CLIP (ohne lora_ prefix)
    clip_standalone_patterns = [
        "conditioner.embedders.",
        "text_model.encoder.layers."
    ]
    has_standalone_clip = any(
        any(p in k for p in clip_standalone_patterns) and "lora" not in k
        for k in keys
    )
    if has_standalone_clip:
        return False, "SKIP: Standalone CLIP", {}

    # 8. DARF NICHT HABEN: ControlNet
    has_controlnet = any(
        "control_model" in k or "input_hint_block" in k
        for k in keys
    )
    if has_controlnet:
        return False, "SKIP: ControlNet (different module)", {}

    # ✅ It's a LoRA!
    return True, "LoRA", {"keys": keys, "metadata": metadata}


# ============================================================================
# LYCORIS DETECTION
# ============================================================================

def is_lycoris(keys, metadata):
    """Prüft ob LoRA ein LyCORIS ist (LoHa, LoKr, LoCon, etc.)

    Returns:
        tuple: (is_lycoris, algo_type)
    """

    meta = metadata.get('__metadata__', {})

    # Priority 1: Metadata Check
    network_module = meta.get('ss_network_module', '').lower()

    if 'lycoris' in network_module:
        # Extract algo type if possible
        if 'loha' in network_module:
            return True, "LoHa"
        elif 'lokr' in network_module:
            return True, "LoKr"
        elif 'locon' in network_module:
            return True, "LoCon"
        else:
            return True, "LyCORIS"

    # Priority 2: Key Pattern Check (Fallback)
    # LoHa (Hadamard Product)
    if any("hada_w1" in k or "hada_w2" in k for k in keys):
        return True, "LoHa"

    # LoKr (Kronecker Product)
    if any("lokr_w1" in k or "lokr_w2" in k for k in keys):
        return True, "LoKr"

    # LoCon (LoRA with Convolution)
    if any("lora_mid.weight" in k for k in keys):
        # Check if also has standard lora keys
        if any("lora_up" in k or "lora_down" in k for k in keys):
            return True, "LoCon"

    # Standard LoRA
    return False, "Standard"


# ============================================================================
# BASE MODEL DETECTION
# ============================================================================

def detect_base_model_from_keys(keys):
    """Erkennt Base Model aus Key Patterns (GROUND TRUTH!)

    Keys sind 100% verlässlich - sie zeigen die tatsächliche Architektur.
    Metadata kann falsch sein (z.B. Training-Script-Fehler).

    Args:
        keys: List/dict of safetensors keys

    Returns:
        str or None: Detected base model or None
    """

    # Convert dict to list if needed
    if isinstance(keys, dict):
        keys = list(keys.keys())

    # Join all keys for pattern matching
    key_str = ' '.join(keys)

    # ========================================================================
    # FLUX: single_transformer_blocks, double_blocks
    # ========================================================================
    if 'single_transformer_blocks' in key_str or 'double_blocks' in key_str:
        return 'Flux'

    # LoRA: diffusion_model.blocks.*.{self_attn|cross_attn|ffn}
    # (Flux LoRAs use different key structure than base models)
    if "diffusion_model.blocks" in key_str and ("self_attn" in key_str or "cross_attn" in key_str or "ffn" in key_str):
        return "Flux"

    # ========================================================================
    # SDXL/Pony/Illustrious: conditioner.embedders (shared architecture)
    # ========================================================================
    if 'conditioner.embedders' in key_str:
        # Cannot distinguish SDXL/Pony/Illustrious from keys alone
        # Must rely on metadata/filename for these
        return 'SDXL'  # Generic SDXL-based

    # ========================================================================
    # SD 1.5/2.x: cond_stage_model.transformer
    # ========================================================================
    if 'cond_stage_model.transformer' in key_str:
        # Check for SD 2.x indicators (more layers in cross-attention)
        # SD 2.x has different layer counts, but hard to detect reliably
        # Default to SD15 (most common)
        return 'SD15'

    # ========================================================================
    # SD 3.x: New architecture with mmdit
    # ========================================================================
    if 'joint_blocks' in key_str and 'context_embedder' in key_str:
        return 'SD30'  # SD3/SD3.5 share architecture

    # ========================================================================
    # WAN (Video Model): Specific architecture
    # ========================================================================
    if 'decoder.model' in key_str and 'mmdit' in key_str.lower():
        return 'WAN'

    # ========================================================================
    # LTX Video: another video architecture
    # ========================================================================
    if 'ltx' in key_str.lower() or ('temporal' in key_str and 'video' in key_str):
        return 'LTX'

    # ========================================================================
    # Cascade: Specific architecture
    # ========================================================================
    if 'effnet' in key_str and 'stage_c' in key_str:
        return 'Cascade'

    # ========================================================================
    # PixArt: Transformer-based
    # ========================================================================
    if 'adaln' in key_str and 'pixart' in key_str.lower():
        return 'PixArt'

    # ========================================================================
    # Hunyuan: Chinese model
    # ========================================================================
    if 'hunyuan' in key_str.lower():
        if 'video' in key_str.lower():
            return 'HunyuanVideo'
        return 'Hunyuan'

    # ========================================================================
    # Kolors: Another architecture
    # ========================================================================
    if 'kolors' in key_str.lower():
        return 'Kolors'

    # ========================================================================
    # AuraFlow
    # ========================================================================
    if 'auraflow' in key_str.lower() or 'aura_flow' in key_str.lower():
        return 'AuraFlow'

    # ========================================================================
    # Playground
    # ========================================================================
    if 'playground' in key_str.lower():
        return 'Playground'

    # ========================================================================
    # Sana
    # ========================================================================
    if 'sana' in key_str.lower() or 'linear_head' in key_str:
        return 'Sana'

    # No match found
    return None


def detect_base_model(metadata, filename, keys=None):
    """Erkennt Base Model mit intelligenter Priorität

    NEUE PRIORITÄT (robuster gegen fehlerhafte Metadata):
    1. Keys Pattern Check (100% Ground Truth!)
    2. Metadata Check (kann falsch sein)
    3. Filename Fallback

    Args:
        metadata: Safetensors metadata dict
        filename: Filename string
        keys: Optional - safetensors keys (list or dict)

    Returns:
        str: Base Model (Flux, SDXL, Pony, SD15, etc. or Unknown)
    """

    # ========================================================================
    # STUFE 1: KEYS PATTERN CHECK (Höchste Priorität - Ground Truth!)
    # ========================================================================
    # NOTE: Funktioniert NUR für Base Models (checkpoints), NICHT für LoRAs!
    # LoRAs haben "lora_unet_...", "lora_te_..." Keys, nicht die Base Model Architektur
    # Für LoRAs: SKIP zu Stufe 2 (Metadata/Filename)
    if keys is not None:
        base_from_keys = detect_base_model_from_keys(keys)
        if base_from_keys:
            # Keys haben gesprochen - das ist die Wahrheit!
            # Aber: Bei SDXL-based müssen wir noch Pony/Illustrious unterscheiden
            if base_from_keys == 'SDXL':
                # Check metadata/filename for specific SDXL variants
                meta = metadata.get('__metadata__', {})
                base_model_meta = meta.get('ss_base_model_version', '').lower()
                filename_lower = filename.lower()

                if 'pony' in base_model_meta or 'pony' in filename_lower:
                    return 'Pony'
                elif 'illustrious' in base_model_meta or 'illustrious' in filename_lower:
                    return 'Illustrious'
                elif 'noobai' in base_model_meta or 'noobai' in filename_lower:
                    return 'NoobAI'
                elif 'animagine' in base_model_meta or 'animagine' in filename_lower:
                    return 'Animagine'
                else:
                    return 'SDXL'  # Generic SDXL
            else:
                # For all other models, keys are definitive
                return base_from_keys
        # If base_from_keys is None -> Fall through to Stufe 2 (Metadata)

    # ========================================================================
    # STUFE 2: METADATA CHECK (kann falsch sein, aber oft korrekt)
    # ========================================================================
    meta = metadata.get('__metadata__', {})

    # Check ss_base_model_version
    base_model = meta.get('ss_base_model_version', '').lower()
    if base_model:
        # Primary Models (most common)
        if 'flux' in base_model:
            return 'Flux'
        # ZImage (Z-Image Turbo, Base, Edit)
        if 'zimage' in base_model or 'z-image' in base_model or 'z_image' in base_model:
            if 'turbo' in base_model:
                return 'ZImageTurbo'
            elif 'edit' in base_model:
                return 'ZImageEdit'
            elif 'base' in base_model:
                return 'ZImageBase'
            else:
                return 'ZImageTurbo'  # Default (most common)
        # SDXL (aber check Filename für Variants!)
        if 'sdxl' in base_model:
            # Check if Filename indicates a variant (Pony, Illustrious, etc.)
            filename_lower = filename.lower()
            if 'pony' in base_model or 'pony' in filename_lower:
                return 'Pony'
            elif 'illustrious' in base_model or 'illustrious' in filename_lower:
                return 'Illustrious'
            elif 'noobai' in base_model or 'noobai' in filename_lower:
                return 'NoobAI'
            elif 'animagine' in base_model or 'animagine' in filename_lower:
                return 'Animagine'
            else:
                return 'SDXL'  # Generic SDXL
        if 'pony' in base_model:
            return 'Pony'
        if 'illustrious' in base_model:
            return 'Illustrious'
        # SD 1.5 (alle Varianten!)
        if any(x in base_model for x in ['sd_v1', 'sd15', 'sd-v1-5', 'sd_1.5', 'sd-1-5', 'sd1.5', 'v1-5', 'v1_5']):
            return 'SD15'

        # SD 2.x / 3.x
        if 'sd_v2' in base_model or 'sd20' in base_model or 'sd-v2-0' in base_model or 'sd-2-0' in base_model:
            return 'SD20'
        if 'sd21' in base_model or 'sd-v2-1' in base_model or 'sd-2-1' in base_model:
            return 'SD21'
        if 'sd30' in base_model or 'sd-3-0' in base_model or 'sd3.0' in base_model:
            return 'SD30'
        if 'sd35' in base_model or 'sd-3-5' in base_model or 'sd3.5' in base_model:
            return 'SD35'

        # Anime-Specialized
        if 'noobai' in base_model:
            return 'NoobAI'
        if 'animagine' in base_model:
            return 'Animagine'

        # Other Major Models
        if 'cascade' in base_model:
            return 'Cascade'
        if 'pixart' in base_model:
            return 'PixArt'
        if 'hunyuan' in base_model:
            if 'video' in base_model:
                return 'HunyuanVideo'
            return 'Hunyuan'
        if 'kolors' in base_model:
            return 'Kolors'
        if 'auraflow' in base_model or 'aura' in base_model:
            return 'AuraFlow'
        if 'playground' in base_model:
            return 'Playground'
        if 'sana' in base_model:
            return 'Sana'

        # Video Models
        if 'wan' in base_model:
            return 'WAN'
        if 'ltx' in base_model:
            return 'LTX'

    # STUFE 2: modelspec.architecture
    architecture = meta.get('modelspec.architecture', '').lower()
    if architecture:
        if 'flux' in architecture:
            return 'Flux'
        if 'zimage' in architecture or 'z-image' in architecture:
            return 'ZImage'
        if 'sdxl' in architecture:
            return 'SDXL'
        if 'sd-v1' in architecture or 'sd1.5' in architecture:
            return 'SD15'
        if 'sd-v2' in architecture or 'sd2.0' in architecture:
            return 'SD20'
        if 'sd2.1' in architecture:
            return 'SD21'
        if 'sd3' in architecture or 'sd-3' in architecture:
            if '3.5' in architecture:
                return 'SD35'
            return 'SD30'
        if 'cascade' in architecture:
            return 'Cascade'
        if 'pixart' in architecture:
            return 'PixArt'
        if 'hunyuan' in architecture:
            if 'video' in architecture:
                return 'HunyuanVideo'
            return 'Hunyuan'
        if 'kolors' in architecture:
            return 'Kolors'
        if 'playground' in architecture:
            return 'Playground'
        if 'wan' in architecture:
            return 'WAN'
        if 'ltx' in architecture:
            return 'LTX'

    # STUFE 3: Filename Keywords (Fallback)
    filename_lower = filename.lower()

    # Primary
    if 'flux' in filename_lower:
        return 'Flux'
    # ZImage (zimage, z_image, zturbo, z-turbo patterns)
    if any(x in filename_lower for x in ['zimage', 'z_image', 'z-image', 'zturbo', 'z-turbo', 'z_turbo']):
        if 'turbo' in filename_lower or 'zturbo' in filename_lower:
            return 'ZImageTurbo'
        elif 'edit' in filename_lower:
            return 'ZImageEdit'
        elif 'base' in filename_lower:
            return 'ZImageBase'
        else:
            return 'ZImageTurbo'  # Default (most common)
    if 'sdxl' in filename_lower and 'pony' not in filename_lower and 'noobai' not in filename_lower:
        return 'SDXL'
    if 'pony' in filename_lower:
        return 'Pony'
    if 'illustrious' in filename_lower:
        return 'Illustrious'
    # SD 1.5 (alle Varianten!)
    if any(x in filename_lower for x in ['sd15', 'sd_v1', 'sd-1-5', 'sd_1.5', 'sd-v1-5', 'sd1.5', 'v1-5', 'v1_5']):
        return 'SD15'

    # SD 2.x / 3.x
    if 'sd20' in filename_lower or 'sd-2-0' in filename_lower:
        return 'SD20'
    if 'sd21' in filename_lower or 'sd-2-1' in filename_lower:
        return 'SD21'
    if 'sd30' in filename_lower or 'sd-3-0' in filename_lower:
        return 'SD30'
    if 'sd35' in filename_lower or 'sd-3-5' in filename_lower:
        return 'SD35'

    # Anime
    if 'noobai' in filename_lower:
        return 'NoobAI'
    if 'animagine' in filename_lower:
        return 'Animagine'

    # Others
    if 'cascade' in filename_lower:
        return 'Cascade'
    if 'pixart' in filename_lower:
        return 'PixArt'
    if 'hunyuan' in filename_lower:
        if 'video' in filename_lower:
            return 'HunyuanVideo'
        return 'Hunyuan'
    if 'kolors' in filename_lower:
        return 'Kolors'
    if 'auraflow' in filename_lower or 'aura-flow' in filename_lower:
        return 'AuraFlow'
    if 'playground' in filename_lower:
        return 'Playground'
    if 'sana' in filename_lower:
        return 'Sana'

    # Video
    if 'wan' in filename_lower or 'i2v' in filename_lower:
        return 'WAN'
    if 'ltx' in filename_lower:
        return 'LTX'

    # Unknown - Log Warning
    print(f"[WARNING] Unknown base model detected!")
    print(f"          File: {filename}")
    if meta.get('ss_base_model_version'):
        print(f"          Metadata 'ss_base_model_version': {meta.get('ss_base_model_version')}")
    if meta.get('modelspec.architecture'):
        print(f"          Metadata 'modelspec.architecture': {meta.get('modelspec.architecture')}")
    print(f"          >> Will be named with 'Unknown' prefix")
    print()

    return 'Unknown'


# ============================================================================
# CATEGORY DETECTION
# ============================================================================

def detect_category(keys, filename):
    """Erkennt LoRA Kategorie aus Keys + Filename

    Returns:
        str: Category (Character, Anatomy, Pose, Body, Clothing, Props, Concept, Style, Enhancement, Misc)
    """

    # Kombiniere Keys und Filename für Search
    search_text = ' '.join(keys) + ' ' + filename
    search_text = search_text.lower()

    # ========================================================================
    # MASSIV ERWEITERTE KEYWORD-LISTEN
    # Prioritätsreihenfolge: SPEZIFISCH → GENERISCH
    # WICHTIG: Pose VOR Character (wegen "cowgirl" vs "girl")!
    # ========================================================================

    keywords = {
        # 1. ANATOMY - Sehr spezifisch (Körperteile + Sex Acts)
        'Anatomy': [
            # Körperteile
            'nipple', 'breast', 'pussy', 'penis', 'ass', 'ahegao', 'vulva',
            'hands', 'feet', 'anatomy', 'cock', 'dick', 'tits', 'boobs',
            'booty', 'butt', 'genitals', 'vagina', 'anus', 'balls', 'testicles',
            'areola', 'clitoris', 'labia', 'shaft', 'glans', 'foreskin',
            'tongue', 'mouth', 'lips', 'throat', 'neck', 'shoulders',
            'thighs', 'legs', 'arms', 'fingers', 'toes', 'nails',
            # Sex Acts (gehören zu Anatomy, nicht Pose!)
            'licking', 'titfuck', 'blowjob', 'handjob', 'footjob',
            'rimming', 'cunnilingus', 'fellatio', 'oral',
            'cumshot', 'cum', 'creampie', 'facial', 'orgasm',
            'penetration', 'insertion', 'bulge', 'erection',
            'lactation', 'milk', 'squirt', 'ejaculation'
        ],

        # 2. POSE - Spezifisch (VOR Character wegen "cowgirl", "girl", etc!)
        'Pose': [
            # Grundpositionen
            'pose', 'position', 'posture', 'stance',
            # Sex Positionen (sehr spezifisch)
            'sex', 'doggystyle', 'missionary', 'cowgirl', 'reverse_cowgirl',
            'spooning', 'scissoring', 'standing', 'kneeling', 'squatting',
            'riding', 'mounted', 'straddling', 'penetrating',
            # Körperpositionen
            'spread', 'spreading', 'bent', 'bending', 'arched', 'arching',
            'kneel', 'kneeling', 'squat', 'squatting', 'sit', 'sitting',
            'stand', 'standing', 'lying', 'laying', 'lean', 'leaning',
            'crossed', 'cross-leg', 'legs_spread', 'legs_apart',
            # Dominanz/Submissiv
            'femdom', 'maledom', 'dominant', 'submissive', 'submission',
            # Video/Animation
            'i2v', 'wan2', 'animation', 'animated', 'motion', 'movement',
            # Richtungen
            'from_behind', 'from_front', 'from_side', 'from_above', 'from_below',
            'pov', 'first_person'
        ],

        # 3. BODY - Spezifisch (Körpertypen und -eigenschaften)
        'Body': [
            'pawg', 'thicc', 'thick', 'curvy', 'curves',
            'athletic', 'muscular', 'toned', 'fit', 'slim', 'skinny',
            'chubby', 'bbw', 'plus_size', 'voluptuous',
            'petite', 'small', 'tiny', 'short', 'tall',
            'tattoo', 'tattooed', 'piercing', 'pierced',
            'tanlines', 'tan', 'pale', 'dark_skin', 'light_skin',
            'body_type', 'physique', 'build', 'figure',
            'hourglass', 'pear', 'apple', 'rectangle'
        ],

        # 4. CLOTHING - Spezifisch
        'Clothing': [
            'dress', 'lingerie', 'bikini', 'outfit', 'underwear',
            'costume', 'fashion', 'clothing', 'clothes', 'garment',
            'naked', 'nude', 'clothed', 'dressed', 'undressed',
            'bra', 'panties', 'thong', 'gstring', 'stockings', 'pantyhose',
            'corset', 'bodysuit', 'leotard', 'swimsuit',
            'fur', 'coat', 'jacket', 'hoodie', 'sweater',
            'pants', 'jeans', 'shorts', 'skirt', 'dress',
            'shirt', 'blouse', 'top', 'crop_top',
            'boots', 'shoes', 'heels', 'sneakers', 'sandals',
            'gloves', 'socks', 'hat', 'cap', 'scarf'
        ],

        # 5. PROPS - Spezifisch (Objekte und Accessoires)
        'Props': [
            'toy', 'toys', 'dildo', 'vibrator', 'buttplug',
            'gag', 'ballgag', 'prop', 'props', 'object', 'accessory',
            'censor', 'censored', 'mosaic', 'blur',
            'bondage', 'rope', 'chain', 'cuffs', 'handcuffs',
            'collar', 'leash', 'restraint', 'restraints',
            'whip', 'paddle', 'flogger', 'crop',
            'furniture', 'bed', 'chair', 'table', 'couch',
            'tool', 'equipment', 'device', 'machine'
        ],

        # 6. CHARACTER - JETZT erst (nach Pose wegen "girl", "woman", etc!)
        'Character': [
            'character', 'person', 'people', 'human',
            'celebrity', 'famous', 'star', 'idol',
            'girl', 'woman', 'female', 'lady', 'she',
            'boy', 'man', 'male', 'guy', 'he',
            'teen', 'milf', 'mature', 'young', 'old',
            'face', 'portrait', 'headshot', 'closeup',
            'expression', 'emotion', 'smile', 'frown'
        ],

        # 7. CONCEPT - Mittel-spezifisch (Szenen, Hintergründe, Settings)
        'Concept': [
            # Räume/Locations
            'room', 'apartment', 'house', 'home', 'bedroom', 'bathroom',
            'interior', 'indoor', 'inside', 'indoors',
            'exterior', 'outdoor', 'outside', 'outdoors',
            'scene', 'background', 'setting', 'location', 'place',
            'environment', 'surroundings', 'scenery',
            # Architektur/Stil
            'brutalist', 'minimalist', 'industrial', 'modern', 'vintage',
            'rustic', 'urban', 'rural', 'cityscape', 'landscape',
            # Natur
            'nature', 'forest', 'beach', 'ocean', 'mountain', 'sky',
            'garden', 'park', 'street', 'road', 'alley',
            # Zeitpunkt
            'night', 'day', 'sunset', 'sunrise', 'evening', 'morning'
        ],

        # 8. STYLE - Generisch (künstlerische Stile)
        'Style': [
            'style', 'styled', 'aesthetic', 'aesthetics',
            'art', 'artistic', 'artwork', 'artsy',
            'film', 'cinematic', 'movie', 'photo', 'photography',
            'noir', 'vintage', 'retro', 'modern', 'contemporary',
            'anime', 'manga', 'cartoon', 'comic', 'illustration',
            'painting', 'drawing', 'sketch', 'render', 'cgi',
            '3d', '2d', 'realistic', 'stylized', 'abstract',
            'dramatic', 'moody', 'bright', 'dark', 'colorful',
            'monochrome', 'black_white', 'sepia'
        ],

        # 9. ENHANCEMENT - Generisch (Qualitätsverbesserungen)
        'Enhancement': [
            'realistic', 'realism', 'photorealistic',
            'detail', 'detailed', 'details', 'high_detail',
            'quality', 'high_quality', 'hq', 'uhd', '4k', '8k',
            'enhance', 'enhanced', 'enhancement', 'improved',
            'skin', 'skin_detail', 'skin_texture', 'pores',
            'texture', 'textures', 'material', 'materials',
            'improvement', 'refine', 'refined', 'sharpness', 'clarity',
            'fix', 'fixed', 'correction', 'corrected'
        ]
    }

    # Durchlaufe Kategorien in DIESER Reihenfolge (wichtig!)
    for category, kw_list in keywords.items():
        if any(kw in search_text for kw in kw_list):
            return category

    # Fallback: Misc
    return 'Misc'


# ============================================================================
# SLIDER DETECTION (Sub-Type wie LyCORIS)
# ============================================================================

def detect_slider(metadata, filename):
    """Detect if LoRA is a Slider LoRA

    Slider LoRAs haben bidirektionale Kontrolle (z.B. age: younger ← 0 → older)
    Sie werden mit Strength -10 bis +10 statt nur 0-1 verwendet.

    Detection basiert auf:
    1. Metadata (ss_network_module, Kommentare)
    2. Filename keywords
    3. Concept patterns (age, weight, size, etc.)

    Args:
        metadata: Safetensors metadata
        filename: Dateiname

    Returns:
        tuple: (is_slider, slider_concept or None)
            is_slider (bool): True wenn Slider LoRA erkannt
            slider_concept (str): Erkanntes Konzept (z.B. "age", "detail") oder "unknown"
    """

    meta = metadata.get('__metadata__', {})
    filename_lower = filename.lower()

    # Priority 1: Metadata check (ss_network_module oder Kommentare)
    network_module = meta.get('ss_network_module', '').lower()
    if 'slider' in network_module:
        return True, "metadata"

    # Check comments/description in metadata
    for key in ['comment', 'description', 'notes']:
        value = meta.get(key, '').lower()
        if 'slider' in value:
            return True, "metadata"

    # Priority 2: Filename contains "slider"
    if 'slider' in filename_lower:
        # Try to extract concept from filename
        slider_concepts = [
            'age', 'weight', 'size', 'eye', 'eyes',
            'detail', 'light', 'contrast', 'saturation',
            'brightness', 'exposure', 'color', 'hue',
            'temperature', 'style', 'quality'
        ]
        for concept in slider_concepts:
            if concept in filename_lower:
                return True, concept
        return True, "unknown"

    # Priority 3: Concept patterns
    # Slider LoRAs often have these bidirectional concepts
    slider_patterns = [
        'age', 'younger', 'older',
        'weight', 'thin', 'fat', 'heavy', 'light',
        'size', 'bigger', 'smaller', 'large', 'small',
        'height', 'tall', 'short',
        'eye_size', 'eyes_size',
        'detail_level', 'more_details', 'less_details',
        'light', 'lighting', 'brightness', 'darker', 'brighter',
        'contrast', 'high_contrast', 'low_contrast',
        'saturation', 'vivid', 'muted',
        'temperature', 'warm', 'cool'
    ]

    # Check if filename has slider-typical patterns
    if any(pattern in filename_lower for pattern in slider_patterns):
        # Sliders typically have SHORT names (not complex LoRAs)
        # Check: simple name structure
        words = filename_lower.replace('_', ' ').replace('-', ' ').split()
        # Remove common suffixes
        words = [w for w in words if not w.startswith('v') or not w[1:].replace('.', '').isdigit()]
        words = [w for w in words if w not in ['lora', 'model', 'safetensors', 'flux', 'sdxl', 'pony']]

        # If only 1-4 meaningful words → likely a slider
        if len(words) <= 4:
            for pattern in slider_patterns:
                if pattern in filename_lower:
                    return True, pattern

    return False, None


# ============================================================================
# TRIGGER WORD EXTRACTION
# ============================================================================

def extract_trigger_word(metadata):
    """Extrahiert Trigger Word aus ss_tag_frequency

    Returns:
        str or None: Trigger word or None
    """

    meta = metadata.get('__metadata__', {})
    ss_tag_frequency = meta.get('ss_tag_frequency', '')

    if not ss_tag_frequency:
        return None

    try:
        tags_json = json.loads(ss_tag_frequency)

        # Handle nested format {"dataset": {"tag": count}}
        if tags_json and isinstance(list(tags_json.values())[0], dict):
            tag_counts = {}
            for dataset, tag_dict in tags_json.items():
                for tag, count in tag_dict.items():
                    tag_counts[tag] = tag_counts.get(tag, 0) + count
        else:
            tag_counts = tags_json

        if not tag_counts:
            return None

        # Get TOP TAG
        top_tag = max(tag_counts.items(), key=lambda x: x[1])[0]

        # Validation: Not too long
        word_count = len(top_tag.split())
        if word_count > 3 or len(top_tag) > 30:
            return None  # Too long

        return top_tag

    except Exception:
        return None


# ============================================================================
# NAME CLEANING
# ============================================================================

def clean_lora_name(filename, base_model, category):
    """Bereinigt LoRA Namen

    Returns:
        str: Cleaned name (NO VERSION!)
    """

    name = Path(filename).stem  # Ohne .safetensors

    # First: Replace spaces with dashes (before lowercasing for comparison)
    name = name.replace(' ', '-')

    name_lower = name.lower()

    # 1. Entferne Base Model Prefixes
    prefixes_to_remove = [
        # ZImage (WICHTIG: Längere zuerst!)
        'zimageturbo_', 'zimageturbo-', 'zimage_turbo_', 'zimage-turbo-',
        'zimageedit_', 'zimageedit-', 'zimage_edit_', 'zimage-edit-',
        'zimagebase_', 'zimagebase-', 'zimage_base_', 'zimage-base-',
        'zimage_', 'zimage-', 'z-image_', 'z-image-', 'z_image_', 'z_image-',
        'zturbo_', 'zturbo-', 'z-turbo_', 'z-turbo-', 'z_turbo_',
        # Primary
        'flux_', 'fluxd_', 'fluxs_', 'flux-',
        'sdxl_', 'sdxl-',
        'pony_', 'ponyxl_', 'pony-',
        'illustrious_', 'illustrious-',
        'sd15_', 'sd_v1_', 'sd-v1_', 'sd-1-5_',
        # SD 2.x / 3.x
        'sd20_', 'sd-2-0_', 'sd_v2_',
        'sd21_', 'sd-2-1_',
        'sd30_', 'sd-3-0_', 'sd3_', 'sd-3_',
        'sd35_', 'sd-3-5_',
        # Anime
        'noobai_', 'noobai-', 'noob_',
        'animagine_', 'animagine-',
        # Others
        'cascade_', 'cascade-',
        'pixart_', 'pixart-',
        'hunyuan_', 'hunyuan-', 'hunyuanvideo_',
        'kolors_', 'kolors-',
        'auraflow_', 'aura-flow_', 'aura_',
        'playground_', 'playground-',
        'sana_', 'sana-',
        # Video
        'wan_', 'wan-', 'i2v_',
        'ltx_', 'ltx-'
    ]
    for prefix in prefixes_to_remove:
        if name_lower.startswith(prefix):
            name = name[len(prefix):]
            name_lower = name.lower()

    # 1b. Entferne Base Model Suffixes (am Ende)
    suffixes_to_remove = [
        # ZImage (WICHTIG: Längere zuerst!)
        '-zimageturbo', '-zimage-turbo', '-zimage_turbo',
        '-zimageedit', '-zimage-edit', '-zimage_edit',
        '-zimagebase', '-zimage-base', '-zimage_base',
        '-zimage', '-z-image', '-z_image',
        '-zturbo', '-z-turbo', '-z_turbo',
        '_zimage', '_z-image', '_zturbo',
        # Others
        '-flux', '-sdxl', '-pony', '-sd15', '-sd21', '-sd30',
        '-illustrious', '-noobai', '-animagine'
    ]
    for suffix in suffixes_to_remove:
        if name_lower.endswith(suffix):
            name = name[:-len(suffix)]
            name_lower = name.lower()

    # 2. Entferne Category Prefixes
    category_prefixes = [
        'anatomy_', 'pose_', 'character_', 'body_',
        'clothing_', 'props_', 'concept_', 'style_',
        'enhancement_', 'lora_', 'lycoris_', 'misc_',
        'slider_'  # Remove slider prefix (will be re-added if is_slider=True)
    ]
    for prefix in category_prefixes:
        if name_lower.startswith(prefix):
            name = name[len(prefix):]
            name_lower = name.lower()

    # 2b. Entferne Trigger-Prefix (von vorheriger Benennung)
    # Format: "_Trig-{trigger}" am Ende vor .safetensors
    import re
    # Entfernt "_Trig-{anything}" am Ende (greedy bis Ende)
    name = re.sub(r'_Trig-.*$', '', name)

    # 3. Entferne Precision Markers
    precision_markers = ['_fp8', '_fp16', '_fp32', '_bf16', '-fp8', '-fp16', '-fp32', '-bf16']
    for marker in precision_markers:
        name = name.replace(marker, '').replace(marker.upper(), '')

    # 4. Entferne GGUF Quants (falls vorhanden)
    gguf_markers = ['_q4', '_q8', '_q4_k_m', '_q8_k', '-q4', '-q8']
    for marker in gguf_markers:
        name = name.replace(marker, '').replace(marker.upper(), '')

    # 5. Entferne Version aus Name (LoRAs haben KEINE Version im Namen!)
    name = re.sub(r'[_-]?v\d+(?:\.\d+)?', '', name, flags=re.IGNORECASE)

    # 6. Entferne redundante Keywords (inkl. Base Models & Categories!)
    # WICHTIG: Nur STANDALONE löschen, nicht in Compound-Words!
    # z.B. "Body" löschen, ABER "Body-Hair" behalten!
    redundant = [
        # Generic
        'nsfw', 'realistic', 'dev', 'lora', 'model',
        'checkpoint', 'file', 'download', 'trained',
        'base', 'version', 'turbo', 'edit',
        # ZImage (wenn sie in der Mitte auftauchen)
        'zimage', 'zimageturbo', 'zimageedit', 'zimagebase',
        'z-image', 'z-image-turbo', 'z-image-edit', 'z-image-base',
        'zturbo', 'z-turbo', 'zimg',
        # Base Models (wenn sie in der Mitte auftauchen)
        'flux', 'fluxd', 'fluxs', 'sdxl', 'pony', 'illustrious',
        'sd15', 'sd21', 'sd30', 'sd35', 'noobai', 'animagine',
        # Categories (wenn sie in der Mitte auftauchen)
        'anatomy', 'pose', 'character', 'clothing',
        'props', 'concept', 'style', 'enhancement', 'misc'
        # NOTE: 'body' REMOVED from list - causes "Body-Hair" -> "Hair" problem!
        # Body is common in compound words (body-hair, body-type, etc.)
    ]

    # Split by both _ AND - and filter
    # ONLY remove words that are STANDALONE (whole word match)
    words = name.split('_')
    filtered = []
    for word in words:
        # Split this word by dashes too
        dash_words = word.split('-')

        # Filter: only remove if ENTIRE dash-segment matches redundant
        # e.g., "Body" alone -> remove
        # e.g., "Body-Hair" -> keep (Body is part of compound)
        cleaned_dash_words = []
        for dw in dash_words:
            # Check if this is a standalone redundant word
            # OR if it's part of a multi-word segment (compound)
            if len(dash_words) > 1:
                # Part of compound - keep it (unless ALL parts are redundant)
                cleaned_dash_words.append(dw)
            else:
                # Standalone word - check redundant list
                if dw.lower() not in redundant:
                    cleaned_dash_words.append(dw)

        if cleaned_dash_words:
            filtered.append('-'.join(cleaned_dash_words))

    name = '_'.join(filtered) if filtered else name

    # 7. Konvertiere zu Title Case mit Bindestrichen (underscore → dash)
    name = name.replace('_', '-')

    # Title case each word
    words = name.split('-')
    words = [w.capitalize() for w in words if w]  # Skip empty
    name = '-'.join(words)

    # 8. Remove duplicate words (case-insensitive, including compound duplicates)
    # Example: "Brazilian-Tanlines-Braziliantanlines" → "Brazilian-Tanlines"
    words = name.split('-')
    cleaned_words = []
    seen_words_lower = set()

    for word in words:
        word_lower = word.lower()

        # Skip if we've seen this exact word before
        if word_lower in seen_words_lower:
            continue

        # Check if this word is a combination of previous words
        # e.g., "braziliantanlines" contains "brazilian" + "tanlines"
        is_compound = False
        if len(cleaned_words) >= 2:
            # Check last 2-3 words combined
            for i in range(min(3, len(cleaned_words))):
                combined = ''.join([w.lower() for w in cleaned_words[-(i+1):]])
                if combined == word_lower:
                    is_compound = True
                    break

        if not is_compound:
            cleaned_words.append(word)
            seen_words_lower.add(word_lower)

    name = '-'.join(cleaned_words)

    # Fallback: Wenn Name leer, nutze Kategorie
    if not name or name == '-':
        name = category

    return name


# ============================================================================
# GENERATE PROPER NAME
# ============================================================================

def generate_proper_name(base_model, category, cleaned_name, trigger_word, is_slider=False, slider_concept=None):
    """Generiert standardisierten Namen

    Format:
        {BaseModel}_Slider_{Category}_{Name}_Trig-{trigger}.safetensors
        ODER
        {BaseModel}_{Category}_{Name}_Trig-{trigger}.safetensors

    Args:
        base_model: Base Model (Flux, SDXL, etc.)
        category: Category (Anatomy, Pose, etc.)
        cleaned_name: Cleaned LoRA name
        trigger_word: Optional trigger word
        is_slider: If this is a Slider LoRA
        slider_concept: Slider concept (age, detail, etc.)

    Returns:
        str: Generierter Filename
    """

    # Base Model + Slider (if detected)
    if is_slider:
        parts = [base_model, "Slider", category, cleaned_name]
    else:
        parts = [base_model, category, cleaned_name]

    # Trigger Word (optional)
    if trigger_word:
        # Replace spaces with underscores in trigger
        trigger_clean = trigger_word.replace(' ', '_')
        parts.append(f"Trig-{trigger_clean}")

    return "_".join(parts) + ".safetensors"


# ============================================================================
# ANALYZE FILE
# ============================================================================

def analyze_file(file_path):
    """Analysiert eine LoRA Datei komplett

    Returns:
        dict: Analysis result
    """

    # Module Boundary Check
    is_match, reason, details = is_lora(file_path)

    if not is_match:
        return {
            "status": "SKIP",
            "reason": reason,
            "filename": file_path.name
        }

    keys = details["keys"]
    metadata = details["metadata"]
    filename = file_path.name

    # Type Detection (Standard LoRA vs LyCORIS)
    is_lyc, lyc_type = is_lycoris(keys, metadata)
    lora_type = "LyCORIS" if is_lyc else "Standard"
    target_folder = LORAS_LYCORIS_DIR if is_lyc else LORAS_DIR

    # Base Model Detection (mit Keys für höchste Genauigkeit!)
    base_model = detect_base_model(metadata, filename, keys=keys)

    # Category Detection
    category = detect_category(keys, filename)

    # Trigger Word
    trigger_word = extract_trigger_word(metadata)

    # Slider Detection (Sub-Type wie LyCORIS)
    is_slider, slider_concept = detect_slider(metadata, filename)

    # Name Cleaning (NO VERSION for LoRAs!)
    cleaned_name = clean_lora_name(filename, base_model, category)

    # Generate Proper Name (mit Slider-Info!)
    proper_name = generate_proper_name(base_model, category, cleaned_name, trigger_word,
                                        is_slider=is_slider, slider_concept=slider_concept)

    return {
        "status": "PROCESSED",
        "type": lora_type,
        "lycoris_algo": lyc_type if is_lyc else None,
        "is_slider": is_slider,
        "slider_concept": slider_concept,
        "base_model": base_model,
        "category": category,
        "trigger_word": trigger_word,
        "cleaned_name": cleaned_name,
        "proper_name": proper_name,
        "target_folder": target_folder
    }


# ============================================================================
# EXPORTABLE FUNCTIONS (for all_modules.py batch processing)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads and return file list (for batch processing)"""
    all_files = list(downloads_path.glob("*.safetensors"))

    results = []
    for file_path in all_files:
        result = analyze_file(file_path)
        result["source_path"] = file_path
        results.append(result)

    # Filter: Only PROCESSED
    to_install = [r for r in results if r["status"] == "PROCESSED"]
    skipped = [r for r in results if r["status"] == "SKIP"]

    # Format for batch
    files_to_install = []
    for r in to_install:
        size_gb = r['source_path'].stat().st_size / (1024**3)
        files_to_install.append({
            'path': r['source_path'],
            'filename': r['source_path'].name,
            'size_gb': size_gb,
            'result': r
        })

    skipped_formatted = []
    for r in skipped:
        size_gb = r['source_path'].stat().st_size / (1024**3)
        skipped_formatted.append({
            'filename': r['source_path'].name,
            'reason': r['reason'],
            'size_gb': size_gb
        })

    return {
        'module_name': 'LoRAs & LyCORIS',
        'files': files_to_install,
        'skipped': skipped_formatted
    }

# ============================================================================
# MODUS A - INSTALLATION (Standalone mode)
# ============================================================================

def modus_a():
    """Modus A: Installation aus downloads/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="LORAS & LYCORIS (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.safetensors",
        module_type="LoRAs",
        target_folders="loras/, loras/LyCORIS/"
    )

    all_files = list(DOWNLOADS_DIR.glob("*.safetensors"))

    if not all_files:
        print_no_files_found("LoRA files")
        return

    # ========================================================================
    # PHASE 1: PREVIEW - Analyze all files (silently collect)
    # ========================================================================
    files_to_install = []
    skipped = []

    for file_path in all_files:
        result = analyze_file(file_path)
        result["source_path"] = file_path
        size_mb = file_path.stat().st_size / (1024**2)
        size_gb = size_mb / 1024

        if result["status"] == "SKIP":
            skipped.append({'filename': result['filename'], 'size_mb': size_mb})
        else:
            # Build detected string
            type_str = result['type']
            if result['lycoris_algo']:
                type_str += f" ({result['lycoris_algo']})"
            if result['is_slider']:
                type_str += f" [Slider]"

            files_to_install.append({
                'path': file_path,
                'filename': file_path.name,
                'result': result,
                'detected_str': f"{type_str}, {result['base_model']}, {result['category']}",
                'trigger_word': result['trigger_word'],
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
        skip_reason="Not a LoRA (no LoRA keys, too large, or different model type)",
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
        extra_line = f"Trigger: {file_info['trigger_word']}" if file_info['trigger_word'] else None
        print_preview_item(
            index=i,
            filename=file_info['filename'],
            size_mb=file_info['size_mb'],
            detected_info=file_info['detected_str'],
            target_path=f"{r['target_folder'].name}/{r['proper_name']}",
            extra_line=extra_line
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
        r = file_info['result']
        source_path = file_info['path']
        target_path = r['target_folder'] / r['proper_name']

        r['target_folder'].mkdir(parents=True, exist_ok=True)

        success, final_path, msg = handle_duplicate_move(
            source_path,
            target_path,
            expected_target_name=r['proper_name'],
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

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check bestehender LoRAs

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="LORAS & LYCORIS",
        folders="loras/, loras/LyCORIS/",
        extensions="*.safetensors",
        module_type="LoRAs",
        target_folders="loras/, loras/LyCORIS/",
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
                result = analyze_file(file_path)

                if result["status"] == "PROCESSED":
                    filename = file_path.name
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)

                    print(f"[RESCUE] Found misplaced LoRA: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                    target_path = result['target_folder'] / result['proper_name']
                    result['target_folder'].mkdir(parents=True, exist_ok=True)

                    success, final_path, msg = handle_duplicate_move(
                        file_path,
                        target_path,
                        expected_target_name=result['proper_name'],
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    if success:
                        print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: {result['target_folder'].name}/{final_path.name}")
                        remove_misplaced_file(file_path)
                        rescued += 1
                    else:
                        print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                    print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced LoRA(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    standard_files = list(LORAS_DIR.glob("*.safetensors")) if LORAS_DIR.exists() else []
    lycoris_files = list(LORAS_LYCORIS_DIR.glob("*.safetensors")) if LORAS_LYCORIS_DIR.exists() else []
    all_files = standard_files + lycoris_files

    if not all_files:
        print_no_files_found("LoRA files")
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
                problems_list.append(('misplaced', filename, file_size_mb, result['reason'], file_path.parent.name))
            continue

        if scan_only:
            continue

        current_folder = file_path.parent
        expected_folder = result['target_folder']
        current_name = file_path.name
        expected_name = result['proper_name']

        if current_folder != expected_folder:
            problems_list.append(('wrong_folder', filename, file_size_mb, result['type'], result['category'], current_folder, expected_folder, expected_name, file_path))
        elif current_name != expected_name:
            problems_list.append(('wrong_name', filename, file_size_mb, result['type'], result['category'], current_folder, expected_name, file_path))
        else:
            correct_files.append((file_path.parent.name, filename, file_size_mb))

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
                    warning=f"Not a LoRA file: {reason}"
                )

            elif problem_type == 'wrong_folder':
                _, fname, size_mb, lora_type, category, curr_folder, expected_folder, expected_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{lora_type}, {category}",
                    target_path=f"{expected_folder.name}/{expected_name}",
                    warning="Wrong folder detected"
                )

                if not preview_mode:
                    expected_folder.mkdir(parents=True, exist_ok=True)
                    target_path = expected_folder / expected_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=expected_name,
                        mode="B",
                        keep_source_option=False,
                        dry_run=False
                    )

                    print_fix_result(success, "Moved and renamed to correct location" if success else msg)
                    if success:
                        renamed += 1

            elif problem_type == 'wrong_name':
                _, fname, size_mb, lora_type, category, curr_folder, expected_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{lora_type}, {category}",
                    target_path=f"{curr_folder.name}/{expected_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    target_path = fpath.parent / expected_name

                    success, final_path, msg = handle_duplicate_move(
                        fpath,
                        target_path,
                        expected_target_name=expected_name,
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
    import argparse

    parser = argparse.ArgumentParser(description="Modul 4 - LoRAs & LyCORIS")
    parser.add_argument("mode", choices=["A", "B"], help="A = Installation, B = Reinstall/Check")
    parser.add_argument("--scan-only", action="store_true", help="PASS 1: Scan-Only mode (Modus B)")
    parser.add_argument("--preview", action="store_true", help="Preview mode: show problems without fixing")
    parser.add_argument("--batch", action="store_true", help="Batch mode (skip user prompts)")

    args = parser.parse_args()

    if args.mode == "A":
        modus_a()
    else:  # Modus B
        modus_b(scan_only=args.scan_only, batch_mode=args.batch, preview_mode=args.preview)
