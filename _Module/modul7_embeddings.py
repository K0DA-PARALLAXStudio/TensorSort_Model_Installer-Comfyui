#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 7 - Embeddings (Textual Inversion)
Status: IM TEST
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
    EMBEDDINGS_DIR,
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
# CONFIGURATION
# ============================================================================

# Paths (from shared_utils.py)
DOWNLOADS_PATH = DOWNLOADS_DIR
EMBEDDINGS_PATH = EMBEDDINGS_DIR

# ============================================================================
# PHASE 1: BASIS-FUNKTIONEN
# ============================================================================

def read_model_metadata(file_path):
    """Liest Metadata aus .safetensors oder .pt Datei

    .safetensors: JSON-Parsing (sicher, kein Code-Execution)
    .pt: torch.load(..., weights_only=True) (sicher, NUR Tensoren)

    WICHTIG: Kein torch.load() OHNE weights_only=True!
    """
    file_path_str = str(file_path)

    # SAFETENSORS: JSON-Parsing (100% sicher)
    if file_path_str.endswith(".safetensors"):
        try:
            with open(file_path, 'rb') as f:
                # Header Size (8 bytes)
                header_size_bytes = f.read(8)
                header_size = struct.unpack('<Q', header_size_bytes)[0]

                # Metadata (JSON)
                metadata_bytes = f.read(header_size)
                metadata = json.loads(metadata_bytes.decode('utf-8'))

                return metadata
        except Exception as e:
            print(f"[ERROR] Cannot read safetensors metadata from {file_path}: {e}")
            return {}

    # PT: torch.load mit weights_only=True (sicher)
    elif file_path_str.endswith(".pt"):
        try:
            import torch

            # weights_only=True: Lädt NUR Tensoren, KEIN Code!
            # Verfügbar ab PyTorch 1.13+
            data = torch.load(file_path, map_location='cpu', weights_only=True)

            # PT Format hat direkt Tensoren, keine Safetensors-Struktur
            # Konvertiere zu kompatiblem Format
            metadata = {}

            if isinstance(data, dict):
                # STANDARD PT FORMAT:
                # Option 1: {"string_to_param": {"*": tensor}, ...}
                # Option 2: {"emb_params": tensor, ...}
                
                # Option 1: string_to_param (Nested Dict)
                if 'string_to_param' in data and isinstance(data['string_to_param'], dict):
                    # Extrahiere das erste Embedding (meist Key="*")
                    for token, tensor in data['string_to_param'].items():
                        if hasattr(tensor, 'shape'):
                            metadata['emb_params'] = {
                                "dtype": str(tensor.dtype).replace('torch.', ''),
                                "shape": list(tensor.shape),
                                "data_offsets": [0, 0]
                            }
                            break  # Nur erstes Embedding
                
                # Option 2: Direkte Tensoren
                for key, value in data.items():
                    if hasattr(value, 'shape'):  # Ist ein Tensor
                        metadata[key] = {
                            "dtype": str(value.dtype).replace('torch.', ''),
                            "shape": list(value.shape),
                            "data_offsets": [0, 0]
                        }
            elif hasattr(data, 'shape'):  # Direkter Tensor
                metadata["emb_params"] = {
                    "dtype": str(data.dtype).replace('torch.', ''),
                    "shape": list(data.shape),
                    "data_offsets": [0, 0]
                }

            return metadata

        except Exception as e:
            print(f"[ERROR] Cannot read PT metadata from {file_path}: {e}")
            return {}

    else:
        print(f"[ERROR] Unsupported format: {file_path}")
        return {}


def is_embedding(file_path):
    """Prüft ob Datei ein Embedding ist (Module Boundary)

    2-Stufen Detection:
    1. Schnelle Ausschluss-Checks (Size, Extension)
    2. Tensor-Shape Analyse (PFLICHT - Ground Truth!)

    Returns:
        tuple: (is_embedding, reason, details)
    """

    # Convert Path to string if needed
    file_path_str = str(file_path)

    # ========================================================================
    # STUFE 1: SCHNELLE AUSSCHLUSS-CHECKS
    # ========================================================================

    # Extension Check - Safetensors UND PT
    if not (file_path_str.endswith(".safetensors") or file_path_str.endswith(".pt")):
        return False, "Only .safetensors and .pt supported", None

    # Size Pre-Filter (NUR als AUSSCHLUSS, NICHT als Bestätigung!)
    # WICHTIG: PT-Files sind VIEL kleiner als Safetensors (komprimiert)!
    try:
        size_mb = Path(file_path_str).stat().st_size / (1024 * 1024)
    except:
        return False, "Cannot read file size", None

    # Format-abhängige Limits
    if file_path_str.endswith(".pt"):
        # PT: 2 KB - 50 KB (komprimiert)
        if size_mb < 0.002:  # < 2 KB
            return False, "File too small (probably corrupt)", None
        if size_mb > 0.05:  # > 50 KB
            return False, "File too large (probably not an embedding)", None
    else:
        # Safetensors: 5 KB - 2 MB (unkomprimiert)
        if size_mb < 0.005:  # < 5 KB
            return False, "File too small (probably corrupt)", None
        if size_mb > 2.0:  # > 2 MB
            return False, "File too large (probably not an embedding)", None

    # ========================================================================
    # STUFE 2: TENSOR-SHAPE ANALYSE (PFLICHT - Ground Truth!)
    # ========================================================================

    metadata = read_model_metadata(file_path_str)

    if not metadata:
        return False, "Cannot read metadata", None

    # Keys vorhanden?
    # MODERN FORMAT: clip_l / clip_g
    # OLD FORMAT: emb_params (legacy SD1.5 embeddings)
    has_modern_keys = "clip_l" in metadata or "clip_g" in metadata
    has_old_key = "emb_params" in metadata

    if not has_modern_keys and not has_old_key:
        return False, "No clip_l/clip_g/emb_params keys found", None

    # Prüfe Tensor-Shape (KRITISCH!)
    if has_modern_keys:
        key_to_check = "clip_l" if "clip_l" in metadata else "clip_g"
    else:
        key_to_check = "emb_params"

    if "shape" not in metadata[key_to_check]:
        return False, "No shape info in metadata", None

    shape = metadata[key_to_check]["shape"]

    # EMBEDDING: [num_vectors, embedding_dim] (2D)
    # Beispiel: [8, 768] = 8 Vektoren, 768-dimensional
    if len(shape) == 2:
        num_vectors = shape[0]
        embedding_dim = shape[1]

        # Sanity Check: Ist embedding_dim plausibel?
        if embedding_dim in [768, 1024, 1280, 2048]:
            return True, "Embedding detected", {
                'num_vectors': num_vectors,
                'embedding_dim': embedding_dim,
                'keys': list(metadata.keys())
            }
        else:
            return False, f"Unusual embedding_dim: {embedding_dim}", None

    # CLIP MODEL: [embedding_dim] (1D tensor)
    # Beispiel: [768] = CLIP Encoder Gewichte
    elif len(shape) == 1:
        return False, "CLIP Model detected (1D tensor, not embedding)", None

    # ANDERES: Mehr als 2 Dimensionen
    else:
        return False, f"Unknown tensor shape: {len(shape)}D", None


# ============================================================================
# PHASE 2: DETECTION-FUNKTIONEN
# ============================================================================

def detect_base_model_from_tensor_shape(metadata):
    """Erkennt Base Model aus Tensor-Shape (Ground Truth!)

    SDXL Embeddings haben:
    - clip_l: [N, 768]  (CLIP-L Encoder)
    - clip_g: [N, 1280] (CLIP-G Encoder)

    SD1.5 Embeddings haben:
    - clip_l: [N, 768]  (nur CLIP-L, kein clip_g)

    Flux Embeddings haben:
    - clip_g: [N, 1280] (nur CLIP-G, T5 separat)

    Args:
        metadata: Safetensors metadata dict

    Returns:
        str: "SDXL", "SD15", "Flux", oder None
    """

    has_clip_l = "clip_l" in metadata
    has_clip_g = "clip_g" in metadata
    has_clip_l_out = "clip_l_out" in metadata
    has_clip_g_out = "clip_g_out" in metadata
    has_emb_params = "emb_params" in metadata

    # OLD FORMAT: emb_params (SD1.5 legacy embeddings from PT conversion)
    if has_emb_params and not has_clip_l and not has_clip_g:
        emb_shape = metadata["emb_params"].get("shape", [])
        if len(emb_shape) == 2 and emb_shape[1] == 768:
            return "SD15"

    # Check shapes to validate
    if has_clip_l and has_clip_g:
        # Validate dimensions
        clip_l_shape = metadata["clip_l"].get("shape", [])
        clip_g_shape = metadata["clip_g"].get("shape", [])

        if len(clip_l_shape) == 2 and len(clip_g_shape) == 2:
            clip_l_dim = clip_l_shape[1]  # Should be 768
            clip_g_dim = clip_g_shape[1]  # Should be 1280

            if clip_l_dim == 768 and clip_g_dim == 1280:
                # Illustrious: Hat zusätzlich *_out Tensors
                if has_clip_l_out and has_clip_g_out:
                    return "Illustrious"
                # SDXL: Nur clip_l + clip_g (ohne *_out)
                else:
                    return "SDXL"

    elif has_clip_l and not has_clip_g:
        # SD1.5: Nur CLIP-L
        clip_l_shape = metadata["clip_l"].get("shape", [])

        if len(clip_l_shape) == 2:
            clip_l_dim = clip_l_shape[1]

            if clip_l_dim == 768:
                return "SD15"

    elif has_clip_g and not has_clip_l:
        # Flux: Nur CLIP-G (T5 wird separat gespeichert)
        clip_g_shape = metadata["clip_g"].get("shape", [])

        if len(clip_g_shape) == 2:
            clip_g_dim = clip_g_shape[1]

            if clip_g_dim == 1280:
                return "Flux"

    return None


def detect_base_model(filename):
    """Erkennt Base Model AUS FILENAME

    WICHTIG: Illustrious MUSS vor SDXL geprüft werden!
    (sonst könnte "sdxl" in "illustrious" matchen)

    Returns:
        str: Base Model Name oder "Unknown"
    """

    filename_lower = filename.lower()

    # Priority Order (wichtig!)
    if filename_lower.startswith("illustrious_"):
        return "Illustrious"
    elif filename_lower.startswith("pony_"):
        return "Pony"
    elif filename_lower.startswith("flux"):
        return "Flux"
    elif filename_lower.startswith("sdxl_"):
        return "SDXL"
    elif filename_lower.startswith("sd15_") or filename_lower.startswith("sd1.5_"):
        return "SD15"

    # Fallback: Unknown
    return "Unknown"


def detect_type(filename):
    """Erkennt ob Positive oder Negative Embedding

    Returns:
        str: "Pos", "Neg", oder "Unknown"
    """

    filename_lower = filename.lower()

    # Standard Pattern
    if "_pos_" in filename_lower or "_positive_" in filename_lower:
        return "Pos"
    elif "_neg_" in filename_lower or "_negative_" in filename_lower:
        return "Neg"

    # Community-Convention Fallback
    if filename_lower.endswith("-neg.safetensors") or filename_lower.endswith("-neg.pt"):
        return "Neg"
    elif filename_lower.endswith("-pos.safetensors") or filename_lower.endswith("-pos.pt"):
        return "Pos"

    return "Unknown"


def detect_version(filename):
    """Erkennt Version aus Filename

    Returns:
        str: Version (z.B. "v1", "v2", "v1.5") oder "v1" (Default)
    """

    # Standard Pattern: _v1, _v2, _v1.5
    match = re.search(r'_v(\d+(?:\.\d+)?)', filename.lower())
    if match:
        return f"v{match.group(1)}"

    # Default (wenn nicht erkennbar)
    return "v1"


def detect_num_vectors(metadata):
    """Liest Anzahl Vektoren aus Tensor-Shape

    Returns:
        int: Anzahl Vektoren oder None
    """

    if "clip_l" in metadata:
        if "shape" in metadata["clip_l"]:
            shape = metadata["clip_l"]["shape"]

            # Shape: [num_vectors, embedding_dim]
            # Beispiel: [8, 768]
            if len(shape) == 2:
                return shape[0]

    return None


def extract_embedding_name(filename):
    """Extrahiert sauberen Namen aus Embedding-Filename

    Cleaning-Rules:
    1. Entferne Extension (.safetensors)
    2. Entferne Base Model Prefixes
    3. Entferne Type Markers
    4. Entferne redundante Keywords
    5. Entferne Version Pattern
    6. Konvertiere zu Title Case mit Bindestrichen

    Args:
        filename: Original-Filename (z.B. "negative_hand-neg.safetensors")

    Returns:
        str: Gereinigter Name (z.B. "NegativeHand")
    """

    # 1. Extension entfernen
    name = Path(filename).stem
    name_lower = name.lower()

    # 2. GENERISCHE PREFIX-ENTFERNUNG: Alles vor "_Embedding_" entfernen
    # Funktioniert mit JEDEM Prefix (CyberRealistic, SD15, SDXL, etc.)
    if "_embedding_" in name_lower:
        pos = name_lower.find("_embedding_")
        name = name[pos + len("_embedding_"):]
        name_lower = name.lower()

        # ZUSÄTZLICH: Entferne Type Prefix direkt nach "_Embedding_" (Neg_, Pos_)
        for type_prefix in ['neg_', 'pos_', 'negative_', 'positive_']:
            if name_lower.startswith(type_prefix):
                name = name[len(type_prefix):]
                name_lower = name.lower()
                break

    # 3. Type Markers entfernen (restliche im Text)
    # Pattern 1: _pos_, _neg_, _positive_, _negative_
    name = re.sub(r'_(pos|positive|neg|negative)_', '_', name, flags=re.IGNORECASE)

    # Pattern 2: -pos, -neg ÜBERALL (am Ende UND im Text)
    # Beispiel: "Simpleneg-Neg" → "Simpleneg", "P0rnxl-Neg" → "P0rnxl"
    name = re.sub(r'-(pos|neg)\b', '', name, flags=re.IGNORECASE)

    # 4. Redundante Keywords entfernen
    keywords_to_remove = [
        'embedding', 'textual_inversion', 'ti', 'trained',
        'model', 'file', 'download'
    ]
    words = name.split('_')
    words = [w for w in words if w.lower() not in keywords_to_remove]
    name = '_'.join(words)

    # Pattern 3: Im Text eingebettet (z.B. "90sphoto-neg-embedding")
    for keyword in keywords_to_remove:
        name = re.sub(f'-{keyword}$', '', name, flags=re.IGNORECASE)
        name = re.sub(f'{keyword}-', '', name, flags=re.IGNORECASE)

    # 5. Version Pattern entfernen (wird separat erkannt)
    name = re.sub(r'[_-]?v\d+(?:\.\d+)?', '', name, flags=re.IGNORECASE)

    # 6. Cleanup: Mehrfache Underscores/Bindestriche
    name = re.sub(r'_+', '_', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('_-')

    # 7. Konvertiere Underscores zu Bindestrichen
    name = name.replace('_', '-')

    # 8. DEDUPLIZIERUNG: Entferne doppelte Wörter
    # Beispiel: "Cyberrealistic-Cyberrealistic-Simpleneg" → "Cyberrealistic-Simpleneg"
    words = name.split('-')
    unique_words = []
    seen = set()
    for word in words:
        word_lower = word.lower()
        if word_lower and word_lower not in seen:
            unique_words.append(word)
            seen.add(word_lower)
    name = '-'.join(unique_words)

    # 9. FINAL CLEANUP: Entferne -Pos/-Neg am Ende (nach Deduplizierung)
    # Beispiel: "Cyberrealistic-Simpleneg-Neg" → "Cyberrealistic-Simpleneg"
    name = re.sub(r'-(pos|neg)$', '', name, flags=re.IGNORECASE)

    # 10. Title Case: Jedes Wort großschreiben
    name = '-'.join(word.capitalize() for word in name.split('-'))

    # Wenn nichts übrig bleibt, Return "Unknown"
    if not name or len(name) < 2:
        return "Unknown"

    return name


def generate_proper_name(file_info):
    """Generiert korrekten Namen nach Konvention

    Template: {BaseModel}_Embedding_{Type}_{Name}_{Version}.{ext}
    Extension: Behält Original-Extension bei (.safetensors oder .pt)

    Args:
        file_info: dict mit base_model, type, name, version, extension

    Returns:
        str: Generierter Filename
    """

    base_model = file_info["base_model"]
    emb_type = file_info["type"]
    name = file_info["name"]
    version = file_info["version"]
    extension = file_info.get("extension", ".safetensors")  # Default: safetensors

    return f"{base_model}_Embedding_{emb_type}_{name}_{version}{extension}"


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_file(file_path):
    """Komplette Analyse einer Embedding-Datei

    Returns:
        dict: Analyse-Ergebnis mit status, proper_name, etc.
    """

    # Convert Path to string if needed
    file_path_str = str(file_path)

    filename = os.path.basename(file_path_str)
    file_size = os.path.getsize(file_path_str)
    size_mb = file_size / (1024**2)

    # 1. Ist Embedding?
    is_emb, reason, details = is_embedding(file_path_str)

    if not is_emb:
        return {
            "status": "SKIP",
            "reason": reason,
            "filename": filename,
            "size_mb": size_mb
        }

    # 2. Metadata lesen
    metadata = read_model_metadata(file_path_str)

    # 3. Base Model Detection
    # PRIORITY 1: Tensor-Shape (Ground Truth!)
    base_model = detect_base_model_from_tensor_shape(metadata)

    # PRIORITY 2: Filename (Fallback wenn Tensor-Shape fehlschlägt)
    if not base_model:
        base_model = detect_base_model(filename)

    if base_model == "Unknown" or not base_model:
        return {
            "status": "SKIP",
            "reason": "Base Model not recognized (tensor-shape + filename failed)",
            "filename": filename,
            "size_mb": size_mb
        }

    # 4. Type Detection
    emb_type = detect_type(filename)

    # DEFAULT: Wenn Type nicht erkannt, assume "Pos" (most embeddings are positive)
    if emb_type == "Unknown":
        emb_type = "Pos"

    # 5. Version Detection
    version = detect_version(filename)

    # 6. Name Extraction
    name = extract_embedding_name(filename)

    # 7. Num Vectors Detection (optional, nur für Info)
    num_vectors = detect_num_vectors(metadata)

    # 8. Generate proper name (keep original extension)
    file_extension = Path(filename).suffix  # .safetensors or .pt
    file_info = {
        "base_model": base_model,
        "type": emb_type,
        "name": name,
        "version": version,
        "extension": file_extension
    }

    proper_name = generate_proper_name(file_info)

    # Return result
    return {
        "status": "PROCESSED",
        "current_path": file_path_str,
        "current_name": filename,
        "proper_name": proper_name,
        "base_model": base_model,
        "type": emb_type,
        "name": name,
        "version": version,
        "num_vectors": num_vectors,
        "size_mb": size_mb,
        "target_folder": "embeddings"
    }


# ============================================================================
# EXPORTABLE FUNCTIONS (for all_modules.py batch processing)
# ============================================================================

def scan_for_batch(downloads_path):
    """Scan downloads and return file list (for batch processing)"""
    files_to_install = []
    skipped = []

    all_files = sorted(list(downloads_path.glob("**/*.safetensors")) + list(downloads_path.glob("**/*.pt")))  # recursive

    for file_path in all_files:
        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024**2)
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
        'module_name': 'Embeddings (Textual Inversion)',
        'files': files_to_install,
        'skipped': skipped
    }


# ============================================================================
# MODUS A: INSTALLATION (Standalone mode)
# ============================================================================

def modus_a():
    """Modus A: Installiert Embeddings von downloads/ nach embeddings/"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_a_header(
        module_name_caps="EMBEDDINGS (INSTALLATION)",
        downloads_path=DOWNLOADS_PATH,
        extensions="*.safetensors, *.pt",
        module_type="Embeddings",
        target_folders="embeddings/"
    )

    if not DOWNLOADS_PATH.exists():
        print(f"\n[ERROR] Downloads-Ordner nicht gefunden: {DOWNLOADS_PATH}")
        return

    all_files = sorted(list(DOWNLOADS_PATH.glob("**/*.safetensors")) + list(DOWNLOADS_PATH.glob("**/*.pt")))  # recursive

    if not all_files:
        print_no_files_found("embedding files")
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

        if result["status"] == "SKIP":
            skipped.append({'filename': filename, 'size_mb': size_mb})
        else:
            files_to_install.append({
                'path': file_path,
                'filename': filename,
                'result': result,
                'detected_str': f"{result['base_model']}, {result['type']}, {result['version']}",
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
        skip_reason="Not an embedding (no embedding keys found)",
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
            target_path=f"embeddings/{r['proper_name']}"
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
        result = file_info['result']
        target_path = EMBEDDINGS_PATH / result['proper_name']

        success, final_path, msg = handle_duplicate_move(
            result['current_path'],
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
# MODUS B: REINSTALL/CHECK
# ============================================================================

def modus_b(scan_only=False, batch_mode=False, preview_mode=False):
    """Modus B: Check/Korrektur in embeddings/

    Args:
        scan_only: PASS 1 - Build queue only
        batch_mode: Skip user prompts (execute fixes)
        preview_mode: Show problems only, no execution, no prompts
    """

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="EMBEDDINGS",
        folders="embeddings/",
        extensions="*.safetensors, *.pt",
        module_type="Embeddings",
        target_folders="embeddings/",
        preview_mode=preview_mode
    )

    if not EMBEDDINGS_PATH.exists():
        print(f"[ERROR] Embeddings folder not found: {EMBEDDINGS_PATH}")
        return

    # ========================================================================
    # RESCUE FROM QUEUE (only in execute mode, not preview/scan)
    # ========================================================================
    rescued = 0
    if not scan_only and not preview_mode:
        misplaced = read_misplaced_files()

        if misplaced:
            for file_path in misplaced:
                is_emb, reason, details = is_embedding(file_path)

                if is_emb:
                    filename = file_path.name
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"[RESCUE] Found misplaced Embedding: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

                    result = analyze_file(file_path)

                    if result["status"] != "SKIP":
                        target_path = EMBEDDINGS_PATH / result['proper_name']

                        success, final_path, msg = handle_duplicate_move(
                            file_path,
                            target_path,
                            expected_target_name=result['proper_name'],
                            mode="B",
                            keep_source_option=False,
                            dry_run=False
                        )

                        if success:
                            print(f"         {Colors.GREEN}OK{Colors.RESET} Rescued to: embeddings/{final_path.name}")
                            remove_misplaced_file(file_path)
                            rescued += 1
                        else:
                            print(f"         {Colors.RED}ERROR{Colors.RESET} {msg}")
                    print()

        if rescued > 0:
            print(f"[SUCCESS] Rescued {rescued} misplaced Embedding(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDER
    # ========================================================================
    all_files = sorted(list(EMBEDDINGS_PATH.glob("**/*.safetensors")) + list(EMBEDDINGS_PATH.glob("**/*.pt")))  # recursive

    if not all_files:
        print_no_files_found("embedding files")
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

        if scan_only:
            continue

        current_name = result["current_name"]
        proper_name = result["proper_name"]

        if current_name != proper_name:
            problems_list.append(('wrong_name', filename, file_size_mb, result['base_model'], result['type'], result['version'], proper_name, file_path))
        else:
            correct_files.append(("embeddings", filename, file_size_mb))

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
                    warning=f"Not an Embedding: {reason}"
                )

            elif problem_type == 'wrong_name':
                _, fname, size_mb, model, typ, ver, proper_name, fpath = problem
                print_problem_item(
                    index=idx,
                    total=len(problems_list),
                    filename=fname,
                    size_mb=size_mb,
                    detected_info=f"{model}, {typ}, {ver}",
                    target_path=f"embeddings/{proper_name}",
                    warning="Non-standard name detected"
                )

                if not preview_mode:
                    new_path = EMBEDDINGS_PATH / proper_name

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
        print("Usage: python modul7_embeddings.py <A|B> [--scan-only] [--preview] [--batch]")
        print("  A           = Modus A (Installation)")
        print("  B           = Modus B (Reinstall/Check)")
        print("  --scan-only = PASS 1: Nur Queue aufbauen (nur Modus B)")
        print("  --preview   = PASS 2: Zeige Probleme, keine Ausführung")
        print("  --batch     = Skip user prompts (batch mode)")
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
