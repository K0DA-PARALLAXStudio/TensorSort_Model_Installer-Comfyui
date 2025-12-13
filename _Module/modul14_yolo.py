#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 14 - YOLO (Ultralytics Object Detection & Segmentation)

Organisiert und benennt YOLO Models (.pt) nach standardisiertem Schema.

Unterstützt:
- Modus A: Installation aus downloads/
- Modus B: Check/Fix bestehender YOLO Models in ultralytics/bbox/ und ultralytics/segm/

WICHTIG: 2 SEPARATE ORDNER für unterschiedliche Output-Typen!
- models/ultralytics/bbox/  → Detection (bbox), Pose, Classify, OBB
- models/ultralytics/segm/  → Segmentation (Instance Masks)

Namenskonvention:
    YOLO_{Version}{Size}_{OutputType}_{Specialization}[_vX].pt

Beispiele:
    YOLO_8m_bbox_Face.pt
    YOLO_11s_segm_AnimeNSFW_v5.pt
    YOLO_8x_segm_Penis.pt
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
    ULTRALYTICS_BBOX_DIR,       # Detection/Pose/Classify/OBB
    ULTRALYTICS_SEGM_DIR,       # Segmentation
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
# MODULE BOUNDARY CHECK
# ============================================================================

def is_yolo_ultralytics(file_path):
    """Prüft ob Datei ein Ultralytics YOLO Model ist (Module Boundary)

    Detection basiert auf:
    - Extension (.pt)
    - PyTorch Checkpoint Structure
    - Tensor Patterns (model.*, *.cv2.*, *.m.*)
    - Metadata (train_args)
    - Size Range (5-200 MB)
    - ZIP-Fallback für alte Ultralytics-Versionen

    Returns:
        tuple: (is_match, reason)
            is_match: bool - True wenn YOLO
            reason: str - Grund (für Logging)
    """
    # Extension Check
    if file_path.suffix.lower() != '.pt':
        return False, "Not .pt file"

    # Size Check
    size_mb = file_path.stat().st_size / (1024 * 1024)

    # Zu klein für YOLO
    if size_mb < 5:
        return False, f"Too small ({size_mb:.1f} MB)"

    # Zu groß für YOLO (selbst x ist ~140 MB)
    if size_mb > 200:
        return False, f"Too large ({size_mb:.1f} MB, likely Base Model)"

    # Load PyTorch Checkpoint
    try:
        import torch
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
    except Exception as e:
        # Fallback: Try ZIP-based analysis for old Ultralytics versions
        # PyTorch .pt files are ZIP archives internally
        zip_result = _analyze_yolo_via_zip(file_path)
        if zip_result:
            return True, f"Ultralytics YOLO detected (ZIP-fallback: {zip_result})"
        return False, f"Cannot load .pt file: {e}"

    # Check Checkpoint Structure
    if not isinstance(checkpoint, dict):
        return False, "Not a dict checkpoint"

    # Required keys for YOLO
    if 'model' not in checkpoint or 'train_args' not in checkpoint:
        return False, "Missing required keys (model, train_args)"

    # Extract state_dict
    # Try 'model' first, fallback to 'ema' (used by ADetailer/custom YOLO models)
    model = checkpoint.get('model')
    if model is None:
        model = checkpoint.get('ema')

    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        return False, "Cannot extract state_dict"

    tensor_keys = list(state_dict.keys())

    # Ultralytics Pattern: model.*, *.cv2.*, *.m.*
    has_model = any('model.' in k for k in tensor_keys)
    has_cv2 = any('.cv2.' in k for k in tensor_keys)
    has_m = any('.m.' in k for k in tensor_keys)

    if not (has_model and has_cv2 and has_m):
        return False, "No Ultralytics tensor pattern (missing model.*, .cv2.*, .m.*)"

    # EXCLUSION: YOLOX (different architecture, goes to annotator/)
    if 'yolox' in file_path.name.lower():
        return False, "YOLOX - different module (annotator/)"

    return True, "Ultralytics YOLO detected"


def _analyze_yolo_via_zip(file_path):
    """Fallback: Analysiert YOLO Model via ZIP wenn torch.load fehlschlägt.

    Für alte Ultralytics-Versionen die nicht mehr ladbar sind.

    Returns:
        str or None: Model-Typ ('DetectionModel'/'SegmentationModel') oder None
    """
    import zipfile

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Suche nach data.pkl (kann in verschiedenen Ordnern sein)
            pkl_files = [n for n in zf.namelist() if n.endswith('data.pkl')]
            if not pkl_files:
                return None

            # Lese erste data.pkl
            with zf.open(pkl_files[0]) as f:
                content = f.read().decode('latin-1', errors='replace')

            # Check für Ultralytics Model-Typen
            if 'ultralytics.nn.tasks' in content or 'ultralytics.yolo' in content:
                if 'DetectionModel' in content:
                    return 'DetectionModel'
                elif 'SegmentationModel' in content:
                    return 'SegmentationModel'
                else:
                    return 'UnknownModel'

            return None
    except Exception:
        return None


def _extract_yolo_info_via_zip(file_path):
    """Extrahiert YOLO-Infos aus ZIP wenn torch.load fehlschlägt.

    Returns:
        dict or None: {'model_type': 'bbox'/'segm', 'class_names': [...], 'num_classes': int}
    """
    import zipfile
    import re

    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            pkl_files = [n for n in zf.namelist() if n.endswith('data.pkl')]
            if not pkl_files:
                return None

            with zf.open(pkl_files[0]) as f:
                content_bytes = f.read()
                content_str = content_bytes.decode('latin-1', errors='replace')

            result = {}

            # Model Type
            if 'DetectionModel' in content_str:
                result['model_type'] = 'bbox'
            elif 'SegmentationModel' in content_str:
                result['model_type'] = 'segm'
            else:
                result['model_type'] = 'bbox'  # Default

            # Extract class names from the 'names' dict in pickle
            # Find the 'names' section and extract class names from there
            class_names = []

            # Pickle metadata keys to exclude (these are NOT class names!)
            pickle_metadata_keys = {
                'args', 'project', 'deterministic', 'resume', 'dropout', 'model', 'names',
                'epoch', 'optimizer', 'train_args', 'best_fitness', 'training', 'updates',
                'ema', 'date', 'version', 'license', 'docs', 'yaml', 'task', 'mode', 'data',
                'weight', 'running', 'bias', 'stride', 'padding', 'kernel', 'momentum',
                'affine', 'scale', 'conv', 'storage', 'zeros', 'inplace', 'eps', 'dilation',
                'groups', 'act', 'batch', 'epochs', 'patience', 'workers', 'exist_ok',
                'pretrained', 'verbose', 'seed', 'close_mosaic', 'cache', 'single_cls',
                'rect', 'cos_lr', 'overlap_mask', 'mask_ratio', 'half', 'dnn', 'plots',
                'save', 'conf', 'iou', 'max_det', 'vid_stride', 'device', 'imgsz', 'augment',
                'agnostic_nms', 'classes', 'retina_masks', 'boxes', 'format', 'keras',
                'optimize', 'int8', 'dynamic', 'simplify', 'opset', 'workspace', 'nms',
                'lr0', 'lrf', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
                'box', 'cls', 'dfl', 'pose', 'kobj', 'label_smoothing', 'nbs', 'hsv_h',
                'hsv_s', 'hsv_v', 'degrees', 'translate', 'shear', 'perspective',
                'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'cfg', 'tracker',
                'save_dir', 'source', 'show', 'save_txt', 'save_conf', 'save_crop',
                'show_labels', 'show_conf', 'show_boxes', 'line_width'
            }

            # Method 1: Look for 'names' dict pattern
            # Pattern: K\x00X\x0b\x00\x00\x00class_name (class_idx + length + string)
            names_idx = content_bytes.find(b'names')
            if names_idx > 0:
                # Extract 500 bytes after 'names' (limit to actual class dict, not train_args)
                names_section = content_bytes[names_idx:names_idx+500]
                # Pattern: K + class_idx + X + length_byte + \x00\x00\x00 + string
                class_pattern = rb'K([\x00-\x10])X([\x03-\x20])\x00\x00\x00([a-zA-Z][a-zA-Z0-9_]{2,25})'
                matches = re.findall(class_pattern, names_section)

                for idx_byte, length_byte, name_bytes in matches:
                    actual_length = length_byte[0] if isinstance(length_byte, bytes) else ord(length_byte)
                    name_str = name_bytes.decode('ascii', errors='ignore')[:actual_length]
                    # Filter: must be valid class name, not pickle metadata
                    if name_str and len(name_str) >= 3 and name_str.lower() not in pickle_metadata_keys:
                        class_names.append(name_str)

            # Method 2: Fallback - search for common class name patterns
            if not class_names:
                # Look for patterns that look like class names (lowercase with underscores)
                fallback_matches = re.findall(r'X[\x05-\x20]\x00\x00\x00([a-z][a-z0-9_]{2,25})', content_str)
                exclude = {'model', 'names', 'epoch', 'optimizer', 'train_args', 'best_fitness',
                           'training', 'updates', 'ema', 'date', 'version', 'license', 'docs',
                           'yaml', 'args', 'task', 'mode', 'data', 'weight', 'running', 'bias',
                           'stride', 'padding', 'kernel', 'momentum', 'affine', 'scale', 'conv',
                           'storage', 'zeros', 'inplace', 'eps', 'dilation', 'groups', 'act'}
                for m in fallback_matches:
                    if m not in exclude and not m.startswith('_'):
                        if not any(x in m for x in ['weight', 'bias', 'running', 'num_batches',
                                                     'channels', 'size', 'tracked', 'features']):
                            class_names.append(m)

            # Deduplicate while preserving order
            seen = set()
            unique_classes = []
            for c in class_names:
                if c not in seen:
                    seen.add(c)
                    unique_classes.append(c)

            result['class_names'] = unique_classes[:10]  # Limit to first 10
            result['num_classes'] = len(unique_classes)

            return result if result.get('model_type') else None

    except Exception:
        return None


def load_yolo_checkpoint(file_path):
    """Lädt YOLO Checkpoint mit Fallback für alte Ultralytics-Versionen.

    Returns:
        tuple: (checkpoint_or_None, zip_info_or_None, is_zip_fallback)
            - checkpoint: dict wenn torch.load erfolgreich, sonst None
            - zip_info: dict mit {'model_type', 'class_names', 'num_classes'} wenn ZIP-Fallback
            - is_zip_fallback: True wenn ZIP-Analyse verwendet wurde
    """
    try:
        import torch
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        return checkpoint, None, False
    except ImportError:
        # torch nicht verfügbar → ZIP-Fallback
        zip_info = _extract_yolo_info_via_zip(file_path)
        if zip_info:
            return None, zip_info, True
        return None, None, False
    except Exception:
        # torch vorhanden aber Laden fehlgeschlagen → ZIP-Fallback
        zip_info = _extract_yolo_info_via_zip(file_path)
        if zip_info:
            return None, zip_info, True
        return None, None, False


def _build_zip_specialization(class_names, filename):
    """Baut Specialization aus ZIP-extrahierten Klassen + Filename-Hints.

    Args:
        class_names: Liste der extrahierten Klassennamen
        filename: Original-Dateiname für Style-Hints (fantasy, anime, etc.)

    Returns:
        str: Specialization wie 'Breast_Fantasy', 'Nipple', 'Face_Anime'
    """
    import re

    # 1. Bestimme Haupt-Kategorie aus Klassennamen
    main_category = None

    if class_names:
        # Finde gemeinsamen Nenner in allen Klassen
        all_lower = [c.lower() for c in class_names]

        # NSFW-Kategorien (spezifisch, nicht generisch "NSFW")
        category_keywords = {
            'breast': 'Breast',
            'nipple': 'Nipple',
            'pussy': 'Pussy',
            'penis': 'Penis',
            'genital': 'Genital',
            'face': 'Face',
            'hand': 'Hand',
            'foot': 'Foot',
            'feet': 'Feet',
            'eye': 'Eye',
            'mouth': 'Mouth',
            'teeth': 'Teeth',
            'hair': 'Hair',
            'body': 'Body',
            'person': 'Person',
        }

        for keyword, category in category_keywords.items():
            if any(keyword in name for name in all_lower):
                main_category = category
                break

        # Fallback: Erste Klasse bereinigt verwenden
        if not main_category and class_names:
            first_class = class_names[0]
            # Clean up: remove _seg, _pair, _single suffixes, capitalize
            main_category = re.sub(r'_(seg|pair|single|left|right)$', '', first_class, flags=re.I)
            main_category = main_category.replace('_', ' ').title().replace(' ', '')

    # 2. Extrahiere Style-Hint aus Filename
    style_hint = None
    filename_lower = filename.lower()

    style_keywords = {
        'fantasy': 'Fantasy',
        'anime': 'Anime',
        'realistic': 'Realistic',
        'real': 'Real',
        'hentai': 'Hentai',
        '3d': '3D',
        'cartoon': 'Cartoon',
        'photo': 'Photo',
    }

    for keyword, style in style_keywords.items():
        if keyword in filename_lower:
            style_hint = style
            break

    # 3. Kombiniere: Category + Style (wenn vorhanden)
    if main_category and style_hint:
        return f"{main_category}_{style_hint}"
    elif main_category:
        return main_category
    elif style_hint:
        return style_hint
    else:
        return "Custom"


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_yolo_version(checkpoint, filename):
    """Erkennt YOLO Version (v5/v8/v11)

    Priorität:
    1. Metadata (version)
    2. Filename
    3. Fallback: YOLO8

    Returns:
        str: Version ('5', '8', '11')
    """
    # Priorität 1: Metadata Version
    version_str = checkpoint.get('version', '')
    if '11.' in version_str or 'v11' in version_str:
        return '11'
    elif '8.' in version_str or 'v8' in version_str:
        return '8'
    elif '5.' in version_str or 'v5' in version_str:
        return '5'

    # Priorität 2: Filename
    filename_lower = filename.lower()
    if 'yolo11' in filename_lower or 'v11' in filename_lower:
        return '11'
    elif 'yolo8' in filename_lower or 'yolov8' in filename_lower:
        return '8'
    elif 'yolo5' in filename_lower or 'yolov5' in filename_lower:
        return '5'

    # Fallback: v8 (häufigste Version)
    return '8'


def detect_model_size(checkpoint, filename, file_size_mb):
    """Erkennt Model Size (n/s/m/l/x)

    Priorität:
    1. Parameter Count (wenn verfügbar und > 0)
    2. Filename
    3. File Size

    Returns:
        str: Size ('n', 's', 'm', 'l', 'x')
    """
    # Priorität 1: Parameter Count (nur wenn checkpoint vorhanden)
    if checkpoint is None:
        model = None
    else:
        model = checkpoint.get('model')
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        state_dict = {}

    total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))

    # Grenzen basierend auf Analyse (22_Modul22_YOLO.md)
    if total_params > 0:
        if total_params < 5_000_000:
            return 'n'  # Nano: ~3M
        elif total_params < 15_000_000:
            return 's'  # Small: ~11M
        elif total_params < 35_000_000:
            return 'm'  # Medium: ~25M
        elif total_params < 55_000_000:
            return 'l'  # Large: ~43M
        else:
            return 'x'  # Extra-Large: ~68M

    # Priorität 2: Filename
    filename_lower = filename.lower()
    for size in ['n', 's', 'm', 'l', 'x']:
        # Pattern: yolo8n, yolo8s, etc.
        if f'yolo{size}' in filename_lower or f'v{size}' in filename_lower:
            return size

    # Priorität 3: File Size
    if file_size_mb < 15:
        return 'n'
    elif file_size_mb < 30:
        return 's'
    elif file_size_mb < 60:
        return 'm'
    elif file_size_mb < 100:
        return 'l'
    else:
        return 'x'


def detect_task_type(checkpoint):
    """Erkennt Task Type (detect/segment/pose/classify/obb)

    WICHTIG: Tensor-Pattern alleine kann NICHT unterscheiden!
    → Metadata train_args['task'] ist kritisch!

    Returns:
        str: Task Type ('detect', 'segment', 'pose', 'classify', 'obb')
    """
    train_args = checkpoint.get('train_args', {})

    # Convert to dict if it's a namespace or other object
    if hasattr(train_args, '__dict__'):
        train_args = vars(train_args)

    task = train_args.get('task', 'detect')

    # Normalize task names
    if task in ['detect', 'detection']:
        return 'detect'
    elif task in ['segment', 'segmentation', 'seg']:
        return 'segment'
    elif task in ['pose', 'keypoint']:
        return 'pose'
    elif task in ['classify', 'classification', 'cls']:
        return 'classify'
    elif task in ['obb', 'oriented']:
        return 'obb'

    # Fallback
    return 'detect'


def get_output_type(task):
    """Konvertiert Task Type zu Output Type für Naming

    Returns:
        str: Output Type ('bbox', 'segm', 'pose', 'class', 'obb')
    """
    mapping = {
        'detect': 'bbox',
        'segment': 'segm',
        'pose': 'pose',
        'classify': 'class',
        'obb': 'obb'
    }
    return mapping.get(task, 'bbox')


def get_target_folder(output_type):
    """Bestimmt Ziel-Ordner basierend auf Output Type

    Returns:
        Path: Ziel-Ordner
    """
    # Segmentation → segm/, alle anderen → bbox/
    if output_type == 'segm':
        return ULTRALYTICS_SEGM_DIR
    else:
        return ULTRALYTICS_BBOX_DIR


def detect_specialization(checkpoint, filename):
    """Erkennt Specialization (Face, Hand, COCO80, Custom, etc.)

    Priorität:
    1. Class Names aus Metadata (Single-Class → direkt verwenden)
    2. Multi-Class: NSFW-Detection oder Filename-Keywords
    3. Fallback: Bereinigter Original-Filename

    Returns:
        str: Specialization ('Face', 'Hand', 'COCO80', 'Person', 'NSFW7', etc.)
    """
    # Priorität 1: Class Names aus Metadata
    # Skip if checkpoint is None (ZIP-fallback mode)
    if checkpoint is None:
        model = None
    else:
        # Try 'model' first, fallback to 'ema' (used by ADetailer/custom YOLO models)
        model = checkpoint.get('model')
        if model is None:
            model = checkpoint.get('ema')

    # Try to get class names from model
    class_names = None

    if hasattr(model, 'names'):
        class_names = model.names
    elif isinstance(model, dict) and 'names' in model:
        class_names = model['names']

    # Also check train_args
    if not class_names:
        train_args = checkpoint.get('train_args', {})
        if hasattr(train_args, '__dict__'):
            train_args = vars(train_args)
        if 'names' in train_args:
            class_names = train_args.get('names')

    # Analyze class names
    if class_names:
        # Convert to list if dict
        if isinstance(class_names, dict):
            class_list = list(class_names.values())
        else:
            class_list = class_names

        num_classes = len(class_list)

        # Single class models - use class name directly
        if num_classes == 1:
            class_name = class_list[0].lower()

            # Special case: Generic 'object' class - try to get real name from train_args.data
            if class_name == 'object':
                train_args = checkpoint.get('train_args', {})
                if hasattr(train_args, '__dict__'):
                    train_args = vars(train_args)
                data_path = train_args.get('data', '') if isinstance(train_args, dict) else ''
                if data_path:
                    import re
                    # Extract folder name from path like "datasets\Feet\data.yaml"
                    # or "E:/Yolov8/models/Project_Jewels/Project_Jewels.yaml"
                    path_parts = re.split(r'[/\\]', data_path)
                    # Get the folder name (second to last) or yaml name without extension
                    for part in reversed(path_parts):
                        if part.endswith('.yaml'):
                            # Try folder name first
                            continue
                        if part and part.lower() not in ['datasets', 'data', 'models']:
                            # Clean up: remove underscores, capitalize
                            clean_name = part.replace('_', ' ')
                            clean_name = ''.join(word.capitalize() for word in clean_name.split())
                            return clean_name

            # Map common class names to standardized names
            if 'face' in class_name or 'head' in class_name:
                return 'Face'
            elif 'hand' in class_name:
                return 'Hand'
            elif 'person' in class_name or 'people' in class_name:
                return 'Person'
            elif 'penis' in class_name or 'cock' in class_name or 'dick' in class_name:
                return 'Penis'
            elif 'breast' in class_name or 'boob' in class_name:
                return 'Breast'
            elif 'earring' in class_name:
                return 'Earring'
            else:
                # Clean up class name: remove _seg suffix, replace underscores, capitalize
                clean_name = class_list[0]
                clean_name = clean_name.replace('_seg', '').replace('_', ' ')
                # Capitalize each word and join
                clean_name = ''.join(word.capitalize() for word in clean_name.split())
                return clean_name

        # COCO dataset (80 classes)
        elif num_classes == 80:
            # Check if it's standard COCO
            coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane']
            if all(c in [name.lower() for name in class_list[:5]] for c in coco_classes):
                return 'COCO80'

        # Multi-class models
        # Check for NSFW content (pussy, penis, nipples, anus, etc.)
        nsfw_keywords = ['pussy', 'penis', 'nipple', 'nipples', 'anus', 'testicle', 'cock', 'dick', 'boob', 'breast']
        if any(any(kw in str(c).lower() for kw in nsfw_keywords) for c in class_list):
            return f'NSFW{num_classes}'

        # Check for known patterns in filename
        filename_lower = filename.lower()
        if 'jewelry' in filename_lower or 'jewel' in filename_lower:
            return 'Jewelry'
        elif 'clothing' in filename_lower or 'clothes' in filename_lower:
            return 'Clothing'
        elif 'animal' in filename_lower:
            return 'Animals'
        elif 'vehicle' in filename_lower:
            return 'Vehicles'
        elif 'food' in filename_lower:
            return 'Food'

        # Check for body-related
        if all('person' in str(c).lower() or 'body' in str(c).lower() for c in class_list):
            return 'Body'

        # Fallback for multi-class: Extract meaningful name from filename
        # Remove common prefixes/suffixes and version numbers
        import re
        clean_filename = filename_lower
        # Remove extension
        clean_filename = re.sub(r'\.(pt|pth)$', '', clean_filename)
        # Remove version patterns
        clean_filename = re.sub(r'_?v\d+(\.\d+)?$', '', clean_filename)
        clean_filename = re.sub(r'_v\d+$', '', clean_filename)
        # Remove common prefixes
        for prefix in ['adetailer', 'yolo', 'yolov5', 'yolov8', 'yolov11', 'ultralytics']:
            clean_filename = re.sub(f'^{prefix}_?', '', clean_filename)
        # Remove size indicators
        clean_filename = re.sub(r'^[nsmblx]_', '', clean_filename)
        # Clean up
        clean_filename = clean_filename.strip('_- ')

        if clean_filename:
            # Capitalize properly
            return ''.join(word.capitalize() for word in clean_filename.replace('_', ' ').split())

        # Ultimate fallback
        return f'Custom{num_classes}'

    # Priorität 2: Filename Keywords (wenn keine Metadata)
    filename_lower = filename.lower()

    if 'face' in filename_lower:
        return 'Face'
    elif 'hand' in filename_lower:
        return 'Hand'
    elif 'person' in filename_lower:
        return 'Person'
    elif 'penis' in filename_lower or 'cock' in filename_lower:
        return 'Penis'
    elif 'nipple' in filename_lower:
        return 'Nipple'
    elif 'pussy' in filename_lower:
        return 'Pussy'
    elif 'breast' in filename_lower or 'boob' in filename_lower:
        return 'Breast'
    elif 'jewelry' in filename_lower or 'jewel' in filename_lower:
        return 'Jewelry'
    elif 'nsfw' in filename_lower:
        return 'NSFW'
    elif 'coco' in filename_lower:
        return 'COCO80'
    elif 'pose' in filename_lower:
        return 'Body'

    # Fallback: Try to extract from filename
    import re
    clean_filename = filename.lower()
    clean_filename = re.sub(r'\.(pt|pth)$', '', clean_filename)
    clean_filename = re.sub(r'_?v\d+(\.\d+)?$', '', clean_filename)
    for prefix in ['adetailer', 'yolo', 'yolov5', 'yolov8', 'yolov11']:
        clean_filename = re.sub(f'^{prefix}_?', '', clean_filename)
    clean_filename = clean_filename.strip('_- ')

    if clean_filename:
        return ''.join(word.capitalize() for word in clean_filename.replace('_', ' ').split())

    return 'Object'


def detect_version_suffix(filename):
    """Erkennt Version Suffix aus Filename (wenn vorhanden)

    Returns:
        str or None: Version Suffix (z.B. "v5", "v4.7") oder None
    """
    import re

    # Pattern: _v5, _v4.7, -v10, etc.
    version_match = re.search(r'[_-]v(\d+(?:\.\d+)?)', filename.lower())

    if version_match:
        return f"v{version_match.group(1)}"

    return None


def generate_proper_name(version, size, output_type, specialization, version_suffix):
    """Generiert standardisierten Namen

    Format: YOLO_{Version}{Size}_{OutputType}_{Specialization}[_vX].pt

    Returns:
        str: Generierter Filename
    """
    # Build parts
    parts = [f'YOLO_{version}{size}', output_type, specialization]

    # Add version suffix if present
    if version_suffix:
        parts.append(version_suffix)

    # Join with underscore
    proper_name = '_'.join(parts) + '.pt'

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
        module_name_caps="YOLO (INSTALLATION)",
        downloads_path=DOWNLOADS_DIR,
        extensions="*.pt",
        module_type="YOLO",
        target_folders="ultralytics/bbox/, ultralytics/segm/"
    )

    all_files = list(DOWNLOADS_DIR.glob("**/*.pt"))  # recursive

    if not all_files:
        print_no_files_found("YOLO files")
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

        is_match, reason = is_yolo_ultralytics(file_path)
        if not is_match:
            skipped.append({'filename': filename, 'size_mb': size_mb})
            continue

        # Load checkpoint with ZIP-fallback for old Ultralytics versions
        checkpoint, zip_info, is_zip_fallback = load_yolo_checkpoint(file_path)

        if is_zip_fallback and zip_info:
            # ZIP-Fallback: Use extracted info
            version = '8'  # Default, can't detect from ZIP
            model_size = detect_model_size(None, filename, size_mb)
            output_type = zip_info['model_type']
            # Build specialization from ZIP-extracted class names + filename hints
            class_names = zip_info.get('class_names', [])
            specialization = _build_zip_specialization(class_names, filename)
            version_suffix = detect_version_suffix(filename)
        else:
            # Normal: Use checkpoint
            version = detect_yolo_version(checkpoint, filename)
            model_size = detect_model_size(checkpoint, filename, size_mb)
            task = detect_task_type(checkpoint)
            output_type = get_output_type(task)
            specialization = detect_specialization(checkpoint, filename)
            version_suffix = detect_version_suffix(filename)

        target_folder = get_target_folder(output_type)
        proper_name = generate_proper_name(version, model_size, output_type, specialization, version_suffix)

        files_to_install.append({
            'path': file_path,
            'filename': filename,
            'proper_name': proper_name,
            'target_folder': target_folder,
            'detected_str': f"YOLO{version}{model_size}, {output_type}, {specialization}",
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
        skip_reason="Not a YOLO model",
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
    """Modus B: Check bestehender YOLO Models"""

    # ========================================================================
    # HEADER (using shared helper)
    # ========================================================================
    print_mode_b_header(
        module_name="YOLO",
        folders="ultralytics/bbox/, ultralytics/segm/",
        extensions="*.pt",
        module_type="YOLO",
        target_folders="ultralytics/bbox/, ultralytics/segm/",
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
                is_match, reason = is_yolo_ultralytics(file_path)
                if not is_match:
                    continue

                filename = file_path.name
                file_size_mb = file_path.stat().st_size / (1024 * 1024)

                # Load checkpoint with ZIP-fallback for old Ultralytics versions
                checkpoint, zip_info, is_zip_fallback = load_yolo_checkpoint(file_path)

                if checkpoint is None and zip_info is None:
                    print(f"[ERROR] Cannot load YOLO checkpoint {filename}")
                    continue

                if is_zip_fallback and zip_info:
                    # ZIP-Fallback: Use extracted info
                    version = '8'  # Default for old models
                    model_size = detect_model_size(None, filename, file_size_mb)
                    output_type = zip_info['model_type']
                    class_names = zip_info.get('class_names', [])
                    specialization = _build_zip_specialization(class_names, filename)
                    version_suffix = detect_version_suffix(filename)
                else:
                    # Normal: Use checkpoint
                    version = detect_yolo_version(checkpoint, filename)
                    model_size = detect_model_size(checkpoint, filename, file_size_mb)
                    task = detect_task_type(checkpoint)
                    output_type = get_output_type(task)
                    specialization = detect_specialization(checkpoint, filename)
                    version_suffix = detect_version_suffix(filename)

                target_folder = get_target_folder(output_type)
                proper_name = generate_proper_name(version, model_size, output_type, specialization, version_suffix)
                target_path = target_folder / proper_name

                print(f"[RESCUE] Found misplaced YOLO: {file_path.parent.name}/{filename} ({file_size_mb:.0f} MB)")

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
            print(f"[SUCCESS] Rescued {rescued} misplaced YOLO model(s)")
            print()

    # ========================================================================
    # SCAN OWN FOLDERS
    # ========================================================================
    folders_to_scan = [
        (ULTRALYTICS_BBOX_DIR, "ultralytics/bbox"),
        (ULTRALYTICS_SEGM_DIR, "ultralytics/segm")
    ]

    all_files = []
    for folder, folder_name in folders_to_scan:
        if folder.exists():
            all_files += [(f, folder_name) for f in folder.glob("**/*.pt")]  # recursive

    if not all_files:
        print_no_files_found("YOLO files")
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

        is_match, reason = is_yolo_ultralytics(file_path)
        if not is_match:
            if scan_only:
                add_misplaced_file(file_path)
            else:
                problems_list.append(('misplaced', filename, file_size_mb, reason, current_folder_name))
            continue

        if scan_only:
            continue

        # Load checkpoint with ZIP-fallback for old Ultralytics versions
        checkpoint, zip_info, is_zip_fallback = load_yolo_checkpoint(file_path)

        if is_zip_fallback and zip_info:
            # ZIP-Fallback: Use extracted info
            version = '8'
            model_size = detect_model_size(None, filename, file_size_mb)
            output_type = zip_info['model_type']
            class_names = zip_info.get('class_names', [])
            specialization = _build_zip_specialization(class_names, filename)
            version_suffix = detect_version_suffix(filename)
        else:
            version = detect_yolo_version(checkpoint, filename)
            model_size = detect_model_size(checkpoint, filename, file_size_mb)
            task = detect_task_type(checkpoint)
            output_type = get_output_type(task)
            specialization = detect_specialization(checkpoint, filename)
            version_suffix = detect_version_suffix(filename)

        target_folder = get_target_folder(output_type)
        proper_name = generate_proper_name(version, model_size, output_type, specialization, version_suffix)

        detected_str = f"YOLO{version}{model_size}, {output_type}, {specialization}"

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
                    warning=f"Not a YOLO model: {reason}"
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
    """Scannt downloads/ für YOLO Models (für all_modules.py Preview)

    Returns:
        list: Liste von dicts mit file info
    """
    results = []

    # Finde alle .pt Dateien (recursive)
    all_files = list(downloads_path.glob("**/*.pt"))  # recursive

    for file_path in all_files:
        # Module Boundary Check
        is_match, reason = is_yolo_ultralytics(file_path)
        if not is_match:
            continue

        filename = file_path.name
        size_mb = file_path.stat().st_size / (1024 * 1024)

        # Load checkpoint (with ZIP-fallback for old Ultralytics versions)
        checkpoint, zip_info, is_zip_fallback = load_yolo_checkpoint(file_path)

        if is_zip_fallback and zip_info:
            # ZIP-Fallback Mode: Use extracted info
            version = '8'  # Default assumption for old models
            model_size = detect_model_size(None, filename, size_mb)
            output_type = zip_info.get('model_type', 'bbox')

            # Build specialization from ZIP-extracted class names + filename hints
            class_names = zip_info.get('class_names', [])
            specialization = _build_zip_specialization(class_names, filename)

            version_suffix = detect_version_suffix(filename)
        else:
            # Normal Mode: Full checkpoint analysis
            version = detect_yolo_version(checkpoint, filename)
            model_size = detect_model_size(checkpoint, filename, size_mb)
            task = detect_task_type(checkpoint)
            output_type = get_output_type(task)
            specialization = detect_specialization(checkpoint, filename)
            version_suffix = detect_version_suffix(filename)

        # Target Folder
        target_folder = get_target_folder(output_type)

        # Generate proper name
        proper_name = generate_proper_name(version, model_size, output_type, specialization, version_suffix)

        # Size
        size_gb = file_path.stat().st_size / (1024**3)

        results.append({
            'file_path': file_path,
            'size_gb': size_gb,
            'result': {
                'version': version,
                'size': model_size,
                'output_type': output_type,
                'specialization': specialization,
                'version_suffix': version_suffix,
                'proper_name': proper_name,
                'target_folder': target_folder
            }
        })

    return {
        'module_name': 'YOLO Models',
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
        print("  python modul22_yolo.py A          - Modus A (Installation)")
        print("  python modul22_yolo.py B          - Modus B (Reinstall/Check)")
        print("  python modul22_yolo.py B --scan-only    - Modus B PASS 1")
        print("  python modul22_yolo.py B --preview      - Modus B Preview")
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
