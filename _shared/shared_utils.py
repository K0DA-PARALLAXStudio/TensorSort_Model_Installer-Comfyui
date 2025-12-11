#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Utilities für alle Module
TensorSort Model Installer - Test Version
"""

__version__ = "1.1.0"

import os
import hashlib
import shutil
import json
import warnings
from pathlib import Path

# Suppress pynvml FutureWarning (torch.cuda uses deprecated pynvml package)
# WICHTIG: Diese Filter müssen VOR dem ersten torch-Import stehen!
warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")

# Torch hier importieren damit die Warning-Filter greifen
# Module die torch brauchen können es von hier importieren oder selbst importieren
# (der Filter ist dann bereits aktiv)
try:
    import torch  # noqa: F401 - imported for side effect (warning suppression)
except ImportError:
    pass  # torch nicht installiert - ok, nicht alle Module brauchen es

# ============================================================================
# CONFIG FILE
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
# Config file is in Pro_Version root (parent of _shared/)
PRO_VERSION_DIR = SCRIPT_DIR.parent
CONFIG_FILE = PRO_VERSION_DIR / "config_tensorsort.json"


def load_config():
    """Loads config from JSON (if exists)"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {
                    'comfyui_path': Path(config.get('comfyui_path', '')) if config.get('comfyui_path') else None,
                    'downloads_path': Path(config.get('downloads_path', '')) if config.get('downloads_path') else None
                }
        except Exception as e:
            print(f"[WARNING] Failed to read config: {e}")
    return {'comfyui_path': None, 'downloads_path': None}


def save_config(comfyui_path, downloads_path):
    """Saves config as JSON"""
    config = {
        'comfyui_path': str(comfyui_path) if comfyui_path else '',
        'downloads_path': str(downloads_path) if downloads_path else ''
    }
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")
        return False


# ============================================================================
# PATH AUTO-DETECTION
# ============================================================================

def auto_detect_comfyui():
    """Tries to find ComfyUI automatically

    Returns:
        Path or None
    """
    script_dir = Path(__file__).parent
    parent = script_dir.parent

    # Option 1: Parent is ComfyUI
    if parent.name.lower() == 'comfyui':
        if (parent / 'models').exists():
            return parent

    # Option 2: Grandparent Check (downloads/NewReinstaller/ -> downloads/ -> Parent/)
    grandparent = parent.parent
    if grandparent.exists():
        for sibling in grandparent.iterdir():
            if sibling.is_dir() and sibling.name.lower() == 'comfyui':
                if (sibling / 'models').exists():
                    return sibling

    # Option 3: Sibling Check (Script next to ComfyUI)
    for sibling in parent.iterdir():
        if sibling.is_dir() and sibling.name.lower() == 'comfyui':
            if (sibling / 'models').exists():
                return sibling

    return None


def auto_detect_downloads():
    """Tries to find a sensible downloads folder

    Returns:
        Path or None
    """
    script_dir = Path(__file__).parent
    parent = script_dir.parent

    # Option 1: Parent is named "downloads"
    if parent.name.lower() == 'downloads':
        return parent

    # Option 2: Script is inside ComfyUI, use ComfyUI/downloads
    if parent.name.lower() == 'comfyui':
        downloads = parent / 'downloads'
        if downloads.exists():
            return downloads

    # Option 3: Sibling folder named "downloads"
    for sibling in parent.iterdir():
        if sibling.is_dir() and sibling.name.lower() == 'downloads':
            return sibling

    # Option 4: Check if ComfyUI was found, use downloads next to it
    comfyui = auto_detect_comfyui()
    if comfyui:
        downloads_inside = comfyui / 'downloads'
        if downloads_inside.exists():
            return downloads_inside
        downloads_sibling = comfyui.parent / 'downloads'
        if downloads_sibling.exists():
            return downloads_sibling

    return None


def validate_comfyui_path(path):
    """Checks if path is a valid ComfyUI installation"""
    path = Path(path)
    if not path.exists():
        return False, "Path does not exist"
    if not (path / 'models').exists():
        return False, "No 'models' folder found"
    return True, "OK"


def validate_downloads_path(path):
    """Checks if path exists (or can be created)"""
    path = Path(path)
    if path.exists():
        return True, "OK"
    if path.parent.exists():
        return True, "OK (will be created)"
    return False, "Path and parent do not exist"


def ask_path_with_examples(step, total, title, description, examples, validator, allow_cancel=True):
    """Asks user for path with helpful UI

    Args:
        step: Current step number (1, 2, ...)
        total: Total steps
        title: Section title
        description: What this path is for
        examples: List of example paths
        validator: Function to validate path
        allow_cancel: If True, 0 = cancel

    Returns:
        Path or None (if cancelled)
    """
    print()
    print("-" * 60)
    print(f"  [{step}/{total}] {title}")
    print("-" * 60)
    print()
    print(f"  {description}")
    print()
    print("  Examples:")
    for ex in examples:
        print(f"    {ex}")
    print()

    while True:
        if allow_cancel:
            user_input = input("  Enter path (0 = cancel): ").strip()
        else:
            user_input = input("  Enter path: ").strip()

        # Cancel option
        if user_input == '0' and allow_cancel:
            return None

        if not user_input:
            print("  [!] Input required")
            continue

        user_input = user_input.strip('"').strip("'")

        valid, msg = validator(user_input)
        if valid:
            print(f"  [OK] Valid! {msg}")
            return Path(user_input)
        else:
            print(f"  [!] Invalid: {msg}")
            print("  Please try again.")
            print()


def ask_path(prompt, validator, current=None, allow_cancel=True):
    """Asks user for path with validation (simple version for settings menu)

    Args:
        prompt: Text to show
        validator: Function to validate path
        current: Current value (optional)
        allow_cancel: If True, empty input with no current = None (cancel)

    Returns:
        Path or None (if cancelled)
    """
    while True:
        if current:
            print(f"  Current: {current}")
        if allow_cancel and not current:
            print("  (Enter 0 to cancel)")
        user_input = input(f"  {prompt}: ").strip()

        # Cancel option
        if user_input == '0' and allow_cancel:
            return None

        if not user_input:
            if current:
                return current
            print("  [!] Input required")
            continue

        user_input = user_input.strip('"').strip("'")

        valid, msg = validator(user_input)
        if valid:
            return Path(user_input)
        else:
            print(f"  [!] Invalid: {msg}")
            print("  Please try again.")


def initialize_paths():
    """Initializes paths from config or auto-detection, asks user if needed

    Returns:
        tuple: (comfyui_path, downloads_path)

    Exits:
        sys.exit(1) if user cancels or paths cannot be determined
    """
    import sys

    # 1. Try to load config
    config = load_config()
    comfyui_path = config['comfyui_path']
    downloads_path = config['downloads_path']

    config_changed = False

    # 2. Validate config paths
    if comfyui_path:
        valid, msg = validate_comfyui_path(comfyui_path)
        if not valid:
            print(f"[WARNING] Saved ComfyUI path invalid: {msg}")
            comfyui_path = None

    if downloads_path:
        valid, msg = validate_downloads_path(downloads_path)
        if not valid:
            print(f"[WARNING] Saved downloads path invalid: {msg}")
            downloads_path = None

    # 3. Auto-detection for missing paths
    if not comfyui_path:
        detected = auto_detect_comfyui()
        if detected:
            comfyui_path = detected
            config_changed = True

    if not downloads_path:
        detected = auto_detect_downloads()
        if detected:
            downloads_path = detected
            config_changed = True

    # 4. If still missing -> ask user with friendly setup wizard
    needs_comfyui = not comfyui_path
    needs_downloads = not downloads_path

    if needs_comfyui or needs_downloads:
        # Show welcome header
        print()
        print("=" * 60)
        print("     TENSORSORT MODEL INSTALLER - FIRST TIME SETUP")
        print("=" * 60)
        print()
        print("  Welcome! This tool organizes your ComfyUI model files.")
        print()
        total_steps = (1 if needs_comfyui else 0) + (1 if needs_downloads else 0)
        current_step = 0

    if needs_comfyui:
        current_step += 1
        comfyui_path = ask_path_with_examples(
            step=current_step,
            total=total_steps,
            title="ComfyUI Installation",
            description="Where is ComfyUI installed?\n  (The folder containing 'models/', 'custom_nodes/', etc.)",
            examples=[
                "C:\\ComfyUI",
                "D:\\AI\\ComfyUI_windows_portable\\ComfyUI",
                "E:\\StableDiffusion\\ComfyUI"
            ],
            validator=validate_comfyui_path
        )
        if comfyui_path is None:
            print()
            print("  [CANCELLED] Setup cancelled. Exiting.")
            sys.exit(1)
        config_changed = True

    if needs_downloads:
        current_step += 1
        downloads_path = ask_path_with_examples(
            step=current_step,
            total=total_steps,
            title="Downloads Folder",
            description="Where do you download model files?\n  (New models will be scanned from here)",
            examples=[
                "C:\\Users\\YourName\\Downloads",
                "D:\\AI\\models_to_sort",
                "E:\\Downloads\\AI_Models"
            ],
            validator=validate_downloads_path
        )
        if downloads_path is None:
            print()
            print("  [CANCELLED] Setup cancelled. Exiting.")
            sys.exit(1)
        config_changed = True

    # 5. Save config if changed
    if config_changed:
        save_config(comfyui_path, downloads_path)

        # Show completion message
        print()
        print("=" * 60)
        print("  SETUP COMPLETE")
        print("=" * 60)
        print()
        print(f"  ComfyUI:   {comfyui_path}")
        print(f"  Downloads: {downloads_path}")
        print()
        print(f"  Config saved. You can change these later in Settings [4].")
        print()
        input("  Press Enter to continue...")
        print()

    return comfyui_path, downloads_path


def run_settings_menu():
    """Settings menu for changing paths"""
    global COMFYUI_BASE, MODELS_DIR, DOWNLOADS_DIR

    comfyui_path = COMFYUI_BASE
    downloads_path = DOWNLOADS_DIR

    while True:
        print()
        print("=" * 60)
        print("  SETTINGS")
        print("=" * 60)
        print()
        print(f"  [1] ComfyUI path:   {comfyui_path}")
        print(f"  [2] Downloads path: {downloads_path}")
        print()
        print("  [0] Back")
        print()

        choice = input("Choice [1/2/0]: ").strip()

        if choice == '1':
            comfyui_path = ask_path("New ComfyUI path", validate_comfyui_path, comfyui_path)
            save_config(comfyui_path, downloads_path)
            # Update globals
            COMFYUI_BASE = comfyui_path
            MODELS_DIR = COMFYUI_BASE / "models"
            _update_model_dirs()
            print("[OK] Config saved")
        elif choice == '2':
            downloads_path = ask_path("New downloads path", validate_downloads_path, downloads_path)
            save_config(comfyui_path, downloads_path)
            DOWNLOADS_DIR = downloads_path
            print("[OK] Config saved")
        elif choice == '0':
            break
        else:
            print("Invalid choice!")

    return comfyui_path, downloads_path


def _update_model_dirs():
    """Updates all model directory globals after MODELS_DIR change"""
    global CHECKPOINTS_DIR, UNET_DIR, VAE_DIR, CLIP_DIR, TEXT_ENCODERS_DIR
    global CLIP_VISION_DIR, VLM_DIR, LLM_DIR, LORAS_DIR, LORAS_LYCORIS_DIR
    global CONTROLNET_DIR, T2I_ADAPTER_DIR, EMBEDDINGS_DIR, UPSCALE_MODELS_DIR
    global IPADAPTER_DIR, IPADAPTER_FLUX_DIR, XLABS_DIR, XLABS_IPADAPTER_DIR
    global PHOTOMAKER_DIR, VAE_APPROX_DIR, ANIMATEDIFF_MODELS_DIR, MOTION_LORA_DIR
    global ULTRALYTICS_DIR, ULTRALYTICS_BBOX_DIR, ULTRALYTICS_SEGM_DIR
    global INSIGHTFACE_DIR, SAMS_DIR, GROUNDING_DINO_DIR

    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    UNET_DIR = MODELS_DIR / "unet"
    VAE_DIR = MODELS_DIR / "vae"
    CLIP_DIR = MODELS_DIR / "clip"
    TEXT_ENCODERS_DIR = MODELS_DIR / "text_encoders"
    CLIP_VISION_DIR = MODELS_DIR / "clip_vision"
    VLM_DIR = MODELS_DIR / "VLM"
    LLM_DIR = MODELS_DIR / "LLM"
    LORAS_DIR = MODELS_DIR / "loras"
    LORAS_LYCORIS_DIR = LORAS_DIR / "LyCORIS"
    CONTROLNET_DIR = MODELS_DIR / "controlnet"
    T2I_ADAPTER_DIR = MODELS_DIR / "t2i_adapter"
    EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
    UPSCALE_MODELS_DIR = MODELS_DIR / "upscale_models"
    IPADAPTER_DIR = MODELS_DIR / "ipadapter"
    IPADAPTER_FLUX_DIR = MODELS_DIR / "ipadapter-flux"
    XLABS_DIR = MODELS_DIR / "xlabs"
    XLABS_IPADAPTER_DIR = XLABS_DIR / "ipadapters"
    PHOTOMAKER_DIR = MODELS_DIR / "photomaker"
    VAE_APPROX_DIR = MODELS_DIR / "vae_approx"
    ANIMATEDIFF_MODELS_DIR = MODELS_DIR / "animatediff_models"
    MOTION_LORA_DIR = MODELS_DIR / "animatediff_motion_lora"
    ULTRALYTICS_DIR = MODELS_DIR / "ultralytics"
    ULTRALYTICS_BBOX_DIR = ULTRALYTICS_DIR / "bbox"
    ULTRALYTICS_SEGM_DIR = ULTRALYTICS_DIR / "segm"
    INSIGHTFACE_DIR = MODELS_DIR / "insightface"
    SAMS_DIR = MODELS_DIR / "sams"
    GROUNDING_DINO_DIR = MODELS_DIR / "grounding-dino"


# ============================================================================
# INITIALIZE PATHS ON IMPORT
# ============================================================================

COMFYUI_BASE, DOWNLOADS_DIR = initialize_paths()
MODELS_DIR = COMFYUI_BASE / "models"

# Target Folders (für alle Module)
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
UNET_DIR = MODELS_DIR / "unet"
VAE_DIR = MODELS_DIR / "vae"
CLIP_DIR = MODELS_DIR / "clip"
TEXT_ENCODERS_DIR = MODELS_DIR / "text_encoders"
CLIP_VISION_DIR = MODELS_DIR / "clip_vision"
VLM_DIR = MODELS_DIR / "VLM"
LLM_DIR = MODELS_DIR / "LLM"
LORAS_DIR = MODELS_DIR / "loras"
LORAS_LYCORIS_DIR = LORAS_DIR / "LyCORIS"
CONTROLNET_DIR = MODELS_DIR / "controlnet"
T2I_ADAPTER_DIR = MODELS_DIR / "t2i_adapter"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
UPSCALE_MODELS_DIR = MODELS_DIR / "upscale_models"
IPADAPTER_DIR = MODELS_DIR / "ipadapter"
IPADAPTER_FLUX_DIR = MODELS_DIR / "ipadapter-flux"
XLABS_DIR = MODELS_DIR / "xlabs"
XLABS_IPADAPTER_DIR = XLABS_DIR / "ipadapters"
PHOTOMAKER_DIR = MODELS_DIR / "photomaker"
VAE_APPROX_DIR = MODELS_DIR / "vae_approx"
ANIMATEDIFF_MODELS_DIR = MODELS_DIR / "animatediff_models"  # Motion Modules (SD15/SDXL)
MOTION_LORA_DIR = MODELS_DIR / "animatediff_motion_lora"    # Motion LoRAs (camera movement)
ULTRALYTICS_DIR = MODELS_DIR / "ultralytics"
ULTRALYTICS_BBOX_DIR = ULTRALYTICS_DIR / "bbox"
ULTRALYTICS_SEGM_DIR = ULTRALYTICS_DIR / "segm"
INSIGHTFACE_DIR = MODELS_DIR / "insightface"
SAMS_DIR = MODELS_DIR / "sams"
GROUNDING_DINO_DIR = MODELS_DIR / "grounding-dino"

# Queue File (für Misplaced Files Registry)
QUEUE_FILE = SCRIPT_DIR / "misplaced_files_queue.txt"

# ============================================================================
# DUPLICATE DETECTION
# ============================================================================

def calc_hash(file_path):
    """[TEST VERSION] Hash calculation disabled"""
    return "TEST_VERSION_NO_HASH"


def check_duplicate(source_path, target_path):
    """[TEST VERSION] Duplicate check disabled - always returns 'not exists'"""
    return {
        "exists": False,
        "duplicate": False,
        "size_match": False,
        "hash_match": False,
        "message": "[TEST] Duplicate check disabled"
    }


def find_free_suffix(target_path):
    """[TEST VERSION] Suffix generation disabled"""
    return Path(target_path)


def handle_duplicate_move(source_path, target_path, expected_target_name=None, mode="A", keep_source_option=False, dry_run=False):
    """[TEST VERSION] Move/Copy operations disabled - Preview only"""
    target_path = Path(target_path)
    print()
    print("=" * 60)
    print("  TEST VERSION - Execution disabled")
    print("  Full functionality available in Pro Version")
    print("  Email: kodaparallax@gmail.com")
    print("=" * 60)
    print()
    return (True, target_path, "[TEST] Would install to: " + str(target_path))


# ============================================================================
# MISPLACED FILES QUEUE (Modus B Cross-Module Rescue)
# ============================================================================

def read_misplaced_files():
    """Liest Queue falsch platzierter Dateien

    Returns:
        list: Liste absoluter Pfade zu falsch platzierten Dateien
    """
    if not QUEUE_FILE.exists():
        return []

    misplaced = []
    with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            rel_path = line.strip()
            # Skip Kommentare und leere Zeilen
            if not rel_path or rel_path.startswith('#'):
                continue

            # Konvertiere zu absolutem Pfad
            abs_path = MODELS_DIR / rel_path
            if abs_path.exists():
                misplaced.append(abs_path)

    return misplaced


def add_misplaced_file(file_path):
    """Fügt Datei zur Queue hinzu

    Args:
        file_path (Path/str): Absoluter Pfad zur falsch platzierten Datei
    """
    file_path = Path(file_path)

    # Konvertiere zu relativem Pfad (ab models/)
    try:
        rel_path = file_path.relative_to(MODELS_DIR)
    except ValueError:
        print(f"[WARNING] Datei nicht in models/: {file_path}")
        return

    # Lies bestehende Einträge
    existing = set()
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
            existing = {line.strip() for line in f if line.strip()}

    # Bereits in Queue?
    rel_path_str = str(rel_path).replace('\\', '/')
    if rel_path_str in existing:
        return  # Schon drin

    # Hinzufügen
    with open(QUEUE_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{rel_path_str}\n")


def remove_misplaced_file(file_path):
    """Entfernt Datei aus Queue (wurde gerettet)

    Args:
        file_path (Path/str): Absoluter Pfad zur Datei
    """
    if not QUEUE_FILE.exists():
        return

    file_path = Path(file_path)

    # Konvertiere zu relativem Pfad
    try:
        rel_path = file_path.relative_to(MODELS_DIR)
        rel_path_str = str(rel_path).replace('\\', '/')
    except ValueError:
        return

    # Lies alle Zeilen
    with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Filtere diese Zeile raus
    filtered = [line for line in lines if line.strip() != rel_path_str]

    # Schreibe zurück
    with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
        f.writelines(filtered)


# ============================================================================
# USER INPUT FUNCTIONS (Zentral für alle Module)
# ============================================================================

def ask_keep_or_delete(total_size_gb=None):
    """[TEST VERSION] Returns False (no execution anyway)"""
    return False


def ask_confirm_installation():
    """[TEST VERSION] Shows info and waits for user"""
    print()
    print("=" * 60)
    print("  TEST VERSION - Preview Only")
    print()
    print("  The analysis above shows what WOULD be installed.")
    print("  Full functionality available in Pro Version.")
    print()
    print("  Email: kodaparallax@gmail.com")
    print("=" * 60)
    print()
    input("Press ENTER to return to menu...")
    return False


def ask_confirm_fixes():
    """[TEST VERSION] Shows info and waits for user"""
    print()
    print("=" * 60)
    print("  TEST VERSION - Preview Only")
    print()
    print("  The analysis above shows what WOULD be fixed.")
    print("  Full functionality available in Pro Version.")
    print()
    print("  Email: kodaparallax@gmail.com")
    print("=" * 60)
    print()
    input("Press ENTER to return to menu...")
    return False


# ============================================================================
# OUTPUT HELPER FUNCTIONS (Mode A - Einheitliches Format)
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    WHITE = '\033[97m'
    RESET = '\033[0m'


def print_mode_a_header(module_name_caps, downloads_path, extensions, module_type, target_folders):
    """Druckt Mode A Header + Scanning Info

    Args:
        module_name_caps: z.B. "STABLE DIFFUSION MODELS (INSTALLATION)"
        downloads_path: Path zum downloads Ordner
        extensions: z.B. "*.safetensors, *.gguf"
        module_type: z.B. "Base Models"
        target_folders: z.B. "checkpoints/, unet/"
    """
    print("=" * 80)
    print(module_name_caps)
    print("=" * 80)
    print()
    print(f"Scanning: {downloads_path}")
    print(f"Looking for: {extensions} ({module_type})")
    print(f"Target folders: {target_folders}")
    print("-" * 80)
    print()


def print_mode_b_header(module_name, folders, extensions, module_type, target_folders, preview_mode=False):
    """Druckt Mode B Header + Scanning Info

    Args:
        module_name: z.B. "STABLE DIFFUSION MODELS"
        folders: z.B. "controlnet/, t2i_adapter/"
        extensions: z.B. "*.safetensors"
        module_type: z.B. "ControlNet/T2I-Adapter"
        target_folders: z.B. "controlnet/, t2i_adapter/"
        preview_mode: True = PREVIEW, False = execute
    """
    mode_suffix = "CHECK/FIX - PREVIEW" if preview_mode else "CHECK/FIX"
    print("=" * 80)
    print(f"{module_name} ({mode_suffix})")
    print("=" * 80)
    print()
    print(f"Scanning: {folders}")
    print(f"Looking for: {extensions} ({module_type})")
    print(f"Target folders: {target_folders}")
    print("-" * 80)
    print()


def print_no_files_found(file_type="files"):
    """Druckt Info wenn keine Dateien gefunden"""
    print(f"[INFO] No {file_type} found in downloads folder.")


def print_analysis(found, to_install, skipped):
    """Druckt ANALYSIS Sektion für Mode A (mit Farben)

    Args:
        found: Anzahl gefundener Dateien
        to_install: Anzahl zu installierender Dateien (GELB komplett)
        skipped: Anzahl übersprungener Dateien (keine Farbe)
    """
    print("ANALYSIS:")
    print()
    print(f"Found {found} files")
    print(f"{Colors.YELLOW}To install: {to_install}{Colors.RESET}")
    print(f"Skipped: {skipped}")
    print()


def print_analysis_b(found, already_correct, problems):
    """Druckt ANALYSIS Sektion für Mode B (mit Farben)

    Args:
        found: Anzahl gefundener Dateien
        already_correct: Anzahl korrekter Dateien (GRÜN komplett)
        problems: Anzahl problematischer Dateien (GELB komplett)
    """
    print("ANALYSIS:")
    print()
    print(f"Found {found} files")
    print(f"{Colors.GREEN}Already correct: {already_correct}{Colors.RESET}")
    print(f"{Colors.YELLOW}Problems: {problems}{Colors.RESET}")
    print()


def print_skipped_section(skipped_files, skip_reason, max_show=5):
    """Druckt SKIPPED Sektion

    Args:
        skipped_files: Liste von dicts mit 'filename' key
        skip_reason: Allgemeiner Grund (z.B. "Not a base model...")
        max_show: Max Anzahl anzuzeigender Dateien (default 5)
    """
    if not skipped_files:
        return

    print("-" * 80)
    print(f"SKIPPED ({len(skipped_files)} files)")
    print("-" * 80)
    print(f"  Reason: {skip_reason}")
    print()

    # Show max files
    for item in skipped_files[:max_show]:
        filename = item['filename'] if isinstance(item, dict) else item
        print(f"    {filename}")

    if len(skipped_files) > max_show:
        print(f"    ... (+{len(skipped_files) - max_show} more)")
    print()


def print_preview_header(count):
    """Druckt PREVIEW Header mit Anzahl"""
    print("-" * 80)
    print(f"PREVIEW ({count} files to install)")
    print("-" * 80)
    print()


def print_preview_item(index, filename, size_mb, detected_info, target_path, extra_line=None):
    """Druckt einzelnes Preview Item (3-4 Zeilen Format)

    Args:
        index: Laufende Nummer (1-based)
        filename: Dateiname
        size_mb: Größe in MB
        detected_info: z.B. "SDXL, NoVAE, FP16"
        target_path: z.B. "checkpoints/SDXL_NoVAE-FP16_..."
        extra_line: Optional, z.B. "Trigger: xyz" für LoRAs
    """
    print(f"  {Colors.YELLOW}[PREVIEW/{index}]{Colors.RESET} {filename} ({size_mb:.0f} MB)")
    print(f"       Detected: {detected_info}")
    if extra_line:
        print(f"       {extra_line}")
    print(f"       -> {target_path}")
    print()


def print_total(count, size_gb):
    """Druckt TOTAL Zeile"""
    print("=" * 80)
    print(f"TOTAL: {count} files ({size_gb:.1f} GB)")
    print("=" * 80)
    print()


def print_no_files_to_install():
    """Druckt Info wenn nichts zu installieren"""
    print("[INFO] No files to install.")


def print_installation_header():
    """Druckt INSTALLATION IN PROGRESS Header"""
    print()
    print("=" * 80)
    print("INSTALLATION IN PROGRESS...")
    print("=" * 80)
    print()


def print_install_item(index, total, filename, success, message):
    """Druckt einzelnes Installation Item

    Args:
        index: Laufende Nummer (1-based)
        total: Gesamtanzahl
        filename: Dateiname
        success: True/False
        message: Status-Message
    """
    print(f"[{index}/{total}] {filename}")
    if success:
        print(f"       {Colors.GREEN}✓{Colors.RESET} OK {message}")
    else:
        print(f"       {Colors.RED}✗{Colors.RESET} ERROR {message}")
    print()


def print_summary(processed, installed, collisions, errors, keep_source):
    """Druckt finale SUMMARY

    Args:
        processed: Anzahl verarbeiteter Dateien
        installed: Anzahl installierter Dateien
        collisions: Anzahl Name-Kollisionen (auto-suffixed)
        errors: Anzahl Fehler
        keep_source: True = Originale behalten, False = gelöscht
    """
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print()
    print(f"  Processed: {processed} file{'s' if processed != 1 else ''}")
    print(f"  Installed: {installed}")
    print(f"  Name collisions (auto-suffixed): {collisions}")
    print(f"  Errors: {errors}")
    print()
    if not keep_source:
        print(f"  Deleted from downloads: {installed}")
    else:
        print(f"  Originals kept in downloads/")


# ============================================================================
# OUTPUT HELPER FUNCTIONS (Mode B - Check/Fix)
# ============================================================================

def print_already_correct_section(correct_files):
    """Druckt ALREADY CORRECT FILES Sektion

    Args:
        correct_files: Liste von tuples (folder, filename, size_mb)
    """
    if not correct_files:
        return

    print()
    print("ALREADY CORRECT FILES")
    print("-" * 80)
    print()

    for folder, filename, size_mb in correct_files:
        print(f"  {Colors.GREEN}[OK]{Colors.RESET} {folder}/{filename} ({size_mb:.1f} MB)")
    print()


def print_problems_header():
    """Druckt PROBLEMS FOUND Header"""
    print()
    print("PROBLEMS FOUND")
    print("-" * 80)
    print()
    print("Processing:")
    print()


def print_problem_item(index, total, filename, size_mb, detected_info, target_path, warning):
    """Druckt einzelnes Problem Item

    Args:
        index: Laufende Nummer (1-based)
        total: Gesamtanzahl
        filename: Dateiname
        size_mb: Größe in MB
        detected_info: z.B. "ControlNet, SDXL, Unknown, FP16" (kann None sein)
        target_path: z.B. "controlnet/SDXL_CN-Unknown_FP16_v1.safetensors" (kann None sein)
        warning: z.B. "Non-standard name detected"
    """
    print(f"  {Colors.YELLOW}[{index}/{total}]{Colors.RESET} {filename} ({size_mb:.0f} MB)")

    if detected_info:
        print(f"       Detected: {detected_info}")

    if target_path:
        print(f"       -> {target_path}")

    print(f"       WARNING {warning}")
    print()


def print_fix_result(success, message):
    """Druckt Ergebnis eines Fixes

    Args:
        success: True/False
        message: z.B. "Renamed to standard format"
    """
    if success:
        print(f"         {Colors.GREEN}OK{Colors.RESET} {message}")
    else:
        print(f"         {Colors.RED}ERROR{Colors.RESET} {message}")
