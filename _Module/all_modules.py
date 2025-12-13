#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALL MODULES - Batch Installation/Check
Processes all 15 module types in one go

TensorSort Model Installer - Test Version
"""

import sys
import os
from pathlib import Path

# ============================================================================
# PATH SETUP - Für Unterordner-Struktur
# ============================================================================
_SCRIPT_DIR = Path(__file__).parent
_SHARED_DIR = _SCRIPT_DIR.parent / "_shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

# Add _Module dir itself for module imports
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ============================================================================
# ANSI COLORS
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

# ============================================================================
# MODE A - BATCH INSTALLATION
# ============================================================================

def mode_a_batch():
    """Mode A: Install all file types - PROPER batch processing"""

    print("=" * 80)
    print("BATCH INSTALLATION - ALL FILE TYPES (PREVIEW)")
    print("=" * 80)
    print()
    print("Scanning: downloads/")
    print()
    print("Scanning all modules.....")

    # Import modules
    import modul1_mainmodels
    import modul2_vae
    import modul3_clip
    import modul4_loras
    import modul5_controlnet
    import modul6_upscalers
    import modul7_embeddings
    import modul8_photomaker
    import modul9_insightface
    import modul10_ipadapter
    import modul11_animatediff
    import modul12_sam
    import modul13_groundingdino
    import modul14_yolo
    import modul15_vlm_llm

    from shared_utils import DOWNLOADS_DIR

    # ========================================================================
    # PHASE 1: SCAN ALL MODULES (collect results)
    # ========================================================================
    all_results = []

    # Module 1 - Stable Diffusion Models
    result1 = modul1_mainmodels.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((1, result1, modul1_mainmodels))

    # Module 2 - VAE
    result2 = modul2_vae.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((2, result2, modul2_vae))

    # Module 3 - CLIP & Encoders
    result3 = modul3_clip.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((3, result3, modul3_clip))

    # Module 4 - LoRAs
    result4 = modul4_loras.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((4, result4, modul4_loras))

    # Module 5 - ControlNet & T2I-Adapter
    result5 = modul5_controlnet.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((5, result5, modul5_controlnet))

    # Module 6: Upscalers
    result6 = modul6_upscalers.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((6, result6, modul6_upscalers))

    # Module 7 - Embeddings
    result7 = modul7_embeddings.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((7, result7, modul7_embeddings))

    # Module 8 - PhotoMaker
    result8 = modul8_photomaker.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((8, result8, modul8_photomaker))

    # Module 9 - InsightFace
    result9 = modul9_insightface.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((9, result9, modul9_insightface))

    # Module 10 - IP-Adapter
    result10 = modul10_ipadapter.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((10, result10, modul10_ipadapter))

    # Module 11 - AnimateDiff
    result11 = modul11_animatediff.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((11, result11, modul11_animatediff))

    # Module 12 - SAM
    result12 = modul12_sam.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((12, result12, modul12_sam))

    # Module 13 - Grounding DINO
    result13 = modul13_groundingdino.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((13, result13, modul13_groundingdino))

    # Module 14 - YOLO
    result14 = modul14_yolo.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((14, result14, modul14_yolo))

    # Module 15 - VLM & LLM
    result15 = modul15_vlm_llm.scan_for_batch(DOWNLOADS_DIR)
    all_results.append((15, result15, modul15_vlm_llm))

    print(" Done!")
    print()

    # ========================================================================
    # PHASE 2: SHOW GROUPED PREVIEW
    # ========================================================================
    total_files = 0
    total_size = 0

    for mod_num, result, module in all_results:
        if not result['files']:
            continue

        # Calculate module stats
        mod_files = len(result['files'])
        mod_size = sum(f['size_gb'] for f in result['files'])
        total_files += mod_files
        total_size += mod_size

        # Module header
        print("-" * 80)
        print(f"{result['module_name'].upper()} ({mod_files} files, {mod_size:.1f} GB)")
        print("-" * 80)
        print()

        # Show files (NEW 3-LINE FORMAT)
        for idx, f in enumerate(result['files'], 1):
            size_mb = f['size_gb'] * 1024
            # Line 1: [PREVIEW/X] filename (SIZE MB)
            # Get filename safely (some modules use 'filename', others use file_path.name)
            if 'filename' in f:
                filename = f['filename']
            elif 'file_path' in f:
                filename = Path(f['file_path']).name
            elif 'path' in f:
                filename = Path(f['path']).name
            else:
                filename = 'unknown'
            print(f"  {Colors.YELLOW}[PREVIEW/{idx}]{Colors.RESET} {filename} ({size_mb:.0f} MB)")

            # Line 2: Detected: (Module-specific)
            # IMPORTANT: Check most specific patterns first!

            # Module 2 (VAE Approx) - Check 'vae_type' field (TAESD or Custom)
            if 'vae_type' in f:
                vae_type = f['vae_type']
                r = f['result']
                folder_name = f['target_folder'].name if hasattr(f['target_folder'], 'name') else 'vae_approx'

                if vae_type == "TAESD":
                    # Official TAESD: show base_model and type (Encoder/Decoder)
                    print(f"       Detected: TAESD (Official), {r['base_model']}, {r['type']}")
                else:
                    # Custom VAE Approx: show base_model only
                    print(f"       Detected: VAE Approx (Custom), {r['base_model']}")

                print(f"       -> {folder_name}/{r['proper_name']}")

            # Module 15 (AnimateDiff) has 'type' in top-level (MotionModule or MotionLoRA)
            if 'type' in f and f['type'] in ['MotionModule', 'MotionLoRA']:
                r = f['result']
                folder_name = f['target_folder'].name
                proper_name = f['proper_name']  # proper_name is at top-level, not in result

                if f['type'] == 'MotionModule':
                    variant_str = f", {r['variant']}" if r.get('variant') else ""
                    print(f"       Detected: Motion Module, {r['base_model']}, {r['version']}{variant_str}")
                else:
                    variant_str = f", {r['variant']}" if r.get('variant') else ""
                    print(f"       Detected: Motion LoRA, {r['base_model']}, {r['version']}, {r['motion_type']}{variant_str}")

                print(f"       -> {folder_name}/{proper_name}")

            # Module 12 (IP-Adapter) has 'variant' AND 'encoder' (must check before Module 1!)
            elif 'result' in f and 'encoder' in f['result']:
                r = f['result']
                folder_name = r['target_folder'].name if hasattr(r['target_folder'], 'name') else 'ipadapter'
                print(f"       Detected: {r.get('base_model', 'Unknown')}, {r['variant']}, {r['encoder']}")
                print(f"       -> {folder_name}/{r['proper_name']}")
            # Module 1 (SD Models) has 'result' with variant and precision
            elif 'result' in f and 'variant' in f['result'] and 'precision' in f['result']:
                r = f['result']
                print(f"       Detected: {r['base_model']}, {r['variant']}, {r['precision']}")
                print(f"       -> {r['target_folder']}/{r['proper_name']}")
            # Module 4 (LoRAs) has 'result' but different structure
            # Module 4 (LoRAs) & 5 (ControlNet) have 'result' but different structures
            elif 'result' in f:
                r = f['result']
                # Get folder name safely
                if 'target_folder' in r:
                    folder_name = r['target_folder'].name if hasattr(r['target_folder'], 'name') else str(r['target_folder'])
                else:
                    folder_name = 'target'

                # Module 4 (LoRAs) has 'type' and 'category'
                if 'type' in r and 'category' in r:
                    print(f"       Detected: {r['type']}, Category: {r['category']}")
                # Module 5 (ControlNet) has 'control_type' and 'base_model'
                # Module 6 (Upscalers) has 'scale' and 'types'
                elif 'scale' in r and 'types' in r:
                    types_str = '-'.join(r['types']) if r['types'] else 'General'
                    arch_str = r.get('arch', 'Unknown')
                    print(f"       Detected: {r['scale']}, {types_str}, {arch_str}")
                elif 'control_type' in r:
                    print(f"       Detected: {r['control_type']}, {r.get('base_model', 'Unknown')}, {r.get('precision', 'Unknown')}")
                # Module 22 (YOLO) has 'version', 'size', 'output_type', 'specialization'
                # Module 10 (PhotoMaker) has 'version' only (unique - simpler than YOLO)
                elif 'version' in r and 'proper_name' in r and r['proper_name'].startswith('SDXL_PhotoMaker'):
                    print(f"       Detected: PhotoMaker {r['version']} (SDXL-only)")
                elif 'version' in r and 'output_type' in r and 'specialization' in r:
                    print(f"       Detected: YOLO{r['version']}{r.get('size', '?')}, {r['output_type']}, {r['specialization']}")
                # Fallback
                else:
                    print(f"       Detected: (module-specific)")
                print(f"       -> {folder_name}/{r['proper_name']}")
            elif 'new_name' in f:
                folder_name = f['target_folder'].name if hasattr(f['target_folder'], 'name') else 'target'
                # Try to get detected info if available
                if 'base_model' in f:
                    print(f"       Detected: {f['base_model']}, {f.get('precision', 'Unknown')}")
                print(f"       -> {folder_name}/{f['new_name']}")
                # Special note for Qwen3 (dual-use as LLM)
                if 'qwen3' in filename.lower() or 'qwen_3' in filename.lower():
                    print(f"       {Colors.YELLOW}NOTE: This file can also be used as LLM for chat nodes.{Colors.RESET}")
                    print(f"       {Colors.YELLOW}      To use it there, copy to LLM/ or create a symlink.{Colors.RESET}")
            else:
                print(f"       -> (installation target)")

            print()

    # Total summary
    print("=" * 80)
    print(f"TOTAL: {total_files} files ({total_size:.1f} GB)")
    print("=" * 80)
    print()

    if total_files == 0:
        print("No files to install!")
        return

    # ========================================================================
    # PHASE 3: ASK DELETE/KEEP (ONE question for all!)
    # ========================================================================
    from shared_utils import ask_keep_or_delete, ask_confirm_installation

    keep_source = ask_keep_or_delete(total_size)
    if keep_source is None:
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Installation cancelled.")
        sys.exit(1)  # Exit code 1 = Cancelled (no pause in batch script)

    # ========================================================================
    # PHASE 4: CONFIRMATION (ONE for all!)
    # ========================================================================
    if not ask_confirm_installation():
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Installation cancelled.")
        sys.exit(1)  # Exit code 1 = Cancelled (no pause in batch script)

    # ========================================================================
    # PHASE 5: INSTALL ALL MODULES
    # ========================================================================
    print("\n" + "=" * 80)
    print("INSTALLATION IN PROGRESS...")
    print("=" * 80)
    print()

    total_installed = 0
    total_errors = 0

    mod_count = sum(1 for _, r, _ in all_results if r['files'])
    current_mod = 0

    for mod_num, result, module in all_results:
        if not result['files']:
            continue

        current_mod += 1
        print(f"{Colors.YELLOW}[{current_mod}/{mod_count}] {result['module_name']}...{Colors.RESET}")

        stats = module.install_for_batch(result['files'], keep_source)

        total_installed += stats['installed']
        total_errors += stats['errors']

        print(f"       {Colors.GREEN}[OK]{Colors.RESET} Installed: {stats['installed']}, Errors: {stats['errors']}")
        print()

    # Final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total installed: {total_installed}")
    print(f"  Total errors: {total_errors}")
    print()
    if not keep_source:
        print(f"  Deleted from downloads: {total_installed} files ({total_size:.1f} GB freed)")
    else:
        print(f"  Originals kept in downloads/")
    print()
    print(Colors.GREEN + "[COMPLETED] BATCH INSTALLATION" + Colors.RESET)

# ============================================================================
# MODE B - BATCH CHECK/FIX
# ============================================================================

def mode_b_batch():
    """Mode B: Check/Fix all file types with 2-pass system"""

    print("=" * 80)
    print("BATCH CHECK/FIX - ALL FILE TYPES (2-Pass System)")
    print("=" * 80)
    print()
    print("This will check all installed files in two passes:")
    print("  PASS 1: Scan all modules, build registry of misplaced files")
    print("  PASS 2: Analyze problems and show what needs fixing")
    print()
    print("=" * 80)
    print()

    # Import modules
    import modul1_mainmodels
    import modul2_vae
    import modul3_clip
    import modul4_loras
    import modul5_controlnet
    import modul6_upscalers
    import modul7_embeddings
    import modul8_photomaker
    import modul9_insightface
    import modul10_ipadapter
    import modul11_animatediff
    import modul12_sam
    import modul13_groundingdino
    import modul14_yolo
    import modul15_vlm_llm
    import subprocess
    import sys

    # ========================================================================
    # PASS 1: SCAN-ONLY (Build Registry)
    # ========================================================================
    print("=" * 80)
    print("PASS 1: BUILDING REGISTRY (Scan-Only)")
    print("=" * 80)
    print()

    # Run all modules with --scan-only flag
    # This builds the misplaced_files_queue.txt registry
    modules = [
        ("modul1_mainmodels.py", "Stable Diffusion Models"),
        ("modul2_vae.py", "VAE"),
        ("modul3_clip.py", "Text Encoders"),
        ("modul4_loras.py", "LoRAs & LyCORIS"),
        ("modul5_controlnet.py", "ControlNet & T2I-Adapter"),
        ("modul6_upscalers.py", "Upscalers"),
        ("modul7_embeddings.py", "Embeddings"),
        ("modul8_photomaker.py", "PhotoMaker"),
        ("modul9_insightface.py", "InsightFace"),
        ("modul10_ipadapter.py", "IP-Adapter"),
        ("modul11_animatediff.py", "AnimateDiff"),
        ("modul12_sam.py", "SAM"),
        ("modul13_groundingdino.py", "Grounding DINO"),
        ("modul14_yolo.py", "YOLO"),
        ("modul15_vlm_llm.py", "VLM & LLM")
    ]

    for idx, (module_file, module_name) in enumerate(modules, 1):
        print(f"[{idx}/15] Scanning {module_name}...")
        module_path = _SCRIPT_DIR / module_file
        result = subprocess.run(
            [sys.executable, str(module_path), "B", "--scan-only"],
            capture_output=True,
            text=True
        )
        # Scan-only is silent, just builds registry

    print()
    print(Colors.GREEN + "[OK]" + Colors.RESET + " Registry built")

    # ========================================================================
    # PASS 2: ANALYZE PROBLEMS (PREVIEW - show what needs fixing)
    # ========================================================================
    print()
    print("=" * 80)
    print("PASS 2: ANALYZING PROBLEMS...")
    print("=" * 80)
    print()

    # Run all modules in preview mode (shows problems without fixing)
    for idx, (module_file, module_name) in enumerate(modules, 1):
        print(f"[{idx}/15] Analyzing {module_name}...")
        module_path = _SCRIPT_DIR / module_file
        result = subprocess.run(
            [sys.executable, str(module_path), "B", "--preview"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    # ========================================================================
    # ASK USER - FIX NOW?
    # ========================================================================
    from shared_utils import ask_confirm_fixes

    print()
    print("=" * 80)
    print("All problems have been analyzed (PREVIEW).")
    print()

    if not ask_confirm_fixes():
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Operation cancelled. No changes made.")
        sys.exit(1)  # Exit code 1 = Cancelled (no pause in batch script)

    # ========================================================================
    # PASS 2: EXECUTE FIXES (with --batch flag to skip module prompts)
    # ========================================================================
    print()
    print("=" * 80)
    print("EXECUTING FIXES...")
    print("=" * 80)
    print()

    # Run all modules with --batch flag (skips individual confirmations)
    for idx, (module_file, module_name) in enumerate(modules, 1):
        print(f"[{idx}/15] Fixing {module_name}...")
        module_path = _SCRIPT_DIR / module_file
        result = subprocess.run(
            [sys.executable, str(module_path), "B", "--batch"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    print()
    print("=" * 80)
    print(Colors.GREEN + "[COMPLETED] BATCH CHECK/FIX" + Colors.RESET)
    print("=" * 80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python all_modules.py <A|B>")
        print("  A = Mode A (Installation)")
        print("  B = Mode B (Reinstall/Check)")
        sys.exit(1)

    mode = sys.argv[1].upper()

    if mode == "A":
        mode_a_batch()
    elif mode == "B":
        mode_b_batch()
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'A' for Installation or 'B' for Reinstall/Check")
        sys.exit(1)

if __name__ == "__main__":
    main()
