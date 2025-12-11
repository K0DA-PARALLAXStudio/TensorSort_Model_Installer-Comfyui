@echo off
title ComfyUI Model Installer - Main Menu
color 0F
cd /d "%~dp0"

REM ============================================================================
REM STARTUP: Check paths before showing menu
REM ============================================================================
python -c "import sys; sys.path.insert(0, '_shared'); from shared_utils import COMFYUI_BASE, DOWNLOADS_DIR"
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed or cancelled. Press any key to exit.
    pause >nul
    exit
)

:main_menu
cls
echo.
echo    ================================================================
echo    COMFYUI-TENSORSORT MODEL INSTALLER
echo    ================================================================
echo.
echo    Intelligent model organizer for ComfyUI
echo.
echo      * Analyzes files by internal structure (not just filename)
echo      * Sorts into correct folders automatically
echo      * Renames for consistency and easy search
echo      * Supports 15+ model types (SD/Flux/VAE/LoRA/ControlNet/...)
echo.
echo    Author: K0DA Parallax Studio
echo    Email: kodaparallax@gmail.com
echo.
echo    ================================================================
echo.
echo    [1] INSTALL NEW MODELS - Process downloads/ folder
echo    [2] CHECK ^& FIX MODELS - Verify existing installation
echo    [3] HELP ^& GUIDES - First time? Start here!
echo    [4] SETTINGS - Change ComfyUI/downloads paths
echo    [0] EXIT
echo.
echo    ================================================================
echo.

set /p main_choice="Your choice: "

if "%main_choice%"=="1" goto mode_a_menu
if "%main_choice%"=="2" goto mode_b_menu
if "%main_choice%"=="3" goto docs
if "%main_choice%"=="4" goto settings
if "%main_choice%"=="0" goto end

echo Invalid choice!
timeout /t 2 >nul
goto main_menu

:mode_a_menu
cls
echo ===============================================================================
echo   MODE A - INSTALLATION
echo ===============================================================================
echo.
echo   [1] BATCH PROCESSING (All Types) - RECOMMENDED
echo       Processes all 15 module types sequentially
echo.
echo   [2] INDIVIDUAL FILE TYPES (Advanced)
echo       Select one specific module type
echo.
echo   [0] BACK
echo ===============================================================================
echo.

set /p mode_a_choice="Your choice [1/2/0]: "

if "%mode_a_choice%"=="1" goto mode_a_all
if "%mode_a_choice%"=="2" goto mode_a_individual
if "%mode_a_choice%"=="0" goto main_menu

echo Invalid choice!
timeout /t 2 >nul
goto mode_a_menu

:mode_a_individual
cls
echo ===============================================================================
echo   MODE A - INDIVIDUAL FILE TYPES
echo ===============================================================================
echo.
echo   [1] Stable Diffusion Models (Checkpoints/UNET/GGUF)
echo   [2] VAE (Variational AutoEncoder)
echo   [3] Text Encoders (CLIP/T5/BERT)
echo   [4] LoRAs ^& LyCORIS
echo   [5] ControlNet ^& T2I-Adapter
echo   [6] Upscalers
echo   [7] Embeddings (Textual Inversion)
echo   [8] PhotoMaker
echo   [9] InsightFace
echo   [10] IP-Adapter
echo   [11] AnimateDiff
echo   [12] SAM (Segment Anything)
echo   [13] Grounding DINO
echo   [14] YOLO
echo   [15] VLM ^& LLM
echo.
echo   [0] BACK
echo ===============================================================================
echo.

set /p module_choice="Your choice [1-15, 0]: "

if "%module_choice%"=="1" goto mode_a_module1
if "%module_choice%"=="2" goto mode_a_module2
if "%module_choice%"=="3" goto mode_a_module3
if "%module_choice%"=="4" goto mode_a_module4
if "%module_choice%"=="5" goto mode_a_module5
if "%module_choice%"=="6" goto mode_a_module6
if "%module_choice%"=="7" goto mode_a_module7
if "%module_choice%"=="8" goto mode_a_module8
if "%module_choice%"=="9" goto mode_a_module9
if "%module_choice%"=="10" goto mode_a_module10
if "%module_choice%"=="11" goto mode_a_module11
if "%module_choice%"=="12" goto mode_a_module12
if "%module_choice%"=="13" goto mode_a_module13
if "%module_choice%"=="14" goto mode_a_module14
if "%module_choice%"=="15" goto mode_a_module15
if "%module_choice%"=="0" goto mode_a_menu

echo Invalid choice!
timeout /t 2 >nul
goto mode_a_individual

REM ===================================================================
REM MODE B - REINSTALL/CHECK
REM ===================================================================

:mode_b_menu
cls
echo ===============================================================================
echo   MODE B - REINSTALL/CHECK
echo ===============================================================================
echo.
echo   [1] BATCH CHECK/FIX (All Types) - RECOMMENDED
echo       Checks all 15 module types sequentially
echo.
echo   [2] INDIVIDUAL FILE TYPES (Advanced)
echo       Check one specific module type
echo.
echo   [0] BACK
echo ===============================================================================
echo.

set /p mode_b_choice="Your choice [1/2/0]: "

if "%mode_b_choice%"=="1" goto mode_b_all
if "%mode_b_choice%"=="2" goto mode_b_individual
if "%mode_b_choice%"=="0" goto main_menu

echo Invalid choice!
timeout /t 2 >nul
goto mode_b_menu

:mode_b_individual
cls
echo ===============================================================================
echo   MODE B - INDIVIDUAL FILE TYPES
echo ===============================================================================
echo.
echo   [1] Stable Diffusion Models (Checkpoints/UNET/GGUF)
echo   [2] VAE (Variational AutoEncoder)
echo   [3] Text Encoders (CLIP/T5/BERT)
echo   [4] LoRAs ^& LyCORIS
echo   [5] ControlNet ^& T2I-Adapter
echo   [6] Upscalers
echo   [7] Embeddings (Textual Inversion)
echo   [8] PhotoMaker
echo   [9] InsightFace
echo   [10] IP-Adapter
echo   [11] AnimateDiff
echo   [12] SAM (Segment Anything)
echo   [13] Grounding DINO
echo   [14] YOLO
echo   [15] VLM ^& LLM
echo.
echo   [0] BACK
echo ===============================================================================
echo.

set /p module_choice="Your choice [1-15, 0]: "

if "%module_choice%"=="1" goto mode_b_module1
if "%module_choice%"=="2" goto mode_b_module2
if "%module_choice%"=="3" goto mode_b_module3
if "%module_choice%"=="4" goto mode_b_module4
if "%module_choice%"=="5" goto mode_b_module5
if "%module_choice%"=="6" goto mode_b_module6
if "%module_choice%"=="7" goto mode_b_module7
if "%module_choice%"=="8" goto mode_b_module8
if "%module_choice%"=="9" goto mode_b_module9
if "%module_choice%"=="10" goto mode_b_module10
if "%module_choice%"=="11" goto mode_b_module11
if "%module_choice%"=="12" goto mode_b_module12
if "%module_choice%"=="13" goto mode_b_module13
if "%module_choice%"=="14" goto mode_b_module14
if "%module_choice%"=="15" goto mode_b_module15
if "%module_choice%"=="0" goto mode_b_menu

echo Invalid choice!
timeout /t 2 >nul
goto mode_b_individual

REM ===================================================================
REM MODE A - MODULE 1 (Base Models)
REM ===================================================================

:mode_a_module1
cls
cd /d "%~dp0"
python _Module\modul1_mainmodels.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 2 (VAE)
REM ===================================================================

:mode_a_module2
cls
cd /d "%~dp0"
python _Module\modul2_vae.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 3 (CLIP & Text Encoders)
REM ===================================================================

:mode_a_module3
cls
cd /d "%~dp0"
python _Module\modul3_clip.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 4 (LoRAs & LyCORIS)
REM ===================================================================

:mode_a_module4
cls
cd /d "%~dp0"
python _Module\modul4_loras.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 5 (ControlNet & T2I-Adapter)
REM ===================================================================

:mode_a_module5
cls
cd /d "%~dp0"
python _Module\modul5_controlnet.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 6 (Upscalers)
REM ===================================================================

:mode_a_module6
cls
cd /d "%~dp0"
python _Module\modul6_upscalers.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 7 (EMBEDDINGS)
REM ===================================================================

:mode_a_module7
cls
cd /d "%~dp0"
python _Module\modul7_embeddings.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 8 (PHOTOMAKER)
REM ===================================================================

:mode_a_module8
cls
cd /d "%~dp0"
python _Module\modul8_photomaker.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 9 (INSIGHTFACE)
REM ===================================================================

:mode_a_module9
cls
cd /d "%~dp0"
python _Module\modul9_insightface.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 10 (IP-ADAPTER)
REM ===================================================================

:mode_a_module10
cls
cd /d "%~dp0"
python _Module\modul10_ipadapter.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 11 (ANIMATEDIFF)
REM ===================================================================

:mode_a_module11
cls
cd /d "%~dp0"
python _Module\modul11_animatediff.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 12 (SAM)
REM ===================================================================

:mode_a_module12
cls
cd /d "%~dp0"
python _Module\modul12_sam.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 13 (GROUNDING DINO)
REM ===================================================================

:mode_a_module13
cls
cd /d "%~dp0"
python _Module\modul13_groundingdino.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 14 (YOLO)
REM ===================================================================

:mode_a_module14
cls
cd /d "%~dp0"
python _Module\modul14_yolo.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - MODULE 15 (VLM & LLM)
REM ===================================================================

:mode_a_module15
cls
cd /d "%~dp0"
python _Module\modul15_vlm_llm.py A
if errorlevel 1 goto mode_a_individual
pause
goto mode_a_individual

REM ===================================================================
REM MODE A - ALL MODULES
REM ===================================================================

:mode_a_all
cls
cd /d "%~dp0"
python _Module\all_modules.py A
if errorlevel 1 goto mode_a_menu
pause
goto mode_a_menu

REM ===================================================================
REM MODE B - MODULE 1 (Base Models)
REM ===================================================================

:mode_b_module1
cls
cd /d "%~dp0"
python _Module\modul1_mainmodels.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul1_mainmodels.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 2 (VAE)
REM ===================================================================

:mode_b_module2
cls
cd /d "%~dp0"
python _Module\modul2_vae.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul2_vae.py B
    echo.
    pause
)
goto mode_b_individual


REM ===================================================================
REM MODE B - MODULE 3 (CLIP & Text Encoders)
REM ===================================================================

:mode_b_module3
cls
cd /d "%~dp0"
python _Module\modul3_clip.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul3_clip.py B
    echo.
    pause
)
goto mode_b_individual


REM ===================================================================
REM MODE B - MODULE 4 (LoRAs & LyCORIS)
REM ===================================================================

:mode_b_module4
cls
cd /d "%~dp0"
python _Module\modul4_loras.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul4_loras.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 5 (ControlNet & T2I-Adapter)
REM ===================================================================

:mode_b_module5
cls
cd /d "%~dp0"
python _Module\modul5_controlnet.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul5_controlnet.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 6 (Upscalers)
REM ===================================================================

:mode_b_module6
cls
cd /d "%~dp0"
python _Module\modul6_upscalers.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul6_upscalers.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 7 (EMBEDDINGS)
REM ===================================================================

:mode_b_module7
cls
cd /d "%~dp0"
python _Module\modul7_embeddings.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul7_embeddings.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 8 (PHOTOMAKER)
REM ===================================================================

:mode_b_module8
cls
cd /d "%~dp0"
python _Module\modul8_photomaker.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul8_photomaker.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 9 (INSIGHTFACE)
REM ===================================================================

:mode_b_module9
cls
cd /d "%~dp0"
python _Module\modul9_insightface.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul9_insightface.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 10 (IP-ADAPTER)
REM ===================================================================

:mode_b_module10
cls
cd /d "%~dp0"
python _Module\modul10_ipadapter.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul10_ipadapter.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 11 (ANIMATEDIFF)
REM ===================================================================

:mode_b_module11
cls
cd /d "%~dp0"
python _Module\modul11_animatediff.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul11_animatediff.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 12 (SAM)
REM ===================================================================

:mode_b_module12
cls
cd /d "%~dp0"
python _Module\modul12_sam.py B --preview
echo.
echo ===============================================================================
echo [1] Execute (no fixes needed - SAM keeps original names)
echo [0] Cancel
echo ===============================================================================
echo.
set /p execute="Your choice [1/0]: "
if "%execute%"=="1" (
    cls
    python _Module\modul12_sam.py B
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 13 (GROUNDING DINO)
REM ===================================================================

:mode_b_module13
cls
cd /d "%~dp0"
python _Module\modul13_groundingdino.py B --preview
echo.
echo ===============================================================================
echo [1] Execute (no fixes needed - Grounding DINO keeps original names)
echo [0] Cancel
echo ===============================================================================
echo.
set /p execute="Your choice [1/0]: "
if "%execute%"=="1" (
    cls
    python _Module\modul13_groundingdino.py B
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 14 (YOLO)
REM ===================================================================

:mode_b_module14
cls
cd /d "%~dp0"
python _Module\modul14_yolo.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul14_yolo.py B
    echo.
    pause
)
goto mode_b_individual

REM ===================================================================
REM MODE B - MODULE 15 (VLM & LLM)
REM ===================================================================

:mode_b_module15
cls
cd /d "%~dp0"
python _Module\modul15_vlm_llm.py B --preview
echo.
echo ===============================================================================
echo [1] Execute fixes
echo [0] Cancel
echo ===============================================================================
echo.
set /p choice="Your choice [1/0]: "
if "%choice%"=="1" (
    echo.
    echo Executing fixes...
    python _Module\modul15_vlm_llm.py B
    echo.
    pause
)
goto mode_b_individual


REM ===================================================================
REM MODE B - ALL MODULES
REM ===================================================================

:mode_b_all
cls
cd /d "%~dp0"
python _Module\all_modules.py B
goto mode_b_menu


REM ===================================================================
REM DOCUMENTATION
REM ===================================================================

:docs
cls
echo ===============================================================================
echo   HELP ^& DOCUMENTATION
echo ===============================================================================
echo.
echo   QUICK START:
echo   ------------
echo   [1] README English - Quick guide
echo   [2] README Deutsch - Schnellanleitung
echo.
echo   FULL MANUAL:
echo   ------------
echo   [3] MANUAL English - Complete documentation
echo   [4] MANUAL Deutsch - Vollstaendige Dokumentation
echo.
echo   SUPPORT:
echo   --------
echo   Author: K0DA Parallax Studio
echo   Email: kodaparallax@gmail.com
echo.
echo   [0] BACK
echo ===============================================================================
echo.

set /p doc_choice="Your choice [1-4/0]: "

if "%doc_choice%"=="1" goto show_readme_en
if "%doc_choice%"=="2" goto show_readme_de
if "%doc_choice%"=="3" goto show_manual_en
if "%doc_choice%"=="4" goto show_manual_de
if "%doc_choice%"=="0" goto main_menu

echo Invalid choice!
timeout /t 2 >nul
goto docs

:show_readme_en
start notepad "%~dp0_Documentation\README_EN.txt"
goto docs

:show_readme_de
start notepad "%~dp0_Documentation\README_DE.txt"
goto docs

:show_manual_en
start notepad "%~dp0_Documentation\MANUAL_EN.txt"
goto docs

:show_manual_de
start notepad "%~dp0_Documentation\MANUAL_DE.txt"
goto docs

:settings
cls
cd /d "%~dp0"
python -c "import sys; sys.path.insert(0, '_shared'); from shared_utils import run_settings_menu; run_settings_menu()"
pause
goto main_menu

:end
cls
echo ===================================================================
echo   Goodbye!
echo ===================================================================
timeout /t 1 >nul
exit
