================================================================================
                    COMFYUI-TENSORSORT MODEL INSTALLER
                       Quick Start Guide (DE) - v1.1.0
================================================================================

                    "Weil Dateinamen lügen, aber Tensoren nicht."

Willkommen bei TensorSort! Diese Anleitung bringt dich in 10 Minuten zum
ersten erfolgreichen Durchlauf.


================================================================================
1. WAS IST TENSORSORT?
================================================================================

TensorSort ist ein intelligentes Sortier- und Benennungs-Tool für ComfyUI
Model-Dateien. Es analysiert den INHALT jeder Datei (die Tensor-Strukturen),
nicht nur den Dateinamen.

Was es erkennt:
  - Dateityp (Checkpoint, LoRA, VAE, ControlNet, ...)
  - Base Model (Flux, SDXL, SD 1.5, Pony, ...)
  - Precision (FP32, FP16, BF16, FP8, GGUF-Quantisierung)
  - Komponenten (Full, NoVAE, NoCLIP, UNET-only)

Was es macht:
  1. Sortiert die Datei in den richtigen ComfyUI-Ordner
  2. Benennt nach einheitlicher Konvention um
  3. Erkennt und handhabt Duplikate (per SHA256-Hash)

Das Ergebnis: Eine saubere, durchsuchbare Model-Bibliothek.


================================================================================
2. FÜR WEN IST DAS?
================================================================================

EINSTEIGER:
  - "Error loading LoRA" - und keine Ahnung warum?
  - TensorSort zeigt dir was jede Datei ist und wohin sie gehört
  - Du lernst nebenbei die verschiedenen Dateitypen kennen

SAMMLER:
  - Download-Ordner quillt über (50, 100, 200+ Dateien)?
  - Irgendwo sind Duplikate, aber mit verschiedenen Namen unsichtbar?
  - TensorSort bringt Ordnung, eliminiert Duplikate

PROFIS:
  - 400+ Modelle, historisch gewachsen, wild benannt?
  - Im LoRA-Loader ewig scrollen durch unsortierte Listen?
  - TensorSort gibt dir saubere Struktur, auf einen Blick erkennbar


================================================================================
3. QUICK START - DIE ERSTEN SCHRITTE
================================================================================

SCHRITT 1 - PROGRAMM STARTEN:

  Doppelklick auf: TensorSort_Model_Installer.bat

  → Hauptmenü erscheint

SCHRITT 2 - NEUE MODELLE INSTALLIEREN:

  1. Lege deine heruntergeladenen Dateien in den downloads/ Ordner

  2. Im Menü wähle:
     [1] INSTALL NEW MODELS - Process downloads/ folder

  3. Wähle:
     [1] BATCH PROCESSING (All Types) - RECOMMENDED

     (Dies verarbeitet alle 15 Modul-Typen nacheinander)

  4. TensorSort zeigt dir eine PREVIEW-Liste:
     - Welche Dateien erkannt wurden
     - Was jede Datei ist (Base Model, Typ, Precision)
     - Wohin sie verschoben wird
     - Wie der neue Name lautet

  5. Du entscheidest:
     [1] Keep originals in downloads/
     [2] Delete originals (spart Speicherplatz)
     [0] Cancel

  6. Finale Bestätigung:
     [1] Start installation
     [0] Cancel

  7. Installation läuft → Zusammenfassung wird angezeigt

FERTIG! Deine Modelle sind jetzt sortiert und korrekt benannt.

WICHTIG - NACH DER INSTALLATION:
  → In ComfyUI: Ctrl + Shift + R drücken (Hard Refresh)
  → Sonst zeigt ComfyUI noch die alten Dateinamen an (Browser-Cache!)


================================================================================
4. DIE ZWEI MODI
================================================================================

MODUS A - INSTALLATION (Neue Dateien):

  Du hast neue Dateien runtergeladen und willst sie installieren.

  Workflow:
    1. Dateien in downloads/ legen
    2. TensorSort → Modus A → Batch Processing
    3. Preview prüfen → Bestätigen → Fertig

MODUS B - CHECK & FIX (Bestehende Dateien):

  Dein models/ Ordner ist schon voll, aber unorganisiert.

  Workflow:
    1. TensorSort → Modus B
    2. Scannt alle bestehenden Dateien
    3. Findet: Falsche Ordner, falsche Namen, Duplikate
    4. Zeigt was geändert werden würde
    5. Du bestätigst → Probleme werden behoben


================================================================================
5. DIE 15 MODULE (ÜBERSICHT)
================================================================================

Jedes Modul verarbeitet einen bestimmten Dateityp:

 Nr | Modul               | ComfyUI Ordner                    | Was es macht
----|---------------------|-----------------------------------|------------------
  1 | Base Models         | checkpoints/, unet/               | SD Checkpoints, GGUF
  2 | VAE                 | vae/                              | VAE, TAESD
  3 | Text Encoders       | clip/, text_encoders/             | CLIP, T5, BERT
  4 | LoRAs               | loras/, loras/LyCORIS/            | LoRA, LyCORIS
  5 | ControlNet          | controlnet/, t2i_adapter/         | ControlNet, T2I
  6 | Upscalers           | upscale_models/                   | ESRGAN, etc.
  7 | Embeddings          | embeddings/                       | Textual Inversion
  8 | PhotoMaker          | photomaker/                       | PhotoMaker v1/v2
  9 | InsightFace         | insightface/                      | Face Swap Models
 10 | IP-Adapter          | ipadapter/, ipadapter-flux/       | Image Prompting
 11 | AnimateDiff         | animatediff_models/               | Motion Modules
 12 | SAM                 | sams/                             | Segment Anything
 13 | Grounding DINO      | grounding-dino/                   | Object Detection
 14 | YOLO                | ultralytics/bbox/, /segm/         | YOLO Detection
 15 | VLM & LLM           | VLM/, LLM/                        | Vision-Language

→ Für Details zu jedem Modul siehe: MANUAL_DE.txt


================================================================================
6. NAMENSKONVENTION - WAS BEDEUTEN DIE NAMEN?
================================================================================

TensorSort benennt Dateien nach diesem Schema:

{BaseModel}_{Component}_{Precision}_{Category}_{Name}_{Version}.ext

BEISPIELE:

FluxD_Full-FP16_NSFW_Persephone_v10.safetensors
  ↑      ↑         ↑     ↑           ↑
  │      │         │     │           └─ Version (v10)
  │      │         │     └───────────── Name (Persephone)
  │      │         └─────────────────── Kategorie (NSFW)
  │      └───────────────────────────── Komponenten + Precision (Full-FP16)
  └──────────────────────────────────── Base Model (FluxD = Flux Dev)

SDXL_VAE-FP16_v1.safetensors
  ↑    ↑
  │    └─ VAE in FP16 Precision
  └────── Für SDXL trainiert

FluxD_Style-LoRA_Watercolor_v2.safetensors
  ↑     ↑            ↑
  │     │            └─ Name (Watercolor)
  │     └────────────── LoRA-Typ (Style)
  └──────────────────── Für Flux Dev

WARUM SO BENANNT?

  ✓ Sortierbar: Alle Flux-Modelle stehen zusammen
  ✓ Suchbar: "FP16" findet alle FP16-Modelle
  ✓ Verständlich: Du siehst sofort was die Datei ist
  ✓ Konsistent: Immer gleicher Aufbau


================================================================================
7. HÄUFIGSTE PROBLEME & LÖSUNGEN
================================================================================

PROBLEM: "ComfyUI zeigt neue Dateinamen nicht an"
LÖSUNG:
  → Drücke Ctrl + Shift + R im Browser (Hard Refresh)
  → Das löscht den Cache und lädt die UI neu
  → WICHTIG: Normaler Refresh (F5) reicht NICHT!


PROBLEM: "Error loading LoRA"
URSACHE:
  → LoRA ist für falsches Base Model (z.B. Flux LoRA mit SDXL Checkpoint)
LÖSUNG:
  → Prüfe den Prefix im Dateinamen:
     FluxD_ = Nur mit Flux Dev verwenden
     SDXL_  = Nur mit SDXL verwenden
     SD15_  = Nur mit Stable Diffusion 1.5


PROBLEM: "Model not found"
URSACHE:
  → Datei liegt im falschen Ordner
LÖSUNG:
  → Führe Modus B aus: [2] CHECK & FIX MODELS
  → Zeigt alle falsch platzierten Dateien
  → Verschiebt sie automatisch


PROBLEM: "GGUF Checkpoint lädt nicht in CheckpointLoader"
URSACHE:
  → GGUF-Dateien müssen in unet/ liegen, NICHT in checkpoints/
  → ComfyUI-Beschränkung, nicht TensorSort-Problem
LÖSUNG:
  → TensorSort sortiert GGUF automatisch nach unet/
  → In ComfyUI: Verwende "Load Diffusion Model" Node (nicht CheckpointLoader)


PROBLEM: "Out of Memory (OOM) beim Laden"
URSACHE:
  → Modell zu groß für dein VRAM
  → FP16 Flux Dev = ~24 GB VRAM
  → T5-XXL Encoder = ~9.2 GB VRAM (!!!)
LÖSUNG:
  → Verwende GGUF-quantisiertes Modell:
     Q8_0 = ~12 GB (kaum Qualitätsverlust)
     Q4_K_M = ~6 GB (leichter Qualitätsverlust)
  → Oder: T5-XXL weglassen (nur CLIP verwenden)


PROBLEM: "IP-Adapter nicht gefunden"
URSACHE:
  → IP-Adapter gibt es in 3 verschiedenen Formaten
  → Jedes Format hat eigenen Ordner
LÖSUNG:
  → TensorSort erkennt automatisch:
     SDXL/SD15 → ipadapter/
     Flux (InstantX) → ipadapter-flux/
     Flux (XLabs) → xlabs/ipadapters/
  → Prüfe welchen Node du verwendest und welches Format du brauchst


PROBLEM: "Datei wurde übersprungen (SKIP)"
URSACHE:
  → TensorSort hat Datei nicht erkannt
  → Kann passieren bei:
     - Exotischen/Custom Formaten
     - Korrupten Downloads
     - Nicht-ML Dateien (.txt, .json, etc.)
LÖSUNG:
  → Prüfe ob Download vollständig war
  → Wenn Datei wichtig ist: Kontaktiere Support (siehe unten)
  → Unerkannte Dateien bleiben in downloads/ (sicher!)


================================================================================
8. WEITERFÜHRENDE INFORMATIONEN
================================================================================

VOLLSTÄNDIGES MANUAL:

  Diese Quick-Start-Anleitung deckt ~20% der Funktionen ab.

  Für vollständige Informationen siehe:
    → MANUAL_DE.txt (~1300 Zeilen)

  Enthält:
    - Detaillierte Erklärung aller 15 Module
    - Technische Details (Detection-Hierarchie, Queue System)
    - Namenskonventionen für alle Modul-Typen
    - Erweiterte Troubleshooting-Tipps
    - Quellenangaben und Referenzen


ENGLISH VERSION:

  → README_EN.txt (This guide in English)
  → MANUAL_EN.txt (Complete manual in English)


================================================================================
9. SUPPORT & KONTAKT
================================================================================

FRAGEN? PROBLEME? FEEDBACK?

Author: K0DA Parallax Studio
Email:  kodaparallax@gmail.com

Wir helfen gerne bei:
  - Technischen Problemen
  - Unerkannten Dateien
  - Feature-Requests
  - Bug-Reports

Bitte in der Email angeben:
  - Welches Modul (1-15)
  - Welcher Modus (A oder B)
  - Fehlermeldung (falls vorhanden)
  - Dateiname und Größe


================================================================================
                                VIEL ERFOLG!
================================================================================

TensorSort v1.1.0
"Weil Dateinamen lügen, aber Tensoren nicht."

Author: K0DA Parallax Studio
Email:  kodaparallax@gmail.com

================================================================================
