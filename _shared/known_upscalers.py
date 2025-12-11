#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Base für bekannte Upscaler Models
Zweck: Type + Architecture Info für populäre Upscaler die nicht aus Filename extrahierbar sind

Status: ✅ KOMPLETT (2025-01-29)
Basiert auf: OpenModelDB, Civitai Research, Community Best Practices

Struktur:
- Key: Filename-Pattern (lowercase, partial match)
- Value: {'type': str, 'arch': str}
  - type: Spezialisierung (Anime, Photo, Text, etc.)
  - arch: Architecture (ESRGAN, NMKD, SwinIR, etc.)

Verwendung:
>>> info = lookup_known_upscaler("8x_NMKD-Superscale_150000_G.pth")
>>> print(info)
{'type': 'Photo', 'arch': 'NMKD'}
"""

# ============================================================================
# KNOWN UPSCALERS DATABASE (~50 TOP MODELS)
# ============================================================================

KNOWN_UPSCALERS = {

    # ========================================================================
    # ANIME/MANGA UPSCALERS
    # ========================================================================

    # RealESRGAN Anime Series (most popular)
    'realesrgan_x4plus_anime': {
        'type': 'Anime',
        'arch': 'RealESRGAN',
        'description': 'Most popular anime upscaler, 6B params'
    },
    'realesrgan-x4-plus-anime': {
        'type': 'Anime',
        'arch': 'RealESRGAN',
        'description': 'Alternative naming for x4plus anime'
    },

    # AnimeSharp Series
    'animesharp': {
        'type': 'Anime-Detail',
        'arch': 'ESRGAN',
        'description': 'Sharp anime upscaler with detail enhancement'
    },
    '4xanimesharp': {
        'type': 'Anime-Detail',
        'arch': 'ESRGAN',
        'description': '4x AnimeSharp variant'
    },

    # Fatal Anime Series
    'fatal_anime': {
        'type': 'Anime',
        'arch': 'ESRGAN',
        'description': 'Fatal Anime upscaler series'
    },
    'fatalanime': {
        'type': 'Anime',
        'arch': 'ESRGAN',
        'description': 'Fatal Anime (alternative naming)'
    },

    # NMKD Anime
    'nmkd-ultrayandere': {
        'type': 'Anime',
        'arch': 'NMKD',
        'description': 'NMKD Anime-specialized upscaler'
    },

    # Other Anime Models
    'hfa2k': {
        'type': 'Anime',
        'arch': 'ESRGAN',
        'description': 'HFA2k anime restoration model'
    },
    'anime4k': {
        'type': 'Anime',
        'arch': 'Anime4K',
        'description': 'Real-time anime video upscaler'
    },
    'anijapan': {
        'type': 'Anime',
        'arch': 'ESRGAN',
        'description': 'Anime Japan style upscaler'
    },
    'aniscale': {
        'type': 'Anime',
        'arch': 'ESRGAN',
        'description': 'Anime scale upscaler'
    },

    # ========================================================================
    # PHOTO/REALISTIC UPSCALERS
    # ========================================================================

    # NMKD Photo Series
    'nmkd-superscale': {
        'type': 'Photo',
        'arch': 'NMKD',
        'description': 'NMKD Superscale - photo optimized, most popular photo upscaler'
    },

    # UltraSharp Series
    'ultrasharp': {
        'type': 'Photo-Detail',
        'arch': 'ESRGAN',
        'description': 'UltraSharp for photo detail enhancement'
    },
    '4xultrasharp': {
        'type': 'Photo-Detail',
        'arch': 'ESRGAN',
        'description': '4x UltraSharp variant'
    },

    # RealESRGAN Photo
    'realesrgan-x4plus': {
        'type': 'Photo',
        'arch': 'RealESRGAN',
        'description': 'Standard Real-ESRGAN for photos'
    },
    'realesrgan_x4plus': {
        'type': 'Photo',
        'arch': 'RealESRGAN',
        'description': 'Standard Real-ESRGAN (underscore variant)'
    },

    # Foolhardy Remacri
    'foolhardy': {
        'type': 'Photo-Detail',
        'arch': 'ESRGAN',
        'description': 'Foolhardy Remacri - detail enhancement'
    },
    'remacri': {
        'type': 'Photo-Detail',
        'arch': 'ESRGAN',
        'description': 'Remacri upscaler'
    },

    # Nomos Series (Web/Photo)
    'nomos8k': {
        'type': 'Photo-Web',
        'arch': 'ESRGAN',
        'description': 'Nomos 8k - optimized for web JPGs'
    },
    'nomos2': {
        'type': 'Photo',
        'arch': 'ESRGAN',
        'description': 'Nomos2 photo upscaler'
    },
    'nomos8kdat': {
        'type': 'Photo',
        'arch': 'DAT',
        'description': 'Nomos 8k DAT transformer variant'
    },

    # Face/Portrait Specialized
    'artfaces': {
        'type': 'Photo-Face',
        'arch': 'ESRGAN',
        'description': 'Art faces portrait upscaler'
    },
    'gfpgan': {
        'type': 'Photo-Face',
        'arch': 'GAN',
        'description': 'Face restoration GAN'
    },
    'codeformer': {
        'type': 'Photo-Face',
        'arch': 'Transformer',
        'description': 'Face restoration transformer'
    },

    # SUPIR (Photo-realistic restoration)
    'supir': {
        'type': 'Photo-Realistic',
        'arch': 'Diffusion',
        'description': 'SUPIR photo-realistic restoration'
    },

    # ========================================================================
    # AI-GENERATED CONTENT UPSCALERS
    # ========================================================================

    'aigcsmooth': {
        'type': 'AI-Smooth',
        'arch': 'ESRGAN',
        'description': 'AI-generated content smooth upscaler'
    },
    'smooth_diff': {
        'type': 'AI-Smooth',
        'arch': 'ESRGAN',
        'description': 'Smooth diffusion upscaler'
    },
    'smoothdiff': {
        'type': 'AI-Smooth',
        'arch': 'ESRGAN',
        'description': 'Smooth diff (alternative naming)'
    },

    # ========================================================================
    # SPECIAL PURPOSE UPSCALERS
    # ========================================================================

    # Text/Documents
    'text2hd': {
        'type': 'Text',
        'arch': 'RealPLKSR',
        'description': 'Text to HD upscaler'
    },
    'typescale': {
        'type': 'Text',
        'arch': 'NMKD',
        'description': 'NMKD Typescale for text'
    },

    # Pixel Art
    'pixelart': {
        'type': 'PixelArt',
        'arch': 'ESRGAN',
        'description': 'Pixel art game upscaler'
    },

    # Game Content
    'gamescreenshot': {
        'type': 'GameScreenshot',
        'arch': 'ESRGAN',
        'description': 'Game screenshot upscaler'
    },

    # ========================================================================
    # GENERAL/MODERN ARCHITECTURES
    # ========================================================================

    # SwinIR (Transformer)
    'swinir': {
        'type': 'General',
        'arch': 'SwinIR',
        'description': 'Transformer-based, high quality, slower'
    },

    # DAT (Dual Aggregation Transformer)
    'dat': {
        'type': 'General',
        'arch': 'DAT',
        'description': 'Dual Aggregation Transformer - ultra quality, very slow'
    },

    # RealPLKSR (Modern, VRAM-efficient)
    'realplksr': {
        'type': 'General',
        'arch': 'RealPLKSR',
        'description': 'VRAM-efficient modern upscaler'
    },
    'plksr': {
        'type': 'General',
        'arch': 'RealPLKSR',
        'description': 'PLKSR variant'
    },

    # BSRGAN
    'bsrgan': {
        'type': 'General',
        'arch': 'BSRGAN',
        'description': 'Blind super-resolution GAN'
    },

    # ESRGAN (original)
    'esrgan': {
        'type': 'General',
        'arch': 'ESRGAN',
        'description': 'Original ESRGAN'
    },

    # RealESRGAN General
    'realesrgan': {
        'type': 'General',
        'arch': 'RealESRGAN',
        'description': 'RealESRGAN general (if not anime/photo specific)'
    },

    # RealESRNet
    'realesrnet': {
        'type': 'General',
        'arch': 'RealESRGAN',
        'description': 'RealESRNet variant'
    },

    # HAT (Hybrid Attention Transformer)
    'hat': {
        'type': 'General',
        'arch': 'HAT',
        'description': 'Hybrid Attention Transformer'
    },
}


# ============================================================================
# LOOKUP FUNCTION
# ============================================================================

def lookup_known_upscaler(filename):
    """Prüft ob Upscaler in Knowledge Base bekannt ist

    Args:
        filename (str): Dateiname (mit oder ohne Extension)

    Returns:
        dict or None: {'type': str, 'arch': str, 'description': str} oder None

    Beispiel:
        >>> info = lookup_known_upscaler("8x_NMKD-Superscale_150000_G.pth")
        >>> print(info['type'], info['arch'])
        Photo NMKD
    """
    filename_lower = filename.lower()

    # Entferne Extension für besseres Matching
    if filename_lower.endswith(('.pth', '.pt', '.safetensors')):
        filename_lower = filename_lower.rsplit('.', 1)[0]

    # Check gegen alle bekannten Patterns
    for pattern, info in KNOWN_UPSCALERS.items():
        if pattern in filename_lower:
            return info

    return None  # Unknown


# ============================================================================
# STATISTICS
# ============================================================================

def get_stats():
    """Gibt Statistiken über die Knowledge Base zurück"""

    types = {}
    archs = {}

    for info in KNOWN_UPSCALERS.values():
        type_val = info['type']
        arch_val = info['arch']

        types[type_val] = types.get(type_val, 0) + 1
        archs[arch_val] = archs.get(arch_val, 0) + 1

    return {
        'total': len(KNOWN_UPSCALERS),
        'types': types,
        'architectures': archs
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test Lookup
    test_files = [
        "8x_NMKD-Superscale_150000_G.pth",
        "4x-AnimeSharp.pth",
        "4x_fatal_Anime_500000_G.pth",
        "Upscaler_4x_nomos8k-atd-jpg.pth",
        "mysterious_upscaler.pth",  # Unknown
    ]

    print("="*80)
    print("KNOWN UPSCALERS - KNOWLEDGE BASE TEST")
    print("="*80)
    print()

    for filename in test_files:
        info = lookup_known_upscaler(filename)
        print(f"File: {filename}")
        if info:
            print(f"  Type: {info['type']}")
            print(f"  Architecture: {info['arch']}")
            print(f"  Description: {info['description']}")
        else:
            print(f"  -> Unknown (not in knowledge base)")
        print()

    # Statistics
    stats = get_stats()
    print("="*80)
    print("KNOWLEDGE BASE STATISTICS")
    print("="*80)
    print(f"\nTotal Models: {stats['total']}")
    print(f"\nTypes:")
    for type_name, count in sorted(stats['types'].items()):
        print(f"  {type_name}: {count}")
    print(f"\nArchitectures:")
    for arch_name, count in sorted(stats['architectures'].items()):
        print(f"  {arch_name}: {count}")
