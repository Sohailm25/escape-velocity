# ABOUTME: Paper A figure styling constants derived from sohailmo.ai Everforest palette.
# ABOUTME: Provides condition colors, theme application, and figure save utilities.
"""
Sohail Research Theme — Paper A figure styling constants.
Derived from sohailmo.ai (Everforest-inspired palette).

Usage:
    from theme import CONDITION_COLORS, COLORS, apply_theme, save_figure
    apply_theme()  # sets matplotlib style globally
"""
import matplotlib.pyplot as plt
from pathlib import Path

# === Core palette (for white background PDF) ===
COLORS = {
    # Primary data series
    'green':    '#4A7A5B',   # Forest green
    'teal':     '#5A9BA3',   # Blue-teal
    'coral':    '#C44D5E',   # Warm coral
    'purple':   '#7B6B8A',   # Muted purple
    'amber':    '#B8943E',   # Warm amber
    'charcoal': '#2d353b',   # Dark accent

    # Semantic colors
    'success':  '#4A7A5B',   # Green — low collapse
    'partial':  '#B8943E',   # Amber — moderate collapse
    'fail':     '#C44D5E',   # Coral — high collapse

    # From website directly (for annotations/highlights)
    'sage':     '#A7C080',   # Light sage (website primary)
    'mint':     '#83C092',   # Mint accent (website accent)
    'blue':     '#7FBBB3',   # Soft teal (website secondary)
    'rose':     '#E8868E',   # Soft rose (website warm)

    # Neutrals
    'dark_bg':  '#2d353b',   # Website background
    'card_bg':  '#343f44',   # Website card
    'border':   '#4a555b',   # Website border
    'text':     '#2d353b',   # Text on white bg
    'muted':    '#888888',   # Muted text
}

# Ordered cycle for multi-series plots
CYCLE = [COLORS['green'], COLORS['teal'], COLORS['coral'],
         COLORS['purple'], COLORS['amber'], COLORS['charcoal']]

# Paper A condition colors
CONDITION_COLORS = {
    'HOMO_A': COLORS['green'],      # Llama-3.1-8B
    'HOMO_B': COLORS['teal'],       # Qwen2.5-7B
    'HOMO_C': COLORS['coral'],      # Mistral-7B
    'HETERO_ROT': COLORS['purple'], # Rotation
}


def apply_theme():
    """Apply the sohail_research style globally."""
    style_path = Path(__file__).parent.parent / 'paper' / 'sohail_research.mplstyle'
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        print(f"Warning: style file not found at {style_path}")


def setup_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create a figure with theme defaults."""
    apply_theme()
    if figsize is None:
        w = 6.5 if ncols == 1 else 10
        h = 4.0 * nrows
        figsize = (w, h)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save_figure(fig, name, formats=('pdf', 'png')):
    """Save figure in multiple formats to paper/figures/."""
    out_dir = Path(__file__).parent.parent / 'figures'
    out_dir.mkdir(exist_ok=True)
    for fmt in formats:
        path = out_dir / f'{name}.{fmt}'
        fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {path}")


def collapse_color(rate):
    """Return color based on collapse rate severity."""
    if rate >= 0.6:
        return COLORS['fail']      # High collapse
    elif rate >= 0.3:
        return COLORS['partial']   # Moderate
    else:
        return COLORS['success']   # Low collapse
