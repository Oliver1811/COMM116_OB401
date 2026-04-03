"""
baseline_diagram.py — Simple diagram for baseline VQA model.

Generates a clean, minimal diagram showing the baseline's single-pass approach.

Usage:
    python baseline_diagram.py

Outputs:
    baseline_simple.png (300 DPI)
    baseline_simple.pdf (vector format)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Configuration
FIGSIZE = (8, 10)
DPI = 300

# Minimal grayscale palette (same as agent diagram)
BG_LIGHT = '#FFFFFF'
BG_MED = '#F0F0F0'
BORDER = '#000000'
TEXT = '#000000'
ARROW = '#000000'

def draw_box(ax, x, y, w, h, text, fill='white', lw=2):
    """Draw a simple box with text."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.02',
                          edgecolor=BORDER,
                          facecolor=fill,
                          linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=10, color=TEXT)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw a simple arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='-|>',
                           color=ARROW,
                           linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
                fontsize=8, style='italic', color=TEXT)

# Create figure
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.axis('off')

cx = 4  # center x

# Title
ax.text(cx, 9.5, 'Baseline Architecture', 
        ha='center', fontsize=14, weight='bold')

# Subtitle
ax.text(cx, 9.1, '(Single-pass VQA, no tools, no loop)', 
        ha='center', fontsize=9, style='italic', color='#555')

# 1. Input
y = 8.2
draw_box(ax, cx-1.5, y, 3, 0.6, 'Input\nImage + Question')

# Arrow down
draw_arrow(ax, cx, y, cx, y-0.8)

# 2. Prompt formatting
y = 6.8
draw_box(ax, cx-1.8, y, 3.6, 0.7, 
         'Format Prompt\n"Answer concisely..."', fill=BG_MED)

# Arrow down
draw_arrow(ax, cx, y, cx, y-0.8)

# 3. Model inference (main component)
y = 5.3
draw_box(ax, cx-1.5, y, 3, 0.8, 
         'Single Model Call\nQwen2-VL-2B-Instruct', fill=BG_LIGHT, lw=2.5)

# Arrow down
draw_arrow(ax, cx, y, cx, y-0.9)

# 4. Post-processing
y = 3.8
draw_box(ax, cx-1.5, y, 3, 0.6, 
         'Strip Formatting\nExtract Answer', fill=BG_MED)

# Arrow down
draw_arrow(ax, cx, y, cx, y-0.8)

# 5. Output
y = 2.4
draw_box(ax, cx-1.5, y, 3, 0.6, 'Direct Answer', fill=BG_LIGHT, lw=2.5)

# Arrow down
draw_arrow(ax, cx, y, cx, y-0.8)

# 6. Outputs
y = 1.2
draw_box(ax, cx-2.5, y, 5, 0.7, 
         'Outputs\npredictions.jsonl')

# Save
plt.tight_layout()
plt.savefig('baseline_simple.png', dpi=DPI, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('baseline_simple.pdf', format='pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("✓ Saved: baseline_simple.png (300 DPI)")
print("✓ Saved: baseline_simple.pdf (vector)")
plt.close()
