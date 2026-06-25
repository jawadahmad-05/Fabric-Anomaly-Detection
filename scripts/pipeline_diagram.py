"""
Generate a publication-quality ZSAD pipeline visualization as PNG.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

# ── Color palette ──
C_INPUT   = '#42a5f5'   # blue
C_BACK    = '#ef5350'   # red  
C_KNN     = '#ab47bc'   # purple
C_WASS    = '#26a69a'   # teal
C_CENT    = '#ff7043'   # orange
C_HOMO    = '#78909c'   # blue-grey
C_IMG     = '#ffca28'   # amber
C_FUSE    = '#66bb6a'   # green
C_OUT     = '#8d6e63'   # brown
C_MEM     = '#ec407a'   # pink
C_WHITE   = '#ffffff'
C_DARK    = '#263238'
C_GREY    = '#546e7a'

def box(x, y, w, h, color, text, sub=None, fc=None, ec=None, fs=10, sfs=8, alpha=0.15):
    """Draw a rounded box with text."""
    fc = fc or color
    ec = ec or color
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=ec, linewidth=2, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.05, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', color=C_DARK)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.30, sub, ha='center', va='center',
                fontsize=sfs, color=C_GREY, fontstyle='italic')

def arrow(x1, y1, x2, y2, label='', lw=1.5, color='#78909c'):
    """Draw an arrow with optional label."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color=color))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        # small white background for label
        ax.text(mx, my + 0.12, label, ha='center', va='bottom',
                fontsize=7, color=C_GREY,
                bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.85))

def brace(x, y1, y2, label):
    """Draw a curly brace-like indicator on the left."""
    ax.plot([x, x], [y1, y2], color=C_GREY, lw=1.5, alpha=0.6)
    ax.text(x - 0.15, (y1+y2)/2, label, ha='right', va='center',
            fontsize=8, color=C_GREY, rotation=90)

# ═══════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════
ax.text(11, 13.5, 'ZSAD Pipeline — Step-by-Step Data Flow', 
        ha='center', fontsize=18, fontweight='bold', color=C_DARK)
ax.text(11, 13.0, 'Zero-Shot Anomaly Detection for Textured Surfaces',
        ha='center', fontsize=12, color=C_GREY, fontstyle='italic')

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Input + Preprocessing
# ═══════════════════════════════════════════════════════════════════
box(1.0, 11.0, 2.5, 1.2, C_INPUT, 'Step 1: Input', 'PIL Image (any size)', fs=11, sfs=8)
box(4.0, 11.0, 2.5, 1.2, C_INPUT, 'Preprocess', 'Resize 518Ã—518\nImageNet Norm', fs=11, sfs=8)
arrow(3.5, 11.6, 4.0, 11.6)

# ═══════════════════════════════════════════════════════════════════
# STEP 2: DINOv2 Backbone
# ═══════════════════════════════════════════════════════════════════
box(7.0, 10.2, 3.0, 2.0, C_BACK, 'Step 2: DINOv2 ViT-L/14', 
    'Frozen Backbone\n304M params, requires_grad=False\nBlock 11 forward hook\nPatch size: 14Ã—14', fs=11, sfs=7.5)
arrow(6.5, 11.6, 7.0, 11.6)

# Two output arrows from DINOv2
C_RED = '#ef5350'
arrow(8.5, 10.2, 8.5, 9.3, 'feats_518: (1,1369,1024)', color=C_RED)
arrow(9.0, 10.2, 9.0, 7.5, 'feats_224: (1,256,1024)', color=C_RED)
ax.text(9.6, 8.9, '(also used for', fontsize=6.5, color=C_GREY)
ax.text(9.6, 8.5, 'Homogeneity,', fontsize=6.5, color=C_GREY)
ax.text(9.6, 8.1, 'Image Score)', fontsize=6.5, color=C_GREY)

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Three Scoring Branches + Homogeneity + Image Score
# ═══════════════════════════════════════════════════════════════════

# 3A: kNN Score (top-left)
box(0.5, 6.8, 5.0, 2.0, C_KNN, 'Step 3A: kNN Score',
    'Pairwise L2 distance Ã— k=80 nearest neighbors\n'
    '518-scale: (37,37) + 224-scale: (16,16) upsample Ã— 0.4\n'
    'Fused: (knn_518 + 0.4Ã—knn_224) / 1.4\n'
    'Isolated patches â†’ high score (few similar neighbors)',
    fs=10, sfs=7)
arrow(7.0, 11.2, 5.5, 8.8, '', color=C_KNN)

# 3B: Wasserstein Score (top-right)
box(6.5, 6.8, 5.0, 2.0, C_WASS, 'Step 3B: Wasserstein Distance',
    'SVD â†’ PCA project 1024â†’64 dims\n'
    'Rank-based: sort per dim, |rank âˆ’ 0.5| Ã— 2\n'
    'Top-3 anomalous dims averaged per patch\n'
    'Extreme rank = anomalous patch',
    fs=10, sfs=7)
arrow(9.0, 10.2, 9.0, 8.8, '', color=C_WASS)

# 3C: Centroid Distance (mid-right)
box(12.5, 6.8, 5.0, 2.0, C_CENT, 'Step 3C: Centroid Distance',
    'SVD â†’ PCA project 1024â†’64 dims\n'
    'centroid = mean(all_patches)\n'
    'score_i = ||z_i âˆ’ centroid||â‚‚\n'
    'Far from center = anomalous',
    fs=10, sfs=7)
arrow(9.5, 10.2, 15.0, 8.8, '', color=C_CENT)
ax.text(12.3, 9.6, 'feats_518', fontsize=6.5, color=C_GREY)

# 3D: Homogeneity (bottom-left)
box(0.5, 3.8, 4.5, 1.5, C_HOMO, 'Step 3D: Homogeneity',
    'cosine_sim(patch_i, centroid)\n'
    'h = mean(similarities)\n'
    'Low h â†’ uniform texture, High h â†’ diverse',
    fs=10, sfs=7)
arrow(8.5, 10.2, 2.75, 5.3, '', color=C_HOMO)
ax.text(5.5, 8.2, 'feats_518', fontsize=6.5, color=C_GREY)

# 3E: Image Score (bottom-center)
box(5.5, 3.8, 4.5, 1.5, C_IMG, 'Step 3E: Image Score',
    'cosine_sim(patch_i, centroid)\n'
    'img_score = 1 âˆ’ quantile(sims, 0.005)\n'
    '0.5th percentile = 7th worst patch out of 1369',
    fs=10, sfs=7)
arrow(8.5, 10.2, 7.75, 5.3, '', color=C_IMG)

# ═══════════════════════════════════════════════════════════════════
# Optional: Memory (right side)
# ═══════════════════════════════════════════════════════════════════
box(18.0, 7.5, 3.5, 1.8, C_MEM, 'Step 6 (Optional): Memory',
    'TestSetMemory stores features\n'
    'from previous images\n'
    'kNN density query (k=7)\n'
    'Activates after 5+ images\n'
    'Added to fusion: +0.2Ã—mem',
    fs=9, sfs=7)
arrow(17.5, 8.4, 18.0, 8.4, '', color=C_MEM)
arrow(15.0, 10.2, 19.75, 10.2, 'store feats', color=C_MEM, lw=1)

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Score Fusion
# ═══════════════════════════════════════════════════════════════════
box(5.5, 0.8, 6.0, 2.0, C_FUSE, 'Step 4: Score Fusion & Normalization',
    '1. Min-max normalize each branch: norm(s) = (s âˆ’ min)/(max âˆ’ min)\n'
    '2. Weighted sum: 0.5Ã—kNN + 0.3Ã—Wass + 0.2Ã—Cent\n'
    '3. Homogeneity modulate: fused Ã— (0.5 + 0.5Ã—h)\n'
    '4. Add memory: + 0.2Ã—mem_kNN\n'
    '5. Optional: avg_pool2d smoothing (kernel=3/7)',
    fs=9.5, sfs=7)

# Arrows from all 3 branches + h_score + img_score to fusion
arrow(3.0, 6.8, 3.0, 2.8, '', color=C_KNN)
ax.text(2.5, 5.0, 's_knn (37,37)', fontsize=6.5, color=C_GREY, rotation=90)

arrow(9.0, 6.8, 9.0, 2.8, '', color=C_WASS)
ax.text(9.5, 5.0, 's_wass (37,37)', fontsize=6.5, color=C_GREY, rotation=90)

arrow(15.0, 6.8, 15.0, 2.8, '', color=C_CENT)
ax.text(15.5, 5.0, 's_cent (37,37)', fontsize=6.5, color=C_GREY, rotation=90)

arrow(2.75, 3.8, 3.0, 2.8, '', color=C_HOMO)
ax.text(2.5, 3.3, 'h_score', fontsize=6.5, color=C_GREY)

arrow(7.75, 3.8, 7.5, 2.8, '', color=C_IMG)
ax.text(7.0, 3.3, 'img_score', fontsize=6.5, color=C_GREY)

arrow(19.75, 7.5, 19.75, 2.8, 'mem_map (37,37)', color=C_MEM)
ax.text(20.3, 5.0, 's_mem', fontsize=6.5, color=C_GREY, rotation=90)

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Output
# ═══════════════════════════════════════════════════════════════════
box(13.0, 0.8, 4.5, 2.0, C_OUT, 'Step 5: Output',
    'score_map: (37,37) numpy array [0,1]\n'
    'img_score: float (e.g. 0.89)\n'
    'h_score: float (e.g. 0.35)\n'
    'â†’ Used by IoT server for display + defect naming',
    fs=9.5, sfs=7)
arrow(11.5, 1.8, 13.0, 1.8, '', color=C_OUT)

# ═══════════════════════════════════════════════════════════════════
# SECTION LABELS
# ═══════════════════════════════════════════════════════════════════
# Left side labels
brace(0.1, 11.0, 12.2, 'BACKBONE')
brace(0.1, 6.8, 3.8, 'FEATURE EXTRACTION')

# Group boxes for the 3 branches
# A big dashed box around the three scoring branches
branch_rect = FancyBboxPatch((0.2, 6.6), 17.6, 2.5, 
                              boxstyle="round,pad=0.1",
                              facecolor='none', edgecolor=C_GREY, 
                              linewidth=1.5, linestyle='--', alpha=0.5)
ax.add_patch(branch_rect)
ax.text(9.0, 9.25, 'Three Parallel Scoring Branches', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_GREY,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

# ═══════════════════════════════════════════════════════════════════
# KEY / LEGEND
# ═══════════════════════════════════════════════════════════════════
legend_y = 0.1
legends = [
    (C_INPUT, 'Input / Preprocessing'),
    (C_BACK, 'DINOv2 ViT-L/14 (Frozen)'),
    (C_KNN, 'kNN Score Branch'),
    (C_WASS, 'Wasserstein Score Branch'),
    (C_CENT, 'Centroid Score Branch'),
    (C_HOMO, 'Homogeneity Score'),
    (C_IMG, 'Image-Level Score'),
    (C_FUSE, 'Score Fusion'),
    (C_MEM, 'Test-Set Memory (Optional)'),
    (C_OUT, 'Output'),
]
for i, (c, label) in enumerate(legends):
    x = 0.5 + i * 2.1
    rect = FancyBboxPatch((x, legend_y), 0.4, 0.35, boxstyle="round,pad=0.05",
                          facecolor=c, edgecolor=c, linewidth=1.5, alpha=0.25)
    ax.add_patch(rect)
    ax.text(x + 0.5, legend_y + 0.17, label, ha='left', va='center', fontsize=7, color=C_DARK)

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
plt.tight_layout()
import os
HERE = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(HERE, '..', 'paper_figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'pipeline_diagram.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.3, facecolor='white')
print(f"Saved: {out_path}")
plt.close()
