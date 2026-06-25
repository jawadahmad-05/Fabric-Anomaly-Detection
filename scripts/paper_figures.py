"""
Generate all paper figures for ZSAD research paper.
Output: paper_figures/ directory with publication-quality figures.

Usage:
    python paper_figures.py
"""
import sys, os, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE
from torchmetrics.functional import auroc

from model.zsad_architecture import ZSAD, TestSetMemory

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "paper_figures"
OUTPUT_DIR.mkdir(exist_ok=True)
MVTEC_ROOT = Path('D:/datasets/MVTec AD')

# Texture classes for paper focus
TEXTURE_CLASSES = ['carpet', 'grid', 'leather', 'tile', 'wood']
TEXTURE_LABELS = ['Carpet', 'Grid', 'Leather', 'Tile', 'Wood']

# Color scheme
CMAP_ANOM = 'jet'
COLOR_DEFECT = '#ff1744'
COLOR_NORMAL = '#00c853'
COLOR_GRID = '#2196f3'


# ============================================================
# Helper
# ============================================================
def load_gt_mask(img_path, defect_type):
    """Load ground truth mask for MVTec AD."""
    parts = str(img_path).replace('\\', '/').split('/')
    cls_name = parts[-3]
    fname = parts[-1]
    mask_path = MVTEC_ROOT / cls_name / 'ground_truth' / defect_type / fname.replace('.png', '_mask.png')
    if mask_path.exists():
        return np.array(Image.open(mask_path).convert('L')) > 127
    return None


def overlay_heatmap(img, score_map, alpha=0.6):
    """Overlay jet colormap on image using score map (resized to image)."""
    from scipy.ndimage import zoom
    img_np = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
    h, w = img_np.shape[:2]
    scale = (h / score_map.shape[0], w / score_map.shape[1])
    sm_resized = zoom(score_map, scale, order=1)
    sm_norm = (sm_resized - sm_resized.min()) / (sm_resized.max() - sm_resized.min() + 1e-8)
    cmap = plt.colormaps['jet']
    heatmap = cmap(sm_norm)[:, :, :3]
    overlaid = (1 - alpha) * img_np + alpha * heatmap
    return np.clip(overlaid, 0, 1)


# ============================================================
# Figure 1: MVTec Texture Heatmap Grid (5x5)
# ============================================================
def fig_heatmap_grid(zsad, dpi=200):
    print("[Fig 1] MVTec texture heatmap grid...")
    defect_types = ['cut', 'hole', 'color', 'thread', 'metal_contamination']
    # Use per-class defect lists
    per_class_defects = {
        'carpet': ['cut', 'hole', 'color', 'thread', 'metal_contamination'],
        'grid': ['bent', 'broken', 'glue', 'metal_contamination', 'thread'],
        'leather': ['cut', 'poke', 'color', 'fold', 'glue'],
        'tile': ['crack', 'glue_strip', 'gray_stroke', 'oil', 'rough'],
        'wood': ['scratch', 'hole', 'color', 'liquid', 'combined'],
    }
    display_names = {
        'cut': 'Cut', 'hole': 'Hole', 'color': 'Color', 'thread': 'Thread',
        'metal_contamination': 'Metal', 'bent': 'Bent', 'broken': 'Broken',
        'glue': 'Glue', 'poke': 'Poke', 'fold': 'Fold', 'crack': 'Crack',
        'glue_strip': 'Glue Strip', 'gray_stroke': 'Gray Stroke', 'oil': 'Oil',
        'rough': 'Rough', 'scratch': 'Scratch', 'liquid': 'Liquid',
        'combined': 'Combined',
    }

    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for row, cls in enumerate(TEXTURE_CLASSES):
        defects = per_class_defects[cls]
        for col, defect_type in enumerate(defects):
            img_dir = MVTEC_ROOT / cls / 'test' / defect_type
            imgs = sorted(img_dir.glob('*.png'))
            if not imgs:
                axes[row][col].axis('off')
                continue
            img = Image.open(imgs[0]).convert('RGB')
            score_map, img_score, _ = zsad.score(img)
            overlaid = overlay_heatmap(img, score_map)
            axes[row][col].imshow(overlaid)
            axes[row][col].set_title(f'{TEXTURE_LABELS[row]}\n{display_names.get(defect_type, defect_type)}\nscore={img_score:.3f}',
                                     fontsize=9, fontweight='bold')
            axes[row][col].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    path = OUTPUT_DIR / 'fig1_heatmap_grid.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 2: Overlay + GT comparison (3 samples per texture)
# ============================================================
def fig_overlay_comparison(zsad, dpi=200):
    print("[Fig 2] Overlay vs ground truth comparison...")
    per_cls = {'carpet': 'cut', 'leather': 'poke', 'tile': 'crack', 'wood': 'hole', 'grid': 'broken'}
    fig, axes = plt.subplots(5, 4, figsize=(16, 22))
    for row, (cls, defect_type) in enumerate(per_cls.items()):
        img_dir = MVTEC_ROOT / cls / 'test' / defect_type
        imgs = sorted(img_dir.glob('*.png'))
        if not imgs:
            continue
        img = Image.open(imgs[0]).convert('RGB')
        score_map, img_score, _ = zsad.score(img)
        gt = load_gt_mask(imgs[0], defect_type)
        overlaid = overlay_heatmap(img, score_map)
        sm_disp = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)

        # Original
        axes[row][0].imshow(img)
        axes[row][0].set_title(f'{TEXTURE_LABELS[row]}\n{defect_type}', fontsize=10, fontweight='bold')
        axes[row][0].axis('off')

        # Ground truth
        if gt is not None:
            axes[row][1].imshow(gt, cmap='gray')
            axes[row][1].set_title('Ground Truth', fontsize=10)
        else:
            axes[row][1].text(0.5, 0.5, 'N/A', ha='center', va='center')
        axes[row][1].axis('off')

        # Score map
        im = axes[row][2].imshow(sm_disp, cmap=CMAP_ANOM, vmin=0, vmax=1)
        axes[row][2].set_title(f'Anomaly Map\n(img score={img_score:.3f})', fontsize=10)
        axes[row][2].axis('off')

        # Overlay
        axes[row][3].imshow(overlaid)
        axes[row][3].set_title('Overlay', fontsize=10)
        axes[row][3].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    path = OUTPUT_DIR / 'fig2_overlay_comparison.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 3: Per-class pixel & image ROC curves
# ============================================================
def compute_roc_data(zsad, cls, max_samples=50):
    """Return (img_scores, img_labels, pixel_scores, pixel_labels) for one class."""
    from torch.utils.data import DataLoader, Dataset

    hub_path = Path('D:/ACD-CLIP-main/dataset/hub/mvtec_texture_test.jsonl')
    normal_samples, defect_samples = [], []
    for line in open(hub_path, 'r', encoding='utf-8'):
        d = json.loads(line)
        if d.get('class_name') != cls:
            continue
        if d.get('label', 0) == 0:
            normal_samples.append(d)
        elif d.get('mask_path'):
            defect_samples.append(d)

    np.random.shuffle(normal_samples)
    np.random.shuffle(defect_samples)
    samples = defect_samples[:max_samples] + normal_samples[:max_samples]

    img_scores, img_labels = [], []
    all_pix_s, all_pix_l = [], []
    mem = TestSetMemory()

    for i, s in enumerate(samples):
        img = Image.open(MVTEC_ROOT / s['image_path']).convert('RGB')
        mask = None
        if s.get('mask_path') and (MVTEC_ROOT / s['mask_path']).exists():
            mask = np.array(Image.open(MVTEC_ROOT / s['mask_path']).resize(img.size, Image.LANCZOS).convert('L')) > 127
        else:
            mask = np.zeros(img.size[::-1], dtype=bool)
        score, img_s, _ = zsad.score(img, image_id=i, memory=mem)
        # Pixel-level: interpolate score to image size
        st = torch.from_numpy(score).float().unsqueeze(0).unsqueeze(0)
        st = F.interpolate(st, size=mask.shape, mode='bilinear', align_corners=False)[0, 0]
        all_pix_s.append(st.numpy().flatten())
        all_pix_l.append(mask.astype(np.float32).flatten())
        img_scores.append(img_s)
        img_labels.append(float(s.get('label', 0)))

    return np.array(img_scores), np.array(img_labels), all_pix_s, all_pix_l


def fig_roc_curves(zsad, dpi=200):
    print("[Fig 3] ROC curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#e53935', '#1e88e5', '#43a047', '#fb8c00', '#8e24aa']
    markers = ['o', 's', 'D', '^', 'v']

    for ax, level, title in zip(axes, ['pixel', 'image'],
                                  ['Pixel-Level ROC Curves', 'Image-Level ROC Curves']):
        for idx, cls in enumerate(TEXTURE_CLASSES):
            print(f"  Computing {cls} {level}...")
            img_scores, img_labels, pix_s_list, pix_l_list = compute_roc_data(zsad, cls, max_samples=30)
            if level == 'image':
                scores, labels = img_scores, img_labels
            else:
                scores = np.concatenate(pix_s_list)
                labels = np.concatenate(pix_l_list)
            # Compute ROC manually
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr) * 100
            ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{TEXTURE_LABELS[idx]} ({roc_auc:.1f}%)',
                    marker=markers[idx], markevery=len(fpr)//5, markersize=6)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig3_roc_curves.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 4: Score distribution histograms
# ============================================================
def fig_score_distributions(zsad, dpi=200):
    print("[Fig 4] Score distributions...")
    hub_path = Path('D:/ACD-CLIP-main/dataset/hub/mvtec_texture_test.jsonl')
    all_data = {cls: {'normal': [], 'defect': []} for cls in TEXTURE_CLASSES}
    lines = open(hub_path, 'r', encoding='utf-8').readlines()
    np.random.shuffle(lines)
    for line in lines:
        d = json.loads(line)
        if d.get('class_name') not in TEXTURE_CLASSES:
            continue
        cls = d['class_name']
        if d.get('label', 0) == 0 and len(all_data[cls]['normal']) < 30:
            all_data[cls]['normal'].append(d)
        elif d.get('label', 0) == 1 and len(all_data[cls]['defect']) < 30:
            all_data[cls]['defect'].append(d)
    for cls in TEXTURE_CLASSES:
        for label in ['normal', 'defect']:
            for s in all_data[cls][label]:
                img = Image.open(MVTEC_ROOT / s['image_path']).convert('RGB')
                _, img_s, _ = zsad.score(img)
                s['_score'] = img_s

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for idx, cls in enumerate(TEXTURE_CLASSES):
        ax = axes[idx]
        n_scores = [s['_score'] for s in all_data[cls]['normal']]
        d_scores = [s['_score'] for s in all_data[cls]['defect']]
        ax.hist(n_scores, bins=20, alpha=0.7, color=COLOR_NORMAL, label=f'Normal (n={len(n_scores)})',
                edgecolor='white', linewidth=0.5)
        ax.hist(d_scores, bins=20, alpha=0.7, color=COLOR_DEFECT, label=f'Defect (n={len(d_scores)})',
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Image Score', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(TEXTURE_LABELS[idx], fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = OUTPUT_DIR / 'fig4_score_distributions.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 5: t-SNE feature space visualization
# ============================================================
def fig_tsne(zsad, dpi=200):
    print("[Fig 5] t-SNE feature space...")
    hub_path = Path('D:/ACD-CLIP-main/dataset/hub/mvtec_texture_test.jsonl')
    feat_list, labels_list = [], []

    for line in open(hub_path, 'r', encoding='utf-8'):
        d = json.loads(line)
        if d.get('class_name') not in ['carpet', 'leather']:
            continue
        img = Image.open(MVTEC_ROOT / d['image_path']).convert('RGB')
        sm, _, _, feats, _ = zsad.score(img, return_features=True)
        n_patches = 37 * 37
        patch_feats = feats[0, :n_patches].cpu().numpy()
        is_defect = d.get('label', 0) == 1
        defect_type = None
        if is_defect and d.get('mask_path') and (MVTEC_ROOT / d['mask_path']).exists():
            # Get class name from image_path
            parts = d['image_path'].replace('\\', '/').split('/')
            defect_type = parts[-2]
            mask = np.array(Image.open(MVTEC_ROOT / d['mask_path']).resize((37, 37), Image.NEAREST).convert('L')) > 127
        elif is_defect:
            continue
        if is_defect and defect_type:
            defect_mask = mask.flatten()
            for i in range(n_patches):
                feat_list.append(patch_feats[i])
                labels_list.append(defect_type if defect_mask[i] else 'normal')
        else:
            for i in range(min(n_patches, 200)):  # subsample normal patches
                feat_list.append(patch_feats[i])
                labels_list.append('normal')
        if len(feat_list) >= 2000:
            break

    feat_arr = np.array(feat_list)
    print(f"  t-SNE on {feat_arr.shape[0]} patches...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    emb = tsne.fit_transform(feat_arr)

    color_map = {'normal': COLOR_NORMAL, 'cut': '#e53935', 'hole': '#d32f2f',
                 'color': '#fdd835', 'thread': '#7b1fa2', 'poke': '#c62828',
                 'fold': '#ff8f00', 'glue': '#f9a825', 'bent': '#1565c0',
                 'broken': '#0d47a1', 'metal_contamination': '#455a64',
                 'crack': '#e65100', 'glue_strip': '#f57f17', 'gray_stroke': '#424242',
                 'oil': '#ff6f00', 'rough': '#4e342e', 'scratch': '#1b5e20',
                 'liquid': '#00695c', 'combined': '#4a148c'}
    unique_labels = sorted(set(labels_list))
    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = np.array([l == label for l in labels_list])
        c = color_map.get(label, '#888')
        label_display = label.replace('_', ' ').title() if label != 'normal' else 'Normal'
        ax.scatter(emb[mask, 0], emb[mask, 1], c=c, label=label_display,
                   s=6, alpha=0.5, edgecolors='none')
    ax.set_title('t-SNE of DINOv2 Patch Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, markerscale=3)
    ax.axis('off')
    plt.tight_layout()
    path = OUTPUT_DIR / 'fig5_tsne_features.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 6: Fabric dataset samples (AITEX/WFT/UniFAB7)
# ============================================================
def fig_fabric_samples(zsad, dpi=200):
    print("[Fig 6] Fabric dataset anomaly maps...")
    datasets = {
        'AITEX': Path('D:/datasets/AITEX'),
        'WFT': Path('D:/datasets/WFT'),
        'DTD-Synthetic': Path('D:/datasets/DTD-Synthetic/DTD-Synthetic'),
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for row, (dset_name, dset_root) in enumerate(datasets.items()):
        if not dset_root.exists():
            axes[row][0].axis('off')
            axes[row][1].axis('off')
            axes[row][2].axis('off')
            continue

        # Find defect images
        defect_imgs = []
        if dset_name == 'AITEX':
            test_dir = dset_root / 'test' / 'defect'
            if test_dir.exists():
                defect_imgs = sorted(test_dir.glob('*.png'))[:3]
        elif dset_name == 'WFT':
            test_dir = dset_root
            if test_dir.exists():
                defect_imgs = sorted(test_dir.glob('*/*/*.jpg'))[:3] + sorted(test_dir.glob('*.jpg'))[:3]
                defect_imgs = defect_imgs[:3]
        elif dset_name == 'DTD-Synthetic':
            for sub in ['hole', 'stain']:
                p = dset_root / sub
                if p.exists():
                    defect_imgs.extend(sorted(p.glob('*.jpg'))[:2])
            defect_imgs = defect_imgs[:3]

        for col in range(3):
            if col < len(defect_imgs):
                img_path = defect_imgs[col]
                img = Image.open(img_path).convert('RGB')
                score_map, img_score, _ = zsad.score(img)
                overlaid = overlay_heatmap(img, score_map)
                axes[row][col].imshow(overlaid)
                axes[row][col].set_title(f'{dset_name}\nscore={img_score:.3f}', fontsize=10, fontweight='bold')
            else:
                axes[row][col].axis('off')
            axes[row][col].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.25)
    path = OUTPUT_DIR / 'fig6_fabric_samples.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 7: Architecture overview (data flow diagram)
# ============================================================
def fig_architecture(dpi=200):
    """Generate a clean architecture overview figure."""
    print("[Fig 7] Architecture diagram...")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors
    c_input = '#42a5f5'
    c_backbone = '#ef5350'
    c_scoring = '#66bb6a'
    c_fusion = '#ffa726'
    c_output = '#ab47bc'

    def draw_box(x, y, w, h, color, text, subtext=None):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.05, text, ha='center', va='center', fontsize=10, fontweight='bold')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.25, subtext, ha='center', va='center', fontsize=7, alpha=0.7)

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#666'))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.15, label, ha='center', fontsize=7, color='#666')

    # Boxes
    draw_box(0.5, 2.5, 2, 1, c_input, 'Input Image', '518x518 RGB')
    draw_box(3.5, 2.0, 2, 2, c_backbone, 'DINOv2 ViT-L/14', 'Frozen Backbone')
    draw_box(6.5, 0.5, 2.2, 1.2, c_scoring, 'kNN Score', 'Local feature density')
    draw_box(6.5, 2.0, 2.2, 1.2, c_scoring, 'Wasserstein', 'Statistical distance')
    draw_box(6.5, 3.5, 2.2, 1.2, c_scoring, 'Centroid Dist.', 'Cosine to mean')
    draw_box(9.5, 1.5, 1.8, 2.0, c_fusion, 'Score Fusion', 'Weighted sum\n+ homogeneity')
    draw_box(12.5, 1.5, 1.8, 2.0, c_output, 'Anomaly Map', '37x37 score map\n+ image score')

    # Arrows
    arrow(2.5, 3.0, 3.5, 3.0)
    ax.text(3.0, 3.5, 'Patch features', ha='center', fontsize=7, color='#666')
    arrow(5.5, 2.6, 6.5, 1.1)
    arrow(5.5, 3.0, 6.5, 2.6)
    arrow(5.5, 3.4, 6.5, 4.1)
    arrow(8.7, 2.6, 9.5, 2.5)
    arrow(11.3, 2.5, 12.5, 2.5)

    # Title
    ax.text(8, 5.5, 'ZSAD: Zero-Shot Anomaly Detection Architecture', 
            ha='center', fontsize=16, fontweight='bold')

    path = OUTPUT_DIR / 'fig7_architecture.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("ZSAD Paper Figures Generator")
    print("=" * 60)

    model = ZSAD(device='cuda', pca_dim=64, weights=(0.5, 0.3, 0.2), k_knn=80, block_indices=(11,))
    print("Model loaded.\n")

    fig_architecture()
    fig_heatmap_grid(model)
    fig_overlay_comparison(model)
    fig_fabric_samples(model)
    fig_score_distributions(model)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
