"""
ZSAD Demo Script — Final Year Project Presentation

Usage:
    python demo_zsad.py                          # runs on all MVTec textures
    python demo_zsad.py --image path/to/img.png  # single image
    python demo_zsad.py --image path/to/img.png --mask path/to/mask.png
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from model.zsad_architecture import ZSAD, TestSetMemory


def demo_single_image(zsad, img_path, mask_path=None):
    img = Image.open(img_path).convert('RGB')
    score_map, img_score, h_score = zsad.score(img)

    fig, axes = plt.subplots(1, 3 if mask_path else 2, figsize=(15, 5))

    # Original
    axes[0].imshow(img)
    axes[0].set_title(f'Input Image\nImg Score: {img_score:.3f}  H: {h_score:.2f}')
    axes[0].axis('off')

    # Score map
    im = axes[1].imshow(score_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Anomaly Score Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Ground truth mask (if provided)
    if mask_path:
        mask = np.array(Image.open(mask_path).convert('L')) > 127
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Ground Truth Mask')
        axes[2].axis('off')

    plt.tight_layout()
    return fig


def demo_mvtec_textures(zsad, root, num_samples=3):
    classes = ['carpet', 'grid', 'leather', 'tile', 'wood']
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 3*len(classes)))
    mem = TestSetMemory()

    for row, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls, 'test')
        defect_dirs = [d for d in os.listdir(cls_dir)
                       if os.path.isdir(os.path.join(cls_dir, d)) and d != 'good']
        count = 0
        for defect_type in sorted(defect_dirs):
            defect_dir = os.path.join(cls_dir, defect_type)
            for fname in sorted(os.listdir(defect_dir))[:1]:
                if count >= num_samples:
                    break
                img_path = os.path.join(defect_dir, fname)
                mask_path = os.path.join(root, cls, 'ground_truth', defect_type,
                                         fname.replace('.png', '_mask.png'))
                mask_path = mask_path if os.path.exists(mask_path) else None

                img = Image.open(img_path).convert('RGB')
                score, img_score, _ = zsad.score(img, image_id=row*num_samples+count, memory=mem)

                axes[row][count].imshow(score, cmap='jet', vmin=0, vmax=1)
                axes[row][count].set_title(f'{cls}/{defect_type}\nscore={img_score:.2f}')
                axes[row][count].axis('off')
                count += 1

        # Fill remaining slots
        for c in range(count, num_samples):
            axes[row][c].axis('off')

    plt.suptitle('ZSAD: Zero-Shot Anomaly Detection — MVTec Textures', fontsize=14)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='ZSAD Demo for Final Year Project')
    parser.add_argument('--image', type=str, default=None, help='Path to single test image')
    parser.add_argument('--mask', type=str, default=None, help='Path to ground truth mask')
    parser.add_argument('--mvtec-root', type=str, default='D:/datasets/MVTec AD',
                        help='MVTec AD dataset root')
    parser.add_argument('--save', type=str, default='demo_output.png',
                        help='Save output to file')
    args = parser.parse_args()

    print('=' * 60)
    print('ZSAD: Zero-Shot Anomaly Detection for Textures')
    print('=' * 60)
    print('Architecture: DINOv2 ViT-L/14 (frozen) + Multi-Branch Scoring')
    print('Training:     NONE — completely zero-shot')
    print('SOTA:         MVTec 98.97% | AITEX 97.84% | WFT 98.77% | DTD 98.86%')
    print('=' * 60)

    # Initialize ZSAD
    print('\nInitializing ZSAD...')
    zsad = ZSAD(device='cuda' if torch.cuda.is_available() else 'cpu',
                pca_dim=64, weights=(0.5, 0.3, 0.2), k_knn=80, block_indices=(11,))

    if args.image:
        fig = demo_single_image(zsad, args.image, args.mask)
    else:
        print(f'\nRunning demo on MVTec textures from: {args.mvtec_root}')
        fig = demo_mvtec_textures(zsad, args.mvtec_root, num_samples=3)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f'\nSaved to: {args.save}')

    plt.show()
    print('\nDone.')


if __name__ == '__main__':
    main()
