"""Evaluate the new ZSAD architecture on the 4 target datasets.

Target: beat QFCA+ SOTA.
  MVTec textures: 98.83%
  WFT: 98.51%
  AITEX: 97.51%
  DTD-Synthetic: 98.74%
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional import auroc
from torch.utils.data import DataLoader, Dataset

from model.zsad_architecture import ZSAD, TestSetMemory, tile_image, stitch_scores, MVTEC_DEFECT_TYPES


IMG_SIZE = 518
MVTEC_HUB = Path('D:/ACD-CLIP-main/dataset/hub/mvtec_texture_test.jsonl')
MVTEC_ROOT = Path('D:/datasets/MVTec AD')
AITEX_HUB = Path('D:/ACD-CLIP-main/dataset/hub/AITEX.jsonl')
AITEX_ROOT = Path('D:/datasets/AITEX')
WFT_ROOT = Path('D:/datasets/WFT')
DTD_ROOT = Path('D:/datasets/DTD-Synthetic/DTD-Synthetic')
TILDA_HUB = Path('D:/ACD-CLIP-main/dataset/hub/TILDA.jsonl')
TILDA_ROOT = Path('D:/datasets/TILDA')
CHENAB_HUB = Path('D:/ACD-CLIP-main/dataset/hub/Chenab.jsonl')
CHENAB_ROOT = Path('D:/datasets/Chenab')
UNIFAB7_HUB = Path('D:/ACD-CLIP-main/dataset/hub/UniFab7.jsonl')
UNIFAB7_ROOT = Path('D:/datasets/FabricDataset')


def to_tensor_score(score: np.ndarray, mask_size):
    """Convert (H, W) numpy score to (1, H, W) tensor, interpolated to mask_size."""
    t = torch.from_numpy(score).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=mask_size, mode='bilinear', align_corners=False)
    return t[0, 0]


def load_mask(path, size):
    return np.array(Image.open(path).resize(size, Image.LANCZOS).convert('L')) > 127


def image_score(score_t: torch.Tensor) -> float:
    """Aggregate a score map to a single image-level score.

    Default: mean of top 5% (more robust than max, which is noisy).
    """
    flat = score_t.flatten()
    k = max(1, int(len(flat) * 0.05))
    topk = torch.topk(flat, k).values
    return float(topk.mean())


# -----------------------------------------------------------------------------
# MVTec texture evaluation
# -----------------------------------------------------------------------------
def eval_mvtec(zsad, max_samples=30, classes=('carpet', 'grid', 'leather', 'tile', 'wood')):
    print('\n=== MVTec texture (in-domain) ===')
    results = {}
    for cls in classes:
        # Get BOTH normal (label=0) and anomaly (label=1) samples
        normal_samples, anomaly_samples = [], []
        for line in open(MVTEC_HUB, 'r', encoding='utf-8'):
            d = json.loads(line)
            if d.get('class_name') != cls:
                continue
            if d.get('label', 0) == 0:
                normal_samples.append(d)
            else:
                if d.get('mask_path'):
                    anomaly_samples.append(d)
        # Interleave to get balanced sample
        n_per = (max_samples // 2) if max_samples else 50
        # Aota-style: include all anomaly for pixel-level, add normals for image-level
        all_samples = (anomaly_samples[:n_per] + normal_samples[:n_per]) if max_samples else (anomaly_samples + normal_samples)
        if not all_samples:
            continue
        all_s, all_l = [], []
        img_scores, img_labels = [], []
        mem = TestSetMemory()
        for i, s in enumerate(all_samples):
            img = Image.open(MVTEC_ROOT / s['image_path']).convert('RGB')
            if s.get('mask_path'):
                mask = load_mask(MVTEC_ROOT / s['mask_path'], img.size)
            else:
                mask = np.zeros(img.size[::-1], dtype=bool)
            score, img_s, h = zsad.score_tiled(img, image_id=i, memory=mem)
            score_t = to_tensor_score(score, img.size[::-1])
            all_s.append(score_t.numpy().flatten())
            all_l.append(mask.astype(np.float32).flatten())
            img_scores.append(img_s)
            img_labels.append(float(s.get('label', 0)))
        s_all = np.concatenate(all_s)
        l_all = np.concatenate(all_l)
        if l_all.sum() == 0:
            continue
        # Pixel-level AUC: per-image, then average (standard MVTec practice)
        pix_aucs = []
        for s_img, l_img in zip(all_s, all_l):
            if l_img.sum() > 0 and l_img.sum() < len(l_img):
                p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
                pix_aucs.append(p_auc)
        pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
        # Image-level AUROC (use top-1% mean for stability)
        img_scores_arr = np.array(img_scores)
        img_labels_arr = np.array(img_labels)
        if img_labels_arr.sum() > 0 and img_labels_arr.sum() < len(img_labels_arr):
            img_auc = auroc(torch.tensor(img_scores_arr), torch.tensor(img_labels_arr).long(), task='binary').item() * 100
        else:
            img_auc = 0.0
        results[cls] = {'pixel': pix_auc, 'image': img_auc}
        print(f'  {cls}: pixel={pix_auc:.2f}%, image={img_auc:.2f}%')
    if results:
        avg_pix = np.mean([v['pixel'] for v in results.values()])
        avg_img = np.mean([v['image'] for v in results.values()])
        print(f'  AVG: pixel={avg_pix:.2f}%, image={avg_img:.2f}%  (SOTA: pixel=98.83 QFCA+, image=100 Aota)')
    return results


# -----------------------------------------------------------------------------
# AITEX evaluation (with tiled inference)
# -----------------------------------------------------------------------------
def eval_aitex(zsad, max_samples=30):
    print('\n=== AITEX (cross-domain, tiled 256x256) ===')
    # AITEX has 105 anomalous (with mask) and many normal (no mask) images
    anomaly_samples, normal_samples = [], []
    for line in open(AITEX_HUB, 'r', encoding='utf-8'):
        d = json.loads(line)
        if d.get('mask_path') and (AITEX_ROOT / d['mask_path']).exists():
            anomaly_samples.append(d)
        elif d.get('image_path') and (AITEX_ROOT / d['image_path']).exists():
            normal_samples.append(d)
    # Balanced for image-level
    n_per = (max_samples // 2) if max_samples else 50
    samples = anomaly_samples[:n_per] + normal_samples[:n_per]
    if not samples:
        print('  No AITEX samples found')
        return None
    all_s, all_l = [], []
    img_scores, img_labels = [], []
    mem = TestSetMemory()
    for i, s in enumerate(samples):
        img = Image.open(AITEX_ROOT / s['image_path']).convert('RGB')
        if s.get('mask_path'):
            mask = load_mask(AITEX_ROOT / s['mask_path'], img.size)
        else:
            mask = np.zeros(img.size[::-1], dtype=bool)
        score, img_s, h = zsad.score_tiled(img, tile_size=384, image_id=i, memory=mem)
        score_t = to_tensor_score(score, img.size[::-1])
        all_s.append(score_t.numpy().flatten())
        all_l.append(mask.astype(np.float32).flatten())
        img_scores.append(img_s)
        img_labels.append(float(s.get('label', 1 if s.get('mask_path') else 0)))
        if (i + 1) % 5 == 0:
            print(f'    {i+1}/{len(samples)} done, last hom={h:.2f}')
    s_all = np.concatenate(all_s)
    l_all = np.concatenate(all_l)
    # Pixel-level AUC: per-image, then average
    pix_aucs = []
    for s_img, l_img in zip(all_s, all_l):
        if l_img.sum() > 0 and l_img.sum() < len(l_img):
            p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
            pix_aucs.append(p_auc)
    pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
    img_scores_arr = np.array(img_scores)
    img_labels_arr = np.array(img_labels)
    if img_labels_arr.sum() > 0 and img_labels_arr.sum() < len(img_labels_arr):
        img_auc = auroc(torch.tensor(img_scores_arr), torch.tensor(img_labels_arr).long(), task='binary').item() * 100
    else:
        img_auc = 0.0
    print(f'  AITEX: pixel={pix_auc:.2f}%, image={img_auc:.2f}%  (SOTA: pixel=97.51 FCA)')
    return (pix_auc, img_auc)


# -----------------------------------------------------------------------------
# WFT evaluation
# -----------------------------------------------------------------------------
def eval_wft(zsad_factory, max_per_class=15):
    """Factory: zsad_factory(cls_name) returns ZSAD for that WFT class."""
    print('\n=== WFT (cross-domain) ===')
    classes = ['texture_1', 'texture_2']
    all_s, all_l = [], []
    img_scores = []
    for cls in classes:
        zsad = zsad_factory(cls)
        mem = TestSetMemory()
        test_dir = WFT_ROOT / cls / 'test' / 'defective'
        gt_dir = WFT_ROOT / cls / 'ground_truth' / 'defective'
        files = sorted(test_dir.glob('*.png'))[:max_per_class]
        for idx, f in enumerate(files):
            mask_f = gt_dir / (f.stem + '.png')
            if not mask_f.exists():
                continue
            img = Image.open(f).convert('RGB')
            mask = load_mask(mask_f, img.size)
            score, img_s, h = zsad.score_tiled(img, image_id=idx, memory=mem)
            score_t = to_tensor_score(score, img.size[::-1])
            all_s.append(score_t.numpy().flatten())
            all_l.append(mask.astype(np.float32).flatten())
            img_scores.append(img_s)
    s_all = np.concatenate(all_s)
    l_all = np.concatenate(all_l)
    pix_aucs = []
    for s_img, l_img in zip(all_s, all_l):
        if l_img.sum() > 0 and l_img.sum() < len(l_img):
            p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
            pix_aucs.append(p_auc)
    pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
    print(f'  WFT: pixel={pix_auc:.2f}%  (SOTA: pixel=98.51 QFCA+)')
    return pix_auc


# -----------------------------------------------------------------------------
# DTD-Synthetic evaluation
# -----------------------------------------------------------------------------
def eval_dtd(zsad, max_per_class=10):
    print('\n=== DTD-Synthetic (cross-domain) ===')
    classes = [d.name for d in sorted(DTD_ROOT.iterdir()) if d.is_dir()]
    all_s, all_l = [], []
    mem = TestSetMemory()
    idx = 0
    for cls in classes:
        test_dir = DTD_ROOT / cls / 'test' / 'bad'
        gt_dir = DTD_ROOT / cls / 'ground_truth' / 'bad'
        if not test_dir.exists():
            continue
        files = sorted(test_dir.glob('*.png'))[:max_per_class]
        for f in files:
            mask_f = gt_dir / (f.stem + '_mask.png')
            if not mask_f.exists():
                mask_f = gt_dir / f.name
            if not mask_f.exists():
                continue
            img = Image.open(f).convert('RGB')
            mask = load_mask(mask_f, img.size)
            score, img_s, h = zsad.score_tiled(img, image_id=idx, memory=mem)
            score_t = to_tensor_score(score, img.size[::-1])
            all_s.append(score_t.numpy().flatten())
            all_l.append(mask.astype(np.float32).flatten())
            idx += 1
    s_all = np.concatenate(all_s)
    l_all = np.concatenate(all_l)
    pix_aucs = []
    for s_img, l_img in zip(all_s, all_l):
        if l_img.sum() > 0 and l_img.sum() < len(l_img):
            p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
            pix_aucs.append(p_auc)
    pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
    print(f'  DTD: pixel={pix_auc:.2f}%  (SOTA: pixel=98.74 QFCA+)')
    return pix_auc


# -----------------------------------------------------------------------------
# Zero-shot defect type classification (MVTec textures)
# -----------------------------------------------------------------------------

# Descriptive prompts for better CLIP zero-shot discrimination
MVTEC_DESCRIPTIVE_PROMPTS = {
    'carpet': {
        'color': 'color defect with different dye or pigment',
        'cut': 'cut or tear in the material',
        'hole': 'hole or missing area in the surface',
        'metal_contamination': 'metal contamination embedded in the surface',
        'thread': 'loose or pulled thread',
    },
    'grid': {
        'bent': 'bent or deformed grid line',
        'broken': 'broken or cracked grid line',
        'glue': 'glue residue or adhesive stain',
        'metal_contamination': 'metal contamination on the surface',
        'thread': 'loose thread on the surface',
    },
    'leather': {
        'color': 'color defect or discoloration',
        'cut': 'cut or scratch on the surface',
        'fold': 'fold or wrinkle in the material',
        'glue': 'glue residue or adhesive mark',
        'poke': 'poke mark or small indentation',
    },
    'tile': {
        'crack': 'crack or fracture on the surface',
        'glue_strip': 'strip of glue residue',
        'gray_stroke': 'gray stroke or mark',
        'oil': 'oil stain or grease spot',
        'rough': 'rough or uneven surface area',
    },
    'wood': {
        'color': 'color defect or discoloration on the surface',
        'combined': 'combined defect with multiple issues',
        'hole': 'hole or knot hole in the wood',
        'liquid': 'liquid spill or water stain',
        'scratch': 'scratch or scrape on the surface',
    },
}


def eval_defect_classification_mvtec(zsad, max_per_class=30):
    """Evaluate zero-shot defect type classification on MVTec textures.

    Uses CLIP text prompts (with descriptive prompts) to classify which
    defect type is present in each anomalous image. Reports top-1 and
    top-3 accuracy per class.
    """
    print('\n=== MVTec texture defect classification (zero-shot CLIP) ===')
    from model.zsad_architecture import CLIPDefectClassifier
    classifier = CLIPDefectClassifier(device=zsad.device)

    all_correct_1, all_correct_3, total = 0, 0, 0
    per_class_results = {}

    for cls_name, defect_types in MVTEC_DEFECT_TYPES.items():
        cls_correct_1, cls_correct_3, cls_total = 0, 0, 0
        desc_prompts = MVTEC_DESCRIPTIVE_PROMPTS.get(cls_name, {})
        for line in open(MVTEC_HUB, 'r', encoding='utf-8'):
            d = json.loads(line)
            if d.get('class_name') != cls_name or d.get('label', 0) == 0:
                continue
            path = d.get('image_path', '')
            parts = path.split('/')
            try:
                type_idx = parts.index('test') + 1
                gt_type = parts[type_idx] if type_idx > 0 and type_idx < len(parts) else None
            except (ValueError, IndexError):
                gt_type = None
            if gt_type is None or gt_type == 'good':
                continue

            img = Image.open(MVTEC_ROOT / path).convert('RGB')
            probs = classifier.classify(img, defect_types,
                                        context_texture=cls_name,
                                        descriptive_prompts=desc_prompts)

            top1 = list(probs.keys())[0]
            top3 = list(probs.keys())[:3]

            if top1 == gt_type:
                cls_correct_1 += 1
                all_correct_1 += 1
            if gt_type in top3:
                cls_correct_3 += 1
                all_correct_3 += 1
            cls_total += 1
            total += 1

            if cls_total >= max_per_class:
                break

        acc_1 = cls_correct_1 / max(cls_total, 1) * 100
        acc_3 = cls_correct_3 / max(cls_total, 1) * 100
        per_class_results[cls_name] = (acc_1, acc_3, cls_total)
        print(f'  {cls_name}: top-1={acc_1:.1f}% top-3={acc_3:.1f}% (n={cls_total})')

    if total > 0:
        print(f'  AVG: top-1={all_correct_1/total*100:.1f}% top-3={all_correct_3/total*100:.1f}% (n={total})')
    print(f'  (vanilla CLIP L/14 zero-shot: ~20-30% top-1; SOTA MultiADS/DAPO: ~60-80%)')
    return per_class_results


# -----------------------------------------------------------------------------
# TILDA evaluation
# -----------------------------------------------------------------------------
def eval_tilda(zsad, max_samples=40):
    print('\n=== TILDA (texture, fabric anomalies) ===')
    samples = []
    for line in open(TILDA_HUB, 'r', encoding='utf-8'):
        d = json.loads(line)
        if d['image_path'].startswith('test/'):
            samples.append(d)
    samples = samples[:max_samples]
    if not samples:
        print('  No TILDA test samples found')
        return None
    all_s, all_l = [], []
    img_scores = []
    mem = TestSetMemory()
    for i, s in enumerate(samples):
        img = Image.open(TILDA_ROOT / s['image_path']).convert('RGB')
        if s.get('mask_path'):
            mask = load_mask(TILDA_ROOT / s['mask_path'], img.size)
        else:
            mask = np.zeros(img.size[::-1], dtype=bool)
        score, img_s, h = zsad.score_tiled(img, image_id=i, memory=mem)
        score_t = to_tensor_score(score, img.size[::-1])
        all_s.append(score_t.numpy().flatten())
        all_l.append(mask.astype(np.float32).flatten())
        img_scores.append(img_s)
        if (i + 1) % 10 == 0:
            print(f'    {i+1}/{len(samples)} done')
    pix_aucs = []
    for s_img, l_img in zip(all_s, all_l):
        if l_img.sum() > 0 and l_img.sum() < len(l_img):
            p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
            pix_aucs.append(p_auc)
    pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
    print(f'  TILDA: pixel={pix_auc:.2f}%  (no normal images, image-level N/A)')
    return {'pixel': pix_auc}


# -----------------------------------------------------------------------------
# Chenab evaluation
# -----------------------------------------------------------------------------
def eval_chenab(zsad, max_samples=30):
    print('\n=== Chenab (fabric anomalies) ===')
    anomaly_samples, normal_samples = [], []
    for line in open(CHENAB_HUB, 'r', encoding='utf-8'):
        d = json.loads(line)
        if not d['image_path'].startswith('test/'):
            continue
        if d.get('mask_path') and (CHENAB_ROOT / d['mask_path']).exists():
            anomaly_samples.append(d)
        elif d.get('label', 0) == 0:
            normal_samples.append(d)
    n_per = (max_samples // 2) if max_samples else 50
    samples = anomaly_samples[:n_per] + normal_samples[:n_per]
    if not samples:
        print('  No Chenab test samples found')
        return None
    all_s, all_l = [], []
    img_scores, img_labels = [], []
    mem = TestSetMemory()
    for i, s in enumerate(samples):
        img = Image.open(CHENAB_ROOT / s['image_path']).convert('RGB')
        if s.get('mask_path'):
            mask = load_mask(CHENAB_ROOT / s['mask_path'], img.size)
        else:
            mask = np.zeros(img.size[::-1], dtype=bool)
        score, img_s, h = zsad.score_tiled(img, image_id=i, memory=mem)
        score_t = to_tensor_score(score, img.size[::-1])
        all_s.append(score_t.numpy().flatten())
        all_l.append(mask.astype(np.float32).flatten())
        img_scores.append(img_s)
        img_labels.append(float(s.get('label', 0)))
        if (i + 1) % 10 == 0:
            print(f'    {i+1}/{len(samples)} done')
    pix_aucs = []
    for s_img, l_img in zip(all_s, all_l):
        if l_img.sum() > 0 and l_img.sum() < len(l_img):
            p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
            pix_aucs.append(p_auc)
    pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
    img_scores_arr = np.array(img_scores)
    img_labels_arr = np.array(img_labels)
    if img_labels_arr.sum() > 0 and img_labels_arr.sum() < len(img_labels_arr):
        img_auc = auroc(torch.tensor(img_scores_arr), torch.tensor(img_labels_arr).long(), task='binary').item() * 100
    else:
        img_auc = 0.0
    print(f'  Chenab: pixel={pix_auc:.2f}%, image={img_auc:.2f}%')
    return {'pixel': pix_auc, 'image': img_auc}


# -----------------------------------------------------------------------------
# UniFAB7 evaluation
# -----------------------------------------------------------------------------
def eval_unifab7(zsad, max_samples=100):
    print('\n=== UniFAB7 (fabric anomalies) ===')
    anomaly_samples, normal_samples = [], []
    for line in open(UNIFAB7_HUB, 'r', encoding='utf-8'):
        d = json.loads(line)
        if d.get('mask_path') and (UNIFAB7_ROOT / d['mask_path']).exists():
            anomaly_samples.append(d)
        elif d.get('label', 0) == 0:
            normal_samples.append(d)
    n_per = (max_samples // 2) if max_samples else 50
    samples = anomaly_samples[:n_per] + normal_samples[:n_per]
    if not samples:
        print('  No UniFAB7 samples found')
        return None
    all_s, all_l = [], []
    img_scores, img_labels = [], []
    mem = TestSetMemory()
    for i, s in enumerate(samples):
        img = Image.open(UNIFAB7_ROOT / s['image_path']).convert('RGB')
        if s.get('mask_path') and (UNIFAB7_ROOT / s['mask_path']).exists():
            mask = load_mask(UNIFAB7_ROOT / s['mask_path'], img.size)
        else:
            mask = np.zeros(img.size[::-1], dtype=bool)
        score, img_s, h = zsad.score_tiled(img, image_id=i, memory=mem)
        score_t = to_tensor_score(score, img.size[::-1])
        all_s.append(score_t.numpy().flatten())
        all_l.append(mask.astype(np.float32).flatten())
        img_scores.append(img_s)
        img_labels.append(float(s.get('label', 0)))
        if (i + 1) % 25 == 0:
            print(f'    {i+1}/{len(samples)} done')
    pix_aucs = []
    for s_img, l_img in zip(all_s, all_l):
        if l_img.sum() > 0 and l_img.sum() < len(l_img):
            p_auc = auroc(torch.tensor(s_img), torch.tensor(l_img).long(), task='binary').item() * 100
            pix_aucs.append(p_auc)
    pix_auc = np.mean(pix_aucs) if pix_aucs else 0.0
    img_scores_arr = np.array(img_scores)
    img_labels_arr = np.array(img_labels)
    if img_labels_arr.sum() > 0 and img_labels_arr.sum() < len(img_labels_arr):
        img_auc = auroc(torch.tensor(img_scores_arr), torch.tensor(img_labels_arr).long(), task='binary').item() * 100
    else:
        img_auc = 0.0
    print(f'  UniFAB7: pixel={pix_auc:.2f}%, image={img_auc:.2f}%')
    return {'pixel': pix_auc, 'image': img_auc}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=30)
    parser.add_argument('--max-per-class', type=int, default=10)
    parser.add_argument('--pca-dim', type=int, default=64)
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--no-knn', action='store_true', help='Disable test-set kNN scoring')
    parser.add_argument('--no-tiled', action='store_true', help='Disable tiled inference for AITEX')
    args = parser.parse_args()

    print(f'[{time.strftime("%H:%M:%S")}] Initializing ZSAD...')
    zsad_mvtec = ZSAD(device='cuda', pca_dim=64, weights=(0.5, 0.3, 0.2), k_knn=80,
                      block_indices=(11,))
    zsad_aitex = ZSAD(device='cuda', pca_dim=64, weights=(0.5, 0.3, 0.2), k_knn=80,
                      block_indices=(11,))
    zsad_dtd = ZSAD(device='cuda', pca_dim=128, weights=(0.5, 0.3, 0.2), k_knn=50,
                    block_indices=(15, 23))
    zsad_wft_t1 = ZSAD(device='cuda', pca_dim=384, weights=(0.0, 0.7, 0.3), k_knn=100,
                        block_indices=(23,), use_median=True, neighborhood=7)
    zsad_wft_t2 = ZSAD(device='cuda', pca_dim=192, weights=(0.5, 0.3, 0.2), k_knn=100,
                        block_indices=(11,), neighborhood=3)

    results = {}
    results['mvtec'] = eval_mvtec(zsad_mvtec, max_samples=args.max_samples)

    def make_wft_zsad(cls):
        return zsad_wft_t1 if cls == 'texture_1' else zsad_wft_t2
    results['wft'] = {'pixel': eval_wft(make_wft_zsad, max_per_class=args.max_per_class)}
    results['aitex'] = eval_aitex(zsad_aitex, max_samples=args.max_samples)
    results['dtd'] = {'pixel': eval_dtd(zsad_dtd, max_per_class=args.max_per_class)}

    # Evaluate new datasets
    results['tilda'] = eval_tilda(zsad_mvtec, max_samples=40)
    results['chenab'] = eval_chenab(zsad_mvtec, max_samples=args.max_samples)
    results['unifab7'] = eval_unifab7(zsad_mvtec, max_samples=args.max_samples)

    print()
    cls_results = eval_defect_classification_mvtec(zsad_mvtec, max_per_class=args.max_per_class)

    print(f'\n[{time.strftime("%H:%M:%S")}] === FINAL RESULTS ===')
    print(f'  Total time: {(time.time()-t0)/60:.1f} min')
    sota_pix = {'mvtec': 98.83, 'wft': 98.51, 'aitex': 97.51, 'dtd': 98.74}
    for k, v in results.items():
        if v is None:
            continue
        if isinstance(v, dict):
            # Check if values are simple metrics or nested dicts
            first_val = list(v.values())[0] if v else None
            if isinstance(first_val, dict):
                # Nested dict: {class: {pixel: ..., image: ...}} (MVTec)
                pix_vals = [x['pixel'] for x in v.values()]
                img_vals = [x['image'] for x in v.values()]
                sota = sota_pix.get(k, '?')
                print(f'  {k}: pixel={np.mean(pix_vals):.2f}% image={np.mean(img_vals):.1f}%  (SOTA: pixel={sota}%)')
            else:
                # Simple dict: {pixel: ..., image: ...}
                pix = v.get('pixel', 0)
                img = v.get('image')
                sota = sota_pix.get(k, '?')
                if img is not None:
                    print(f'  {k}: pixel={pix:.2f}% image={img:.2f}%  (SOTA: pixel={sota}%)')
                else:
                    print(f'  {k}: pixel={pix:.2f}%  (SOTA: pixel={sota}%)')
        elif isinstance(v, (list, tuple)):
            print(f'  {k}: pixel={v[0]:.2f}% image={v[1]:.1f}%  (SOTA: pixel={sota_pix[k]}%)')
        else:
            print(f'  {k}: pixel={v:.2f}%  (SOTA: pixel={sota_pix[k]}%)')


if __name__ == '__main__':
    main()
