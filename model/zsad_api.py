"""
ZSAD Inference API — for front-end integration.

Minimal interface: one function that takes an image path, returns results as JSON.

Usage (from front-end):
    import zsad_api
    result = zsad_api.predict("path/to/image.png")
    # result = {
    #     "img_score": 0.234,
    #     "h_score": 0.89,
    #     "score_map": [[0.0, 0.1, ...], ...],   # H x W numpy list
    #     "defect_type": "cut"                     # only if CLIP loaded
    # }
"""

import sys, os, json
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.zsad_architecture import ZSAD

# Global model instance (lazy init on first call)
_ZSAD = None


def get_model(device=None):
    global _ZSAD
    if _ZSAD is None:
        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[ZSAD API] Initializing model on {device}...')
        _ZSAD = ZSAD(device=device, pca_dim=64, weights=(0.5, 0.3, 0.2),
                     k_knn=80, block_indices=(11,))
        print('[ZSAD API] Ready.')
    return _ZSAD


def predict(image_path, classify_defect=False, texture_name=None, defect_types=None):
    """Run ZSAD on a single image.

    Args:
        image_path: str path to image file.
        classify_defect: if True, also run CLIP defect type classifier.
        texture_name: e.g. 'carpet', 'grid' (for better CLIP prompts).
        defect_types: list of candidate defect types (defaults to MVTec types).

    Returns:
        dict with keys: img_score, h_score, score_map, (optionally) defect_probs
    """
    model = get_model()
    img = Image.open(image_path).convert('RGB')
    score_map, img_score, h_score = model.score(img)

    result = {
        'img_score': round(float(img_score), 4),
        'h_score': round(float(h_score), 4),
        'score_map': score_map.tolist(),  # H x W nested list
        'shape': list(score_map.shape),
    }

    if classify_defect:
        if defect_types is None:
            from model.zsad_architecture import MVTEC_DEFECT_TYPES
            defect_types = MVTEC_DEFECT_TYPES.get(texture_name, ['defect'])
        probs = model.classify_defect(img, defect_types, context_texture=texture_name)
        result['defect_probs'] = probs

    return result


def predict_batch(image_paths, **kwargs):
    """Run ZSAD on multiple images. Returns list of results."""
    return [predict(p, **kwargs) for p in image_paths]


if __name__ == '__main__':
    # Demo: run on a sample image
    import sys
    if len(sys.argv) < 2:
        print('Usage: python zsad_api.py <image_path> [--classify]')
        sys.exit(1)
    img_path = sys.argv[1]
    do_classify = '--classify' in sys.argv
    result = predict(img_path, classify_defect=do_classify, texture_name='carpet')
    print(json.dumps(result, indent=2))
