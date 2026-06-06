"""
Diagnosis script: defect pixel % before and after INP-Former transform
INP-Former pipeline (from dataset.py):
  gt_transforms = Compose([
      Resize((size, size)),       # size=448, BILINEAR by default
      CenterCrop(isize),          # isize=392
      ToTensor()                  # scales [0,255] -> [0,1]
  ])
  
White pixel (>0) = defect in ground truth mask.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from pathlib import Path

# ── INP-Former transform ──────────────────────────────────────────────────────
RESIZE = 448
CROP   = 392

gt_transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),   # PIL default: BILINEAR
    transforms.CenterCrop(CROP),
    transforms.ToTensor()
])

# ── helpers ───────────────────────────────────────────────────────────────────
def defect_ratio(arr_binary):
    """Fraction of defect pixels (1s) in a boolean/binary array."""
    return arr_binary.sum() / arr_binary.size

def load_gt_raw(path):
    """Load ground truth mask as binary numpy array (original size)."""
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return arr > 0  # white=defect

def load_gt_transformed(path):
    """Apply INP-Former gt_transform and return binary numpy."""
    img = Image.open(path).convert('L')
    t = gt_transform(img)               # [1, 392, 392], float [0,1]
    arr = t.squeeze(0).numpy()
    return arr > 0

# ── main ──────────────────────────────────────────────────────────────────────
def diagnose(root_path):
    root_path = Path(root_path)
    categories = sorted([d for d in root_path.iterdir() if d.is_dir()])

    rows = []
    per_image_rows = []

    for cat_path in categories:
        cat = cat_path.name
        gt_dir = cat_path / "test_public" / "ground_truth" / "bad"
        if not gt_dir.exists():
            print(f"  [SKIP] {cat}: no ground_truth/bad folder at {gt_dir}")
            continue

        mask_files = sorted(
            list(gt_dir.glob("*.png")) +
            list(gt_dir.glob("*.jpg")) +
            list(gt_dir.glob("*.bmp"))
        )
        if not mask_files:
            print(f"  [SKIP] {cat}: no mask files found in {gt_dir}")
            continue

        raw_ratios, tfm_ratios = [], []
        orig_sizes = []

        for mf in mask_files:
            raw = load_gt_raw(mf)
            tfm = load_gt_transformed(mf)

            orig_sizes.append(raw.shape)
            raw_ratios.append(defect_ratio(raw))
            tfm_ratios.append(defect_ratio(tfm))

            per_image_rows.append({
                "category":        cat,
                "mask_file":       mf.name,
                "orig_H":          raw.shape[0],
                "orig_W":          raw.shape[1],
                "raw_defect_%":    round(defect_ratio(raw) * 100, 4),
                "tfm_defect_%":    round(defect_ratio(tfm) * 100, 4),
                "raw_px_count":    int(raw.sum()),
                "tfm_px_count":    int(tfm.sum()),
                "signal_ratio":    round(defect_ratio(tfm) / defect_ratio(raw), 4) if defect_ratio(raw) > 0 else float('nan'),
            })

        raw_arr = np.array(raw_ratios)
        tfm_arr = np.array(tfm_ratios)

        rows.append({
            "category":             cat,
            "n_masks":              len(mask_files),
            "orig_H":               orig_sizes[0][0],
            "orig_W":               orig_sizes[0][1],
            # raw stats
            "raw_mean_%":           round(raw_arr.mean() * 100, 4),
            "raw_median_%":         round(np.median(raw_arr) * 100, 4),
            "raw_min_%":            round(raw_arr.min() * 100, 4),
            "raw_max_%":            round(raw_arr.max() * 100, 4),
            # after INP-Former transform
            "tfm_mean_%":           round(tfm_arr.mean() * 100, 4),
            "tfm_median_%":         round(np.median(tfm_arr) * 100, 4),
            "tfm_min_%":            round(tfm_arr.min() * 100, 4),
            "tfm_max_%":            round(tfm_arr.max() * 100, 4),
            # signal preservation
            "signal_preserved_%":   round((tfm_arr.mean() / raw_arr.mean() * 100) if raw_arr.mean() > 0 else float('nan'), 2),
        })

        print(f"  [OK] {cat}: {len(mask_files)} masks | raw mean={raw_arr.mean()*100:.3f}% | tfm mean={tfm_arr.mean()*100:.3f}%")

    # ── save CSVs ──────────────────────────────────────────────────────────────
    df_cat = pd.DataFrame(rows)
    df_img = pd.DataFrame(per_image_rows)

    out_dir = Path("/mnt/user-data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_csv = out_dir / "defect_diagnosis_per_category.csv"
    img_csv = out_dir / "defect_diagnosis_per_image.csv"

    df_cat.to_csv(cat_csv, index=False)
    df_img.to_csv(img_csv, index=False)

    print(f"\n{'='*70}")
    print("CATEGORY SUMMARY")
    print('='*70)
    print(df_cat.to_string(index=False))
    print(f"\nSaved:\n  {cat_csv}\n  {img_csv}")
    return df_cat, df_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_defects.py /path/to/dataset_root")
        sys.exit(1)
    root = sys.argv[1]
    print(f"Diagnosing: {root}")
    print(f"Transform: Resize({RESIZE}) -> CenterCrop({CROP})\n")
    diagnose(root)