"""
Diagnosis 00: defect pixel % at ORIGINAL resolution (no transform).
Chỉ đọc mask gốc, không resize/crop gì cả.
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd


def defect_ratio(arr_binary):
    return arr_binary.sum() / arr_binary.size


def load_gt_raw(path):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return arr > 0


def diagnose(root_path):
    root_path = Path(root_path)
    categories = sorted([d for d in root_path.iterdir() if d.is_dir()])

    rows = []
    per_image_rows = []

    for cat_path in categories:
        cat = cat_path.name
        gt_dir = cat_path / "test_public" / "ground_truth" / "bad"
        if not gt_dir.exists():
            print(f"  [SKIP] {cat}: no ground_truth/bad folder")
            continue

        mask_files = sorted(
            list(gt_dir.glob("*.png")) +
            list(gt_dir.glob("*.jpg")) +
            list(gt_dir.glob("*.bmp"))
        )
        if not mask_files:
            print(f"  [SKIP] {cat}: no mask files")
            continue

        ratios = []
        sizes  = []

        for mf in mask_files:
            raw = load_gt_raw(mf)
            r   = defect_ratio(raw)
            ratios.append(r)
            sizes.append(raw.shape)

            per_image_rows.append({
                "category":      cat,
                "mask_file":     mf.name,
                "H":             raw.shape[0],
                "W":             raw.shape[1],
                "defect_px":     int(raw.sum()),
                "total_px":      raw.size,
                "defect_%":      round(r * 100, 4),
            })

        arr = np.array(ratios)
        rows.append({
            "category":    cat,
            "n_masks":     len(mask_files),
            "H":           sizes[0][0],
            "W":           sizes[0][1],
            "mean_%":      round(arr.mean() * 100, 4),
            "median_%":    round(np.median(arr) * 100, 4),
            "min_%":       round(arr.min() * 100, 4),
            "max_%":       round(arr.max() * 100, 4),
            "std_%":       round(arr.std() * 100, 4),
        })

        print(f"  [OK] {cat:15s} | {len(mask_files):3d} masks | "
              f"mean={arr.mean()*100:.3f}%  median={np.median(arr)*100:.3f}%  "
              f"min={arr.min()*100:.3f}%  max={arr.max()*100:.3f}%")

    df_cat = pd.DataFrame(rows)
    df_img = pd.DataFrame(per_image_rows)

    out_dir = Path("/mnt/user-data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_csv = out_dir / "diag00_raw_defect_per_category.csv"
    img_csv = out_dir / "diag00_raw_defect_per_image.csv"
    df_cat.to_csv(cat_csv, index=False)
    df_img.to_csv(img_csv, index=False)

    print(f"\n{'='*70}")
    print("CATEGORY SUMMARY (original resolution)")
    print('='*70)
    print(df_cat.to_string(index=False))
    print(f"\nSaved:\n  {cat_csv}\n  {img_csv}")
    return df_cat, df_img


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnosis_00_raw_defect_ratio.py /path/to/dataset_root")
        sys.exit(1)
    print(f"Diagnosing (raw resolution): {sys.argv[1]}\n")
    diagnose(sys.argv[1])