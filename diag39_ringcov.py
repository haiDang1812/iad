# diag39_ringcov.py
# -----------------------------------------------------------------------------
# DIAG (đo, KHÔNG chấm F1, không lever): phân bố coverage vành của hố lo-threshold.
#
# Bối cảnh: P1 (lofill không van) fabric +0.33 nhưng hố bóng phá 5 cat; P2 (van
#   cov>=0.5 @ bán kính r) chặn TẤT CẢ kể cả fabric -> vành sát hố là mép trong yếu
#   nhất, đỉnh viền rule nằm xa hơn. Câu hỏi: ở bán kính nào (r/2r/3r) coverage của
#   hố-THẬT (giao GT) tách khỏi hố-RỞM (không giao GT), và vực có đủ rộng không.
#
# GT ở đây CHỈ dùng dán nhãn chẩn đoán hố thật/rởm — không vào bất kỳ rule nào.
# ĐỌC: nếu tồn tại (bán kính, ngưỡng) mà hố-thật >= 5x hố-rởm với lề rộng -> MỘT
#   biến thể gate cuối cùng, đóng băng theo cơ chế (tiền lệ fairthr đo-một-lần).
#   Không có vực -> ĐÓNG dòng lofill, giữ P0.
#
#   python diag39_ringcov.py --data_path ../data --cache_dir ./fill \
#       --categories fabric wallplugs sheet_metal vial
# -----------------------------------------------------------------------------
import os
import sys
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, binary_dilation, label as cc_label, find_objects

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import make_map                                                   # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, RULE_G, fuse2, up_grid, guided1                 # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402

warnings.filterwarnings('ignore')

MIN_HOLE = 10000     # chỉ đo hố >= 10k px (hố nhỏ không đáng kể với pooled F1)


def run_cat(cat, args, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    z = np.load(os.path.join(args.cache_dir, f'grids_{cat}.npz'), allow_pickle=True)
    meta = np.load(os.path.join(args.cache_dir, f'meta_{cat}.npz'))
    st = [(float(meta['st'][i][0]), float(meta['st'][i][1])) for i in range(2)]
    thr = float(meta['thr'])
    te_f, te_c = z['te_fine'], z['te_ctx']
    paths = [str(x) for x in z['paths']]
    labels = [int(x) for x in z['labels']]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    gt_of = dict(zip(ds.img_paths, ds.gt_paths))

    rows = []          # (name, is_real, size, cov_r, cov_2r, cov_3r)
    for k in tqdm(range(len(paths)), ncols=70, desc=f'    {cat}', leave=False):
        pil = Image.open(paths[k])
        W, H = pil.size
        fused = fuse2(te_f[k], up_grid(te_c[k], G3, device), st)
        nat = make_map(fused['maxz'], args.canvas, gk, (H, W), device)
        nat = guided1(nat, load_gray(pil, device), max(1, round(min(H, W) / G3)))
        gt = (np.zeros((H, W), bool) if labels[k] == 0
              else np.asarray(Image.open(gt_of[paths[k]]).convert('L')) > 127)
        r = max(1, round(min(H, W) / G3))
        kk = 2 * r + 1
        pred = closing(nat > thr, kk).cpu().numpy().astype(bool)
        b_lo = closing(nat > thr / RULE_G, kk).cpu().numpy().astype(bool)
        del nat
        holes_lo = binary_fill_holes(b_lo) & ~b_lo
        lab, n = cc_label(holes_lo)
        if n == 0:
            continue
        for ci, s in enumerate(find_objects(lab), start=1):
            y0 = max(0, s[0].start - 3 * r - 1); y1 = min(H, s[0].stop + 3 * r + 1)
            x0 = max(0, s[1].start - 3 * r - 1); x1 = min(W, s[1].stop + 3 * r + 1)
            comp = lab[y0:y1, x0:x1] == ci
            sz = int(comp.sum())
            if sz < MIN_HOLE:
                continue
            pd = pred[y0:y1, x0:x1]
            covs = []
            cur = comp
            for _ in range(3):
                nxt = binary_dilation(cur, iterations=r)
                ring = nxt & ~cur          # vành ở lớp bán kính này
                covs.append(float(pd[ring].mean()) if ring.any() else 0.0)
                cur = nxt
            is_real = gt[y0:y1, x0:x1][comp].mean() >= 0.5
            rows.append((os.path.basename(paths[k]), bool(is_real), sz, *covs))

    if not rows:
        p(f'  [{cat}] không hố >= {MIN_HOLE}px'); return
    p(f'  [{cat}] {len(rows)} hố >= {MIN_HOLE}px:')
    for name, real, sz, c1, c2, c3 in sorted(rows, key=lambda x: -x[2])[:20]:
        p(f'    {"THẬT" if real else "rởm "} {name}: {sz // 1000}k  cov r={c1:.3f}  2r={c2:.3f}  3r={c3:.3f}')
    for tag, sel in (('THẬT', [x for x in rows if x[1]]), ('rởm ', [x for x in rows if not x[1]])):
        if sel:
            p(f'    == {tag}: n={len(sel)}  mean cov r={np.mean([x[3] for x in sel]):.3f}  '
              f'2r={np.mean([x[4] for x in sel]):.3f}  3r={np.mean([x[5] for x in sel]):.3f}')


def main():
    ap = argparse.ArgumentParser('diag39: phân bố coverage vành hố lo (đo, không F1)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default='./fill')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--categories', type=str, nargs='+', default=['fabric', 'wallplugs', 'sheet_metal', 'vial'])
    ap.add_argument('--out_dir', type=str, default='./diag39')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag39', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} DIAG coverage vành hố lo @ r/2r/3r (GT chỉ dán nhãn thật/rởm). '
      f'ĐỌC: cần vực >=5x với lề rộng thì mới có gate v3 đóng băng; không thì ĐÓNG dòng lofill.')
    for cat in args.categories:
        if os.path.exists(os.path.join(args.cache_dir, f'grids_{cat}.npz')):
            run_cat(cat, args, gk, device, p)


if __name__ == '__main__':
    main()
