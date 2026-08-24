# eval_postop2.py
# -----------------------------------------------------------------------------
# OFFLINE (grid cache) — RING-GATED LOFILL, bản có van của P1.
#
# P1 (eval_postop) verdict: cơ chế ĐÚNG (fabric 0.3623→0.6905, điền 5/6 ảnh 000)
#   nhưng TRƯỢT vào nền: hố bóng wallplugs/sheet_metal bị điền bừa (012_shift_2 +858k px)
#   → Δmean +0.0001, 5 cat tụt. Phân biệt nằm ở VÀNH: vòng 000 fabric được phủ gần kín
#   bởi mask RULE (chỉ hở đoạn ngắn); "vòng" quanh hố bóng chỉ tồn tại ở ngưỡng thấp.
#
# P2 = P0 | {hố lo-threshold mà VÀNH quanh nó (dilation bán kính r — tái dùng r closing)
#            được phủ >= RING_COV bởi mask rule}
#   RING_COV = 0.5 (luật đa số) — ĐÓNG BĂNG TRƯỚC, KHÔNG SWEEP. Không hằng số nào khác.
#   FAIR: không GT, không per-cat.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy):
#   0) P0 control phải = 0.3748.
#   1) P2 VÀO NỀN nếu Δmean(P2-P0) >= +0.010 VÀ không cat tụt > 0.02.
#   2) Cơ chế van: fabric P2 >= 0.60 (giữ hầu hết lợi P1) VÀ wallplugs Δ >= -0.005
#      (van chặn hố bóng). Van hỏng một trong hai vế -> đọc lại, KHÔNG tune trong run.
#
#   python eval_postop2.py --data_path ../data --cache_dir ./fill --out_dir ./postop2
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

RING_COV = 0.5     # luật đa số — ĐÓNG BĂNG. KHÔNG SWEEP.
VARIANTS = ['P0', 'P2']


def gated_holes(holes_lo, pred, r):
    """Giữ hố lo-threshold có vành (dilation bán kính r) được phủ >= RING_COV bởi pred."""
    lab, n = cc_label(holes_lo)
    if n == 0:
        return np.zeros_like(holes_lo)
    out = np.zeros_like(holes_lo)
    H, W = holes_lo.shape
    for ci, s in enumerate(find_objects(lab), start=1):
        y0 = max(0, s[0].start - r - 1); y1 = min(H, s[0].stop + r + 1)
        x0 = max(0, s[1].start - r - 1); x1 = min(W, s[1].stop + r + 1)
        comp = lab[y0:y1, x0:x1] == ci
        ring = binary_dilation(comp, iterations=r) & ~comp
        if ring.any() and pred[y0:y1, x0:x1][ring].mean() >= RING_COV:
            out[y0:y1, x0:x1] |= comp
    return out


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
    p(f'  [{cat}] cache n={len(paths)} thr={thr:.3f}')

    mst = {v: np.zeros(3, np.float64) for v in VARIANTS}
    dbg = []
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
        P0 = binary_fill_holes(pred)
        holes_lo = binary_fill_holes(b_lo) & ~b_lo
        add = gated_holes(holes_lo, pred, r)
        P2 = P0 | add
        for v, pd in (('P0', P0), ('P2', P2)):
            mst[v] += ((pd & gt).sum(), (pd & ~gt).sum(), ((~pd) & gt).sum())
        if int(add.sum()) >= 10000:
            dbg.append((os.path.basename(paths[k]), 'bad' if labels[k] else 'GOOD',
                        int(add.sum()), int((add & gt).sum())))

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    p(f'    [{cat}] F1@rule P0={out["P0"]:.4f}  P2={out["P2"]:.4f}  Δ={out["P2"] - out["P0"]:+.4f}')
    for name, lb, a, atp in sorted(dbg, key=lambda x: -x[2])[:12]:
        p(f'      +hố({lb}) {name}: thêm={a // 1000}k (trúng GT {atp // 1000}k)')
    return out


def main():
    ap = argparse.ArgumentParser('eval_postop2: ring-gated lofill (P2) trên grid cache')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default='./fill')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['fabric', 'wallplugs', 'sheet_metal', 'can', 'fruit_jelly', 'vial', 'rice', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./postop2')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('postop2', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} P2 = P0 | hố-lo có vành phủ >= {RING_COV} bởi mask rule (dilation r). '
      f'Zero constant mới ngoài RING_COV=0.5 đóng băng. Không GT, không per-cat.')

    res = {}
    for cat in args.categories:
        if not os.path.exists(os.path.join(args.cache_dir, f'grids_{cat}.npz')):
            p(f'  [{cat}] KHÔNG có cache -> bỏ'); continue
        res[cat] = run_cat(cat, args, gk, device, p)
    if not res:
        p('không cache nào.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (FULL test_public, offline cache) =====')
    m0 = float(np.mean([res[c]['P0'] for c in res]))
    m2 = float(np.mean([res[c]['P2'] for c in res]))
    p(f'  F1@rule: P0={m0:.4f}  P2={m2:.4f}  Δ={m2 - m0:+.4f}')
    drops = [(c, round(res[c]['P2'] - res[c]['P0'], 4)) for c in res if res[c]['P2'] - res[c]['P0'] < -0.02]
    p(f'  cat tụt >0.02: {drops if drops else "KHÔNG"}')
    p('\nĐỌC (pre-registered): (0) P0=0.3748. (1) P2 VÀO nếu Δmean>=+0.010 VÀ không cat tụt>0.02. '
      '(2) Van: fabric P2>=0.60 VÀ wallplugs Δ>=-0.005. KHÔNG tune trong run.')


if __name__ == '__main__':
    main()
