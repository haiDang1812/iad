# eval_postop.py
# -----------------------------------------------------------------------------
# OFFLINE POST-OP trên GRID CACHE (./fill/grids_*.npz) — không bank, không backbone,
# không GPU-scoring. Chạy ~40 phút cho 8 cat.
#
# BỐI CẢNH: fill-holes VÀO NỀN (Δmean +0.0214, png F1@rule = 0.3748) nhưng vế cơ chế
#   trượt: fabric +0.1618 < +0.30 kỳ vọng → viền miếng vá 000 HỞ ở ngưỡng rule trên
#   một phần ảnh → fill bó tay các ảnh đó (fill-holes cần vòng kín 100%).
#
# THAY ĐỔI DUY NHẤT (P1 = lofill): điền thêm vùng kín ở NGƯỠNG NỀN p95 (= thr/1.15,
#   chính thành phần của rule đóng băng, gain 1.0 — ZERO hằng số mới):
#     b_lo   = closing(map > thr/RULE_G)          (viền dày hơn → dễ khép kín hơn)
#     holes  = fill(b_lo) & ~b_lo                 (ruột được bao kín ở ngưỡng nền)
#     P1     = fill(pred_rule) | holes
#   Rủi ro đã khai: holes ở ảnh good (vùng bị bao ở ngưỡng thấp) → FP; tiêu chí
#   không-cat-tụt sẽ bắt. FAIR, không per-cat, không GT.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy):
#   0) P0 (control) phải tái lập đúng gfill 0.3748 (sanity đường cache).
#   1) P1 VÀO NỀN nếu Δmean(P1-P0) >= +0.010 VÀ không cat tụt > 0.02.
#   2) Debug fabric: kỳ vọng P1 điền ruột >= 5/6 ảnh 000. Điền đủ mà F1 không lên
#      tương xứng -> phần thiếu là FP viền ngoài, không phải FN ruột. KHÔNG tune.
#
#   python eval_postop.py --data_path ../data --cache_dir ./fill --out_dir ./postop
# -----------------------------------------------------------------------------
import os
import sys
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import make_map                                                   # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, RULE_G, fuse2, up_grid, guided1                 # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402

warnings.filterwarnings('ignore')

VARIANTS = ['P0', 'P1']


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
        if labels[k] == 0:
            gt = np.zeros((H, W), bool)
        else:
            gt = np.asarray(Image.open(gt_of[paths[k]]).convert('L')) > 127
        r = max(1, round(min(H, W) / G3))
        kk = 2 * r + 1
        pred = closing(nat > thr, kk).cpu().numpy().astype(bool)
        b_lo = closing(nat > thr / RULE_G, kk).cpu().numpy().astype(bool)
        del nat
        P0 = binary_fill_holes(pred)
        holes_lo = binary_fill_holes(b_lo) & ~b_lo
        P1 = P0 | holes_lo
        for v, pd in (('P0', P0), ('P1', P1)):
            mst[v] += ((pd & gt).sum(), (pd & ~gt).sum(), ((~pd) & gt).sum())
        if labels[k] == 1 and gt.sum() >= 20000:
            dbg.append((os.path.basename(paths[k]), int(gt.sum()),
                        int((P0 & ~pred).sum()), int((P1 & ~P0).sum())))

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    p(f'    [{cat}] F1@rule P0={out["P0"]:.4f}  P1={out["P1"]:.4f}  Δ={out["P1"] - out["P0"]:+.4f}')
    for name, area, f0, f1x in dbg:
        p(f'      {name}: GT={area // 1000}k  điền_P0={f0 // 1000}k  thêm_P1={f1x // 1000}k')
    return out


def main():
    ap = argparse.ArgumentParser('eval_postop: OFFLINE lofill trên grid cache (P0=gfill control, P1=+holes@p95)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default='./fill')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['fabric', 'can', 'sheet_metal', 'fruit_jelly', 'vial', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./postop')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('postop', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} OFFLINE post-op tu cache {args.cache_dir}. P0=fill@rule (control 0.3748), '
      f'P1=P0 | holes@p95(gain1.0). Zero hằng số mới, không GT, không per-cat.')

    res = {}
    for cat in args.categories:
        if not os.path.exists(os.path.join(args.cache_dir, f'grids_{cat}.npz')):
            p(f'  [{cat}] KHÔNG có cache -> bỏ'); continue
        res[cat] = run_cat(cat, args, gk, device, p)
    if not res:
        p('không cache nào.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (FULL test_public, offline cache) =====')
    m0 = float(np.mean([res[c]['P0'] for c in res]))
    m1 = float(np.mean([res[c]['P1'] for c in res]))
    p(f'  F1@rule: P0={m0:.4f}  P1={m1:.4f}  Δ={m1 - m0:+.4f}')
    drops = [(c, round(res[c]['P1'] - res[c]['P0'], 4)) for c in res if res[c]['P1'] - res[c]['P0'] < -0.02]
    p(f'  cat tụt >0.02: {drops if drops else "KHÔNG"}')
    p('\nĐỌC (pre-registered): (0) P0 phải = 0.3748. (1) P1 VÀO nếu Δmean >= +0.010 VÀ không cat tụt >0.02. '
      '(2) fabric debug: P1 điền >= 5/6 ảnh 000? KHÔNG tune trong run.')


if __name__ == '__main__':
    main()
