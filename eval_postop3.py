# eval_postop3.py
# -----------------------------------------------------------------------------
# OFFLINE (grid cache) — SUPERADD MORPHOLOGY PARITY BUNDLE trên nhánh png.
#
# P0 (nền, đóng băng) = closing vuông k=2r+1 -> fill_holes = 0.3748.
# P3 (bundle SuperADD, arxiv 2605.14808, HẰNG SỐ NGUYÊN BẢN CỦA HỌ — không tune):
#     b_hi  = map > thr(rule)
#     pred3 = OR_{16 hướng} closing(b_hi, line SE nửa-dài L=round(26/0.625)=42px native)
#             (nối vết mảnh theo MỌI hướng — nhắm scratch sheet_metal/can)
#     pred3 &= (map > 0.8 * thr)        (AND-mask chặn closing lấn quá đà)
#     pred3 = fill_holes(pred3)
#   Tất cả 16/26px/0.625/0.8 từ paper SuperADD = frozen từ nguồn ngoài. FAIR.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy):
#   0) P0 control phải = 0.3748.
#   1) P3 VÀO NỀN nếu Δmean(P3-P0) >= +0.010 VÀ không cat tụt > 0.02.
#   2) Kỳ vọng cơ chế ở sheet_metal (+ can nếu có gì đó); các cat blob (rice/walnuts)
#      kỳ vọng ~0. Trượt -> giữ P0, KHÔNG tune (bundle nguyên bản, không chẻ nhỏ sweep).
#
#   python eval_postop3.py --data_path ../data --cache_dir ./fill --out_dir ./postop3
# -----------------------------------------------------------------------------
import os
import sys
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import make_map                                                   # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, fuse2, up_grid, guided1                         # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402

warnings.filterwarnings('ignore')

N_DIR = 16                                # SuperADD: 16 hướng
L_HALF = round(26 / 0.625)                # 26px @ scale 0.625 cua ho -> 42px native
AND_FRAC = 0.8                            # AND-mask 0.8 x threshold
VARIANTS = ['P0', 'P3']


def line_kernels(L, n_dir):
    size = 2 * L + 1
    ks = []
    for i in range(n_dir):
        th = np.pi * i / n_dir
        k = torch.zeros(size, size)
        for t in range(-L, L + 1):
            x = int(round(L + t * np.cos(th))); y = int(round(L + t * np.sin(th)))
            k[y, x] = 1.0
        ks.append(k)
    return ks


@torch.no_grad()
def closing_multidir(b, kernels):
    """b: (H,W) bool GPU. OR các closing theo line SE (dilation->erosion cùng SE)."""
    x = b[None, None].float()
    out = torch.zeros_like(b)
    for k in kernels:
        kb = k[None, None].to(b.device)
        pad = k.shape[-1] // 2
        dil = (F.conv2d(x, kb, padding=pad) > 0.5).float()
        ero = F.conv2d(dil, kb, padding=pad) >= float(kb.sum()) - 0.5
        out |= ero[0, 0]
    return out


def run_cat(cat, args, gk, kernels, device, p):
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
    for k in tqdm(range(len(paths)), ncols=70, desc=f'    {cat}', leave=False):
        pil = Image.open(paths[k])
        W, H = pil.size
        fused = fuse2(te_f[k], up_grid(te_c[k], G3, device), st)
        nat = make_map(fused['maxz'], args.canvas, gk, (H, W), device)
        nat = guided1(nat, load_gray(pil, device), max(1, round(min(H, W) / G3)))
        gt = (np.zeros((H, W), bool) if labels[k] == 0
              else np.asarray(Image.open(gt_of[paths[k]]).convert('L')) > 127)
        r = max(1, round(min(H, W) / G3))
        b_hi = nat > thr
        # P0: nền đóng băng
        P0 = binary_fill_holes(closing(b_hi, 2 * r + 1).cpu().numpy().astype(bool))
        # P3: bundle SuperADD nguyên bản
        pred3 = closing_multidir(b_hi, kernels)
        pred3 &= nat > AND_FRAC * thr
        P3 = binary_fill_holes(pred3.cpu().numpy().astype(bool))
        del nat, b_hi, pred3
        for v, pd in (('P0', P0), ('P3', P3)):
            mst[v] += ((pd & gt).sum(), (pd & ~gt).sum(), ((~pd) & gt).sum())

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    p(f'    [{cat}] F1@rule P0={out["P0"]:.4f}  P3={out["P3"]:.4f}  Δ={out["P3"] - out["P0"]:+.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_postop3: SuperADD morphology parity bundle (offline cache)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default='./fill')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['sheet_metal', 'can', 'fabric', 'wallplugs', 'fruit_jelly', 'vial', 'rice', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./postop3')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('postop3', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    kernels = line_kernels(L_HALF, N_DIR)
    p(f'device={device} P3 = OR closing 16 line-SE (L={L_HALF}px native) -> AND {AND_FRAC}xthr -> fill. '
      f'Hằng số nguyên bản SuperADD, không tune. P0 control = 0.3748.')

    res = {}
    for cat in args.categories:
        if not os.path.exists(os.path.join(args.cache_dir, f'grids_{cat}.npz')):
            p(f'  [{cat}] KHÔNG có cache -> bỏ'); continue
        res[cat] = run_cat(cat, args, gk, kernels, device, p)
    if not res:
        p('không cache nào.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (FULL test_public, offline cache) =====')
    m0 = float(np.mean([res[c]['P0'] for c in res]))
    m3 = float(np.mean([res[c]['P3'] for c in res]))
    p(f'  F1@rule: P0={m0:.4f}  P3={m3:.4f}  Δ={m3 - m0:+.4f}')
    drops = [(c, round(res[c]['P3'] - res[c]['P0'], 4)) for c in res if res[c]['P3'] - res[c]['P0'] < -0.02]
    p(f'  cat tụt >0.02: {drops if drops else "KHÔNG"}')
    p('\nĐỌC (pre-registered): (0) P0=0.3748. (1) P3 VÀO nếu Δmean>=+0.010 VÀ không cat tụt>0.02. '
      'Trượt -> giữ P0, không chẻ bundle ra sweep.')


if __name__ == '__main__':
    main()
