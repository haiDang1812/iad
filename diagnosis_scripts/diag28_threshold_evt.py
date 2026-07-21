# diag28_threshold_evt.py
# -----------------------------------------------------------------------------
# TIẾP diag27. diag27 chốt: map ĐỦ TỐT (oracle 0.597), thủ phạm là k=3 quá nhỏ;
#   mean+kσ trên phân bố ĐÍCH với k~4.5 (ksig_pool) lấy ~93% oracle (0.558). Nhưng:
#     - k là "magic number", không diễn giải được, không rõ transfer.
#     - k tối ưu co giãn theo prevalence (fruit_jelly 0.86% cần k nhỏ, rice 0.15% cần k lớn)
#       => 1 k GLOBAL vẫn là compromise.
#
# diag28 hỏi: có QUY TẮC NGUYÊN TẮC nào (a) núm diễn giải được, (b) tự thích ứng
#   per-category theo đuôi/prevalence, (c) bền với shift, mà (d) >= ksig_pool và tiến gần
#   oracle? -> đó là METHOD cho paper (Part-3), không chỉ "đặt k=4.5".
#
# Các họ (đều tune 1 SIÊU-THAM-SỐ GLOBAL DUY NHẤT qua mọi cat, chấm SegF1 kiểu server):
#   prod        : val mean+3σ                        (baseline production)
#   ksig_pool*  : mean+kσ trên pooled ĐÍCH           (anchor diag27, ~0.558)
#   ksig_img*   : mean+kσ TỪNG ẢNH                    (thích ứng ánh sáng mỗi ảnh -> chống shift)
#   rz_img*     : median + k·MAD TỪNG ẢNH             (robust, miễn nhiễm pixel defect)
#   gpd_val*    : EVT/GPD (POT) fit trên val/good, cắt ở FPR α   (nguyên tắc, NGUỒN)
#   gpd_tgt*    : EVT/GPD fit trên pooled ĐÍCH, cắt ở FPR α       (nguyên tắc, self-cal)
#   oracle      : 1 thr tốt nhất/cat (cần GT)         (TRẦN)
#
# Điểm mấu chốt EVT: MỘT α (false-positive-rate mục tiêu, diễn giải được) -> ngưỡng
#   per-cat TỰ co giãn theo độ nặng đuôi normal-score. Đó là bản nguyên tắc của "chọn k".
#
#   python diag28_threshold_evt.py --data_path ../data --out_dir ./diag28
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, score_grid, gt_grid, up_to, list_split_files,
    VALID, SPLITS, IMG_EXT, SMOOTH_RES,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')


# --------------------------- chấm SegF1 kiểu server -------------------------
def pooled_f1(maps, gts, thr):
    """thr: scalar (dùng chung) HOẶC list per-image. Pool TP/FP/FN qua mọi ảnh."""
    if np.isscalar(thr):
        thr = [thr] * len(maps)
    TP = FP = FN = 0.0
    for m, g, t in zip(maps, gts, thr):
        pred = m >= t
        gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
    return 2 * TP / (2 * TP + FP + FN + 1e-9)


def oracle_pool_f1(maps, gts):
    P = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    G = np.concatenate([g.reshape(-1) for g in gts]).astype(bool)
    npos = int(G.sum())
    if npos == 0:
        return float('nan')
    order = np.argsort(-P, kind='stable')
    tp = np.cumsum(G[order], dtype=np.int64)
    k = np.arange(1, P.size + 1, dtype=np.int64)
    return float((2.0 * tp / (k + npos)).max())


# ------------------------- EVT / GPD (POT) ---------------------------------
def gpd_threshold(x, alpha, u_q=0.90, trim_q=0.995):
    """Peaks-Over-Threshold: fit GPD (moment-matching) cho đuôi normal-score, trả
    ngưỡng t sao cho P(normal > t) = alpha. u = quantile u_q; trim > trim_q để loại
    pixel defect nhiễm đuôi (prevalence < 1% nên vùng (u, trim] gần như thuần normal)."""
    x = x.reshape(-1).astype(np.float64)
    u = float(np.quantile(x, u_q))
    hi = float(np.quantile(x, trim_q))
    exc = x[(x > u) & (x <= hi)] - u
    zeta = float((x > u).mean())                       # P(X > u) ~ khối đuôi
    if exc.size < 20 or zeta <= 0 or alpha >= zeta:
        return float(np.quantile(x, 1.0 - alpha))      # fallback: quantile kinh nghiệm
    m = float(exc.mean()); s2 = float(exc.var())
    if s2 < 1e-12:
        return u
    xi = 0.5 * (1.0 - m * m / s2)                       # moment-matching shape
    xi = float(np.clip(xi, -0.5, 0.5))
    sig = m * (1.0 - xi)
    if sig <= 0:
        return float(np.quantile(x, 1.0 - alpha))
    ratio = alpha / zeta
    if abs(xi) < 1e-6:
        t = u - sig * np.log(ratio)                     # giới hạn mũ (xi->0)
    else:
        t = u + (sig / xi) * (ratio ** (-xi) - 1.0)
    return float(t)


def run_cat(bb, cat, args, layers, gk, device):
    # KHÔNG @torch.no_grad(): build_head phải train (loss.backward()).
    T, gt = args.tiles, args.grid_tile
    hw = args.head_w
    rng_np = np.random.default_rng(args.seed)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr, T, gt * bb.patch, gt, layers, args.enc_batch, args.bank_size, device)

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng_np.shuffle(bad)
    shot_pool = bad[:args.shots]
    head = build_head(bb, ds, shot_pool, bank, args, layers, device)
    if head is None:
        return None

    priv_raw = {}
    for split in SPLITS:
        files, _ = list_split_files(args.data_path, cat, split)
        if not files:
            continue
        if args.max_priv:
            files = files[:args.max_priv]
        priv_raw[split] = [score_grid(bb, Image.open(f), bank, head, args, layers, device)
                           for f in tqdm(files, ncols=70, desc=f'    {cat}/{split}', leave=False)]
    if not priv_raw:
        return None
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for v in priv_raw.values() for d, _ in v]), [1, 99])

    def up(a):
        return up_to(a, (SMOOTH_RES, SMOOTH_RES), gk, device)

    def fuse(d, pr):
        return (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr

    val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e)) for e in IMG_EXT], []))
    if args.max_val:
        val_imgs = val_imgs[:args.max_val]
    val_scores = np.concatenate([up(fuse(*score_grid(bb, Image.open(v), bank, head, args, layers, device))).reshape(-1)
                                 for v in tqdm(val_imgs, ncols=70, desc=f'    {cat}/val', leave=False)]).astype(np.float32)

    shots = set(shot_pool)
    pub_idx = [i for i in bad if i not in shots][:args.max_eval]
    pub_idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    maps, gts = [], []
    for i in tqdm(pub_idx, ncols=70, desc=f'    {cat}/pub', leave=False):
        maps.append(up(fuse(*score_grid(bb, Image.open(ds.img_paths[i]), bank, head, args, layers, device)))
                    .astype(np.float32))
        gts.append(gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8))
    pooled = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    prevalence = float(np.concatenate([g.reshape(-1) for g in gts]).astype(bool).mean())

    # thống kê per-image (tính 1 lần, vector hóa theo k)
    im_mean = np.array([m.mean() for m in maps]); im_std = np.array([m.std() for m in maps])
    im_med = np.array([np.median(m) for m in maps])
    im_mad = np.array([1.4826 * np.median(np.abs(m - md)) for m, md in zip(maps, im_med)])
    pm, ps = float(pooled.mean()), float(pooled.std())

    fixed = {
        'prod':   pooled_f1(maps, gts, float(val_scores.mean() + 3.0 * val_scores.std())),
        'oracle': oracle_pool_f1(maps, gts),
    }
    grids = {
        'ksig_pool': [pooled_f1(maps, gts, pm + k * ps) for k in args.k_grid],
        'ksig_img':  [pooled_f1(maps, gts, list(im_mean + k * im_std)) for k in args.k_grid],
        'rz_img':    [pooled_f1(maps, gts, list(im_med + k * im_mad)) for k in args.k_grid],
        'gpd_val':   [pooled_f1(maps, gts, gpd_threshold(val_scores, a)) for a in args.a_grid],
        'gpd_tgt':   [pooled_f1(maps, gts, gpd_threshold(pooled, a)) for a in args.a_grid],
    }
    return {'prevalence': prevalence, 'fixed': fixed, 'grids': grids}


def main():
    ap = argparse.ArgumentParser('diag28: ngưỡng nguyên tắc (EVT / per-image) vs magic-k')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--max_eval', type=int, default=30)
    ap.add_argument('--max_val', type=int, default=30)
    ap.add_argument('--max_priv', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag28')
    args = ap.parse_args()

    args.k_grid = [round(x, 2) for x in np.arange(0.5, 6.01, 0.25)]
    args.a_grid = [float(a) for a in np.geomspace(5e-2, 1e-4, 40)]          # target FPR

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag28', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} layers={layers} head_w={args.head_w}')
    p('So: per-image / EVT có (a) >= ksig_pool, (b) tiến gần oracle, (c) núm diễn giải được không?')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        p(f'  [{cat}] prevalence={r["prevalence"]*100:.3f}%  prod={r["fixed"]["prod"]:.4f}  '
          f'oracle={r["fixed"]["oracle"]:.4f}')
    if not res:
        return

    fam = ['ksig_pool', 'ksig_img', 'rz_img', 'gpd_val', 'gpd_tgt']
    grid_of = {'ksig_pool': args.k_grid, 'ksig_img': args.k_grid, 'rz_img': args.k_grid,
               'gpd_val': args.a_grid, 'gpd_tgt': args.a_grid}

    def tune(name):
        M = np.array([res[c]['grids'][name] for c in res])       # [n_cat, n_grid]
        j = int(np.argmax(M.mean(axis=0)))
        return float(M.mean(axis=0)[j]), j, grid_of[name][j]

    best = {f: tune(f) for f in fam}
    mean_fixed = lambda k: float(np.mean([res[c]['fixed'][k] for c in res]))     # noqa: E731

    p('\n' + '=' * 78 + '\n===== MEAN SegF1 qua category (siêu-tham-số tune GLOBAL) =====')
    p(f'  prod  (val mean+3σ)          = {mean_fixed("prod"):.4f}   <- baseline')
    p(f'  ksig_pool* (k={best["ksig_pool"][2]:.2f})          = {best["ksig_pool"][0]:.4f}   (anchor diag27)')
    p(f'  ksig_img*  (k={best["ksig_img"][2]:.2f})          = {best["ksig_img"][0]:.4f}   (per-image)')
    p(f'  rz_img*    (k={best["rz_img"][2]:.2f})          = {best["rz_img"][0]:.4f}   (per-image robust)')
    p(f'  gpd_val*   (α={best["gpd_val"][2]:.2e})     = {best["gpd_val"][0]:.4f}   (EVT nguồn)')
    p(f'  gpd_tgt*   (α={best["gpd_tgt"][2]:.2e})     = {best["gpd_tgt"][0]:.4f}   (EVT self-cal)')
    p(f'  oracle_pool (cần GT)          = {mean_fixed("oracle"):.4f}   <- TRẦN')

    # per-category cho phương pháp tốt nhất self-cal, ở siêu-tham-số GLOBAL đã chọn
    win = max(['ksig_img', 'rz_img', 'gpd_tgt'], key=lambda f: best[f][0])
    jw = best[win][1]
    p(f'\n--- per-category: {win}* (núm global) vs prod vs oracle ---')
    for c in res:
        p(f'  [{c:11s}] prev={res[c]["prevalence"]*100:6.3f}%  '
          f'prod={res[c]["fixed"]["prod"]:.4f}  {win}={res[c]["grids"][win][jw]:.4f}  '
          f'oracle={res[c]["fixed"]["oracle"]:.4f}')

    p('\nĐỌC:')
    p(' - gpd_tgt* >= ksig_pool* và núm là α (FPR, diễn giải/transfer được) => METHOD Part-3:')
    p('   1 α global -> ngưỡng per-cat tự co giãn theo đuôi. Ăn đứt "magic k".')
    p(' - ksig_img*/rz_img* > ksig_pool* => per-image (chống light-shift từng ảnh) là chìa khóa;')
    p('   rz_img (median+MAD) miễn nhiễm pixel defect => bền nhất khi có defect lớn.')
    p(' - per-cat: cat nào oracle vẫn thấp (can/sheet_metal) => MAP-limited, không phải ngưỡng')
    p('   => tách rõ phần cần cải thiện backbone/res khỏi phần đã bão hòa bởi ngưỡng.')


if __name__ == '__main__':
    main()
