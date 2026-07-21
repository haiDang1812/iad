# diag27_threshold_rules.py
# -----------------------------------------------------------------------------
# TIỀN ĐỀ (từ diag25): map FUSE của ta ĐỦ TỐT (oracle in-domain 0.597 ~ winner ~0.60),
#   nhưng quy tắc ngưỡng mean+3σ chỉ đạt 0.324 -> BỎ LẠI ~0.27 SegF1 chỉ vì chọn dao cắt,
#   độc lập với shift (mean+3σ in-domain 0.324 ≈ private đã submit 0.331). Và không có
#   quy tắc global đơn giản nào (thr-tgt) chạm oracle.
#
# CÂU HỎI: có QUY TẮC NGƯỠNG MÙ nào (tính trên CHÍNH phân bố đích, KHÔNG dùng GT)
#   tiến gần oracle in-domain không? Nếu có -> đó là đòn bẩy rẻ nhất + novelty Part-1.
#   Then-key: rule tính trên phân bố ĐÍCH tự dịch theo shift -> vá luôn cả cú trôi +2.79σ.
#
# So các họ quy tắc, trên FUSE map test_public (có GT chỉ để CHẤM, không để chọn thr):
#   prod        : mean+3σ trên validation/good   (baseline production, domain NGUỒN)
#   val_kσ*     : mean+kσ trên validation/good, k tối ưu GLOBAL 1 giá trị (trần họ NGUỒN)
#   test_kσ*    : mean+kσ trên pooled test scores, k global    (self-cal, trần họ ĐÍCH)
#   pct*        : cắt ở percentile p của pooled test scores, p global  (self-cal, trần)
#   otsu_pool   : Otsu trên pooled test scores/cat  (MÙ HOÀN TOÀN, 0 hyperparam, 0 GT)
#   otsu_img    : Otsu từng-ảnh (thr riêng mỗi ảnh) (MÙ HOÀN TOÀN, bền nhất với light-shift)
#   oracle_pool : 1 thr tốt nhất/cat (cần GT) = trần operating-point
#
#   (*) = 1 siêu-tham-số GLOBAL DUY NHẤT, tune chung qua MỌI category (không leak per-cat).
#   otsu_* = KHÔNG tham số. Cái ta thật sự muốn thắng là otsu_pool / otsu_img.
#
# SegF1 chấm KIỂU SERVER: mỗi ảnh nhị phân bằng thr của NÓ (per-image) hoặc thr chung
#   (per-cat), rồi pool TP/FP/FN qua mọi ảnh. F1 = 2ΣTP/(2ΣTP+ΣFP+ΣFN).
#
#   python diag27_threshold_rules.py --data_path ../data --out_dir ./diag27
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


# ------------------------- các quy tắc ngưỡng ------------------------------
def otsu(x, bins=256):
    """Ngưỡng Otsu 1D: cực đại phương sai liên-lớp trên histogram."""
    x = x.reshape(-1)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    wb = np.cumsum(hist)
    wf = hist.sum() - wb
    mu = np.cumsum(hist * centers)
    mtot = mu[-1]
    with np.errstate(invalid='ignore', divide='ignore'):
        mb = mu / wb
        mf = (mtot - mu) / wf
        vb = wb * wf * (mb - mf) ** 2
    vb[~np.isfinite(vb)] = -1.0
    return float(centers[int(np.argmax(vb))])


# --------- chấm SegF1 kiểu server: nhị phân từng ảnh rồi pool -------------
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
    """1 thr chung/cat tối ưu, cumulative-TP chính xác trên pixel gộp."""
    P = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    G = np.concatenate([g.reshape(-1) for g in gts]).astype(bool)
    npos = int(G.sum())
    if npos == 0:
        return float('nan')
    order = np.argsort(-P, kind='stable')
    tp = np.cumsum(G[order], dtype=np.int64)
    k = np.arange(1, P.size + 1, dtype=np.int64)
    return float((2.0 * tp / (k + npos)).max())


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

    # lo/hi từ private pooled (y hệt infer_submit) để fuse khớp submission
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

    # validation/good (nguồn ngưỡng production)
    val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e)) for e in IMG_EXT], []))
    if args.max_val:
        val_imgs = val_imgs[:args.max_val]
    val_scores = np.concatenate([up(fuse(*score_grid(bb, Image.open(v), bank, head, args, layers, device))).reshape(-1)
                                 for v in tqdm(val_imgs, ncols=70, desc=f'    {cat}/val', leave=False)]).astype(np.float32)

    # test_public (LOẠI shot chống leak) — giữ per-image map + gt
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

    # ---- các quy tắc KHÔNG-tham-số / cố định ----
    vmean, vstd = float(val_scores.mean()), float(val_scores.std())
    thr_prod = vmean + args.thr_sigma * vstd
    thr_otsu_pool = otsu(pooled)
    thr_otsu_img = [otsu(m) for m in maps]

    fixed = {
        'prod':        pooled_f1(maps, gts, thr_prod),
        'otsu_pool':   pooled_f1(maps, gts, thr_otsu_pool),
        'otsu_img':    pooled_f1(maps, gts, thr_otsu_img),
        'oracle_pool': oracle_pool_f1(maps, gts),
    }

    # ---- các họ có 1 siêu-tham-số: trả F1 theo LƯỚI để main tune GLOBAL ----
    pmean, pstd = float(pooled.mean()), float(pooled.std())
    grids = {
        'val_ksig':  [pooled_f1(maps, gts, vmean + k * vstd) for k in args.k_grid],
        'test_ksig': [pooled_f1(maps, gts, pmean + k * pstd) for k in args.k_grid],
        'pct':       [pooled_f1(maps, gts, float(np.quantile(pooled, q))) for q in args.q_grid],
    }
    return {'prevalence': prevalence, 'fixed': fixed, 'grids': grids}


def main():
    ap = argparse.ArgumentParser('diag27: quy tắc ngưỡng mù nào tiến gần oracle?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64, help='PHẢI trùng submit (64)')
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
    ap.add_argument('--thr_sigma', type=float, default=3.0)
    ap.add_argument('--max_eval', type=int, default=30)
    ap.add_argument('--max_val', type=int, default=30)
    ap.add_argument('--max_priv', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag27')
    args = ap.parse_args()

    args.k_grid = [round(x, 2) for x in np.arange(0.5, 6.01, 0.25)]
    args.q_grid = [round(x, 4) for x in (1.0 - np.geomspace(0.05, 1e-4, 40))]  # PPR 5%..0.01%

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag27', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} layers={layers} head_w={args.head_w}')
    p('MỤC TIÊU: quy tắc MÙ (otsu_*) hay self-cal (test_kσ*/pct*) có tiến gần oracle không?')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ (thiếu data)'); continue
        res[cat] = r
        f = r['fixed']
        p(f'\n  [{cat}] prevalence={r["prevalence"]*100:.3f}%  '
          f"prod={f['prod']:.4f}  otsu_pool={f['otsu_pool']:.4f}  "
          f"otsu_img={f['otsu_img']:.4f}  oracle={f['oracle_pool']:.4f}")
    if not res:
        return

    # ---- tune GLOBAL 1 siêu-tham-số cho từng họ: cực đại MEAN SegF1 qua cat ----
    def tune(name, grid_vals):
        M = np.array([res[c]['grids'][name] for c in res])          # [n_cat, n_grid]
        mean_over_cat = M.mean(axis=0)
        j = int(np.argmax(mean_over_cat))
        return float(mean_over_cat[j]), grid_vals[j]

    au_val, k_val = tune('val_ksig', args.k_grid)
    au_test, k_test = tune('test_ksig', args.k_grid)
    au_pct, q_pct = tune('pct', args.q_grid)

    def mean_fixed(k):
        return float(np.mean([res[c]['fixed'][k] for c in res]))

    p('\n' + '=' * 78 + '\n===== MEAN SegF1 qua category =====')
    p(f'  prod  (val mean+3σ)          = {mean_fixed("prod"):.4f}   <- baseline production')
    p(f'  val_kσ*  (k={k_val:.2f} global)      = {au_val:.4f}   (trần họ NGUỒN)')
    p(f'  test_kσ* (k={k_test:.2f} global)     = {au_test:.4f}   (self-cal ĐÍCH, có tune)')
    p(f'  pct*     (q={q_pct:.4f} global)  = {au_pct:.4f}   (self-cal ĐÍCH, có tune)')
    p(f'  otsu_pool (0 tham số)         = {mean_fixed("otsu_pool"):.4f}   <- MÙ HOÀN TOÀN')
    p(f'  otsu_img  (0 tham số)         = {mean_fixed("otsu_img"):.4f}   <- MÙ HOÀN TOÀN')
    p(f'  oracle_pool (cần GT)          = {mean_fixed("oracle_pool"):.4f}   <- TRẦN')

    p('\nĐỌC:')
    p(' - Nếu otsu_img / otsu_pool >> prod và tiến gần oracle => có quy tắc MÙ ăn điểm ngay,')
    p('   0 hyperparam, tự dịch theo shift => vá cả in-domain LẪN cú trôi +2.79σ. Đây là Part-1.')
    p(' - Nếu test_kσ*/pct* (self-cal) >> val_kσ* => chọn thr trên phân bố ĐÍCH mới là chìa khóa.')
    p(' - Nếu MỌI quy tắc mù đều kẹt xa oracle => trần là per-image adaptivity => cần head/valley')
    p('   riêng từng ảnh, không phải 1 dao chung.')


if __name__ == '__main__':
    main()
