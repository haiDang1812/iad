# eval_guidedup.py
# -----------------------------------------------------------------------------
# MẶT TRẬN NOVELTY: GUIDED NATIVE-RES UPSAMPLING (bản FAIR của cơ chế NRS, không GT).
#
# Cơ chế đã chứng minh (NRS/grid-sweep): defect sub-cell chết vì map grid 144 bilinear-upsample
#   lên 2448x2048 -> biên nhòe ~1 cell (~15px) -> precision sập ở SegF1. NRS cũ sửa bằng head
#   feed GT = UNFAIR. Bản fair: GUIDED FILTER (He et al.) — upsample map với guide = chính ảnh
#   native grayscale -> biên map bám biên ảnh, training-free, zero GT, zero learn.
#
# NỀN = config thắng overlapmap (3:48, bank coreset 50k/cand200k từ cửa sổ chồng lấn, 1-NN,
#   overlap-averaging 5x5, RULE ĐÓNG BĂNG p95 g1.15 morph). Encode + score 1 LẦN/ảnh;
#   variants chỉ khác bước upsample:
#     base : bilinear như hiện tại (tái lập ov của overlapmap trong-run)
#     g*   : guided filter r = 1 grid cell native, eps ∈ {1e-3, 1e-2, 1e-1} (guide chuẩn [0,1])
#   (grid eps chạy MỘT lần rồi ĐÓNG BĂNG theo argmax — tiền lệ fairthr PS×GAINS.)
#
#   FAIR: bank + threshold CHỈ train/good (guided áp cả heldout khi suy ngưỡng); GT chỉ chấm.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   - Variant guided TỐT NHẤT theo TRẦN so base: Δtrần >= +0.020 hoặc ΔAUPRO >= +0.010
#     -> GUIDED VÀO nền, eps đóng băng = argmax mean trần.
#   - ~0/âm toàn dải eps -> premise guided chết ở quick harness (biên ảnh không thẳng hàng
#     biên defect đủ mạnh) -> novelty phải tìm dạng khác (vd. joint multi-scale, can-front).
#   - Kỳ vọng chính ở TRẦN/F1 (cơ chế biên); AUPRO@512 ít nhạy — không đọc ngược.
#
#   python eval_guidedup.py --data_path ../data --out_dir ./guidedup --max_eval 30 \
#       --categories vial wallplugs sheet_metal fabric rice walnuts can fruit_jelly
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import IMG_EXT                                         # noqa: E402
from eval_bankmap import coreset                                                   # noqa: E402
from eval_overlapmap import build_cand_overlap, overlap_score                      # noqa: E402
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from diag30_thin_premise import aupro05                                            # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

RULE_P, RULE_G = 95.0, 1.15          # ĐÓNG BĂNG. KHÔNG ĐỔI.
EPS = [1e-3, 1e-2, 1e-1]
VARIANTS = ['base'] + [f'g{e:g}' for e in EPS]


def boxf(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


@torch.no_grad()
def guided_up(m_t, gray, r):
    """Guided filter: m_t (H,W) score GPU, gray (H,W) guide [0,1] GPU. Trả dict eps -> (H,W).
    Thống kê guide/map dùng chung, mỗi eps chỉ thêm 2 box filter."""
    I = gray[None, None]; p = m_t[None, None]
    mI = boxf(I, r); mp = boxf(p, r)
    varI = boxf(I * I, r) - mI * mI
    cov = boxf(I * p, r) - mI * mp
    out = {}
    for e in EPS:
        a = cov / (varI + e)
        b = mp - a * mI
        out[f'g{e:g}'] = (boxf(a, r) * I + boxf(b, r)).squeeze(0).squeeze(0)
    return out


def load_gray(pil, device, size=None):
    g = pil.convert('L')
    if size is not None:
        g = g.resize(size, Image.BILINEAR)
    return torch.from_numpy(np.asarray(g, np.float32) / 255.0).to(device)


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr
    va = tr[args.max_train:args.max_train + args.n_val] if args.max_train else []
    overlap_warn = False
    if len(va) < 3:
        va, overlap_warn = tr_use[-args.n_val:], True

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={G} heldout={len(va)}{" [TRÙNG BANK]" if overlap_warn else ""}')

    cand = build_cand_overlap(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.cand_size, device).to(device)
    bank = coreset(cand, args.bank_size, device, seed=args.seed)
    del cand; torch.cuda.empty_cache()
    p(f'    [{cat}] bank(coreset,overlap)={bank.shape[0]} tu cand={args.cand_size}')

    r512 = max(1, round(args.aupro_res / G))

    def all_maps(s, pil, W, H):
        """dict variant -> map GPU native + dict variant -> map numpy @aupro_res."""
        m_t = make_map(s, args.canvas, gk, (H, W), device)
        r_nat = max(1, round(min(H, W) / G))
        nat = {'base': m_t}
        nat.update(guided_up(m_t, load_gray(pil, device), r_nat))
        m5 = make_map(s, args.canvas, gk, (args.aupro_res, args.aupro_res), device)
        ap = {'base': m5}
        ap.update(guided_up(m5, load_gray(pil, device, (args.aupro_res, args.aupro_res)), r512))
        return nat, {v: ap[v].cpu().numpy() for v in ap}

    # ---- heldout -> ngưỡng per-variant theo RULE đóng băng ----
    tr_px = {v: [] for v in VARIANTS}
    for path in tqdm(va, ncols=70, desc=f'    {cat} heldout', leave=False):
        pil = Image.open(path)
        W, H = pil.size
        s = overlap_score(bb, pil, T, R, gt_, layers, args.enc_batch, bank, device)
        nat, _ = all_maps(s, pil, W, H)
        for v in VARIANTS:
            tr_px[v].append(nat[v].cpu().numpy().ravel()[::4])
        del nat
    thr = {v: float(np.percentile(np.concatenate(tr_px[v]), RULE_P)) * RULE_G for v in VARIANTS}
    del tr_px

    # ---- test: score 1 lần/ảnh, variants chỉ khác upsample ----
    s_grids = []
    for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
        s_grids.append(overlap_score(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch, bank, device))
    del bank; torch.cuda.empty_cache()

    gmin = min(float(s.min()) for s in s_grids)
    gmax = max(float(s.max()) for s in s_grids)
    h = {v: Hist(gmin - 0.05, gmax + 0.05) for v in VARIANTS}
    mst = {v: np.zeros(3, np.float64) for v in VARIANTS}
    ap_preds = {v: [] for v in VARIANTS}
    ap_gts = []
    for s, i in zip(tqdm(s_grids, ncols=70, desc=f'    {cat} maps', leave=False), idx):
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), np.uint8)
            g_ap = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
        else:
            gpil = Image.open(ds.gt_paths[i]).convert('L')
            g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
            g_ap = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
        ap_gts.append(g_ap)
        nat, ap = all_maps(s, pil, W, H)
        g_t = torch.from_numpy(g_nat).to(device) > 0
        r = max(1, round(min(H, W) / G))
        for v in VARIANTS:
            h[v].add(nat[v].cpu().numpy().reshape(-1), g_nat.reshape(-1))
            ap_preds[v].append(ap[v])
            pred = closing(nat[v] > thr[v], 2 * r + 1)
            mst[v] += ((pred & g_t).sum().item(), (pred & ~g_t).sum().item(), ((~pred) & g_t).sum().item())
        del nat, g_t
    del s_grids

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = {'aupro': aupro05(ap_preds[v], ap_gts, rng), 'f1_max': h[v].f1_max(),
                  'f1': float(2 * tp / (2 * tp + fp + fn + 1e-9))}
        p(f'    [{cat}] {v:6s}: AUPRO0.05={out[v]["aupro"]:.4f}  trần={out[v]["f1_max"]:.4f}  F1@rule={out[v]["f1"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_guidedup: bilinear vs guided-filter native upsample (nền ov+coreset, rule đóng băng)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split. Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./guidedup')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('guidedup', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} bank={args.bank_size}(coreset,overlap) cand={args.cand_size} '
      f'heldout={args.n_val} | RULE ĐÓNG BĂNG p={RULE_P} gain={RULE_G} morph=CÓ | eps={EPS} r=1 cell')
    p('  variants: base (=ov overlapmap, bilinear) | g<eps> (guided filter, guide=ảnh native). FAIR, không per-cat.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN theo variant =====')
    mean = {}
    for v in VARIANTS:
        mean[v] = {k: float(np.mean([res[c][v][k] for c in res])) for k in ('aupro', 'f1_max', 'f1')}
        p(f'  {v:6s}: AUPRO0.05={mean[v]["aupro"]:.4f}  trần={mean[v]["f1_max"]:.4f}  F1@rule={mean[v]["f1"]:.4f}')
    bestg = max(VARIANTS[1:], key=lambda v: mean[v]['f1_max'])
    d = {k: mean[bestg][k] - mean['base'][k] for k in mean['base']}
    p(f'\n  guided tốt nhất theo trần: {bestg}')
    p(f'  Δ GUIDED ({bestg} - base): AUPRO{d["aupro"]:+.4f}  trần{d["f1_max"]:+.4f}  F1{d["f1"]:+.4f}')
    p('\nĐỌC (pre-registered): Δtrần>=+0.020 hoặc ΔAUPRO>=+0.010 -> GUIDED VÀO nền, eps đóng băng theo argmax trần. '
      '~0/âm toàn dải eps -> premise guided chết ở quick harness.')


if __name__ == '__main__':
    main()
