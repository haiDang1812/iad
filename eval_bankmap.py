# eval_bankmap.py
# -----------------------------------------------------------------------------
# LEVER MAP (sau khi resolution CHẾT ở quick harness: 4:48 thua 3:48 kể cả bank90k):
#   2 trục ĐỘC LẬP đo trong 1 run, nền 3:48 chốt:
#     TRỤC BANK : random subsample (hiện tại)  vs  greedy k-center CORESET
#                 (PatchCore-style; cùng bank_size, phủ manifold đều hơn, bỏ near-dup)
#     TRỤC SCORE: 1-NN min dist (hiện tại)     vs  mean top-3 NN (k-NN, bớt nhiễu đuôi)
#   -> 4 variant: rand1 (baseline y hệt eval_fairthr), rand3, core1, core3.
#
#   Threshold: RULE ĐÃ ĐÓNG BĂNG (thắng argmax 3 lần liên tiếp): p=95 gain=1.15 morph=CÓ.
#   KHÔNG sweep nữa. Ngưỡng tự suy per-variant từ heldout train/good (fair).
#
#   FAIR: bank + threshold CHỈ train/good; GT chỉ để chấm. Không setting per-cat.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   - Δ(core - rand) mean: AUPRO >= +0.010 hoặc trần >= +0.020  -> CORESET VÀO config nền.
#   - Δ(knn3 - 1nn)  mean: tương tự                              -> KNN VÀO config nền.
#   - cả hai ~0 / âm -> trục bank/score CHẾT ở quick harness, map lever còn lại =
#     overlap-averaging (SuperADD 640/128) và guided native-res upsampling (novelty front).
#   - variant tốt nhất (nếu có trục ăn) = config nền mới cho mọi run sau.
#
#   python eval_bankmap.py --data_path ../data --out_dir ./bankmap --max_eval 30 \
#       --categories vial wallplugs sheet_metal fabric rice walnuts can fruit_jelly
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
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import img_featgrid, build_bank, subsample, IMG_EXT   # noqa: E402
from eval_fairthr import closing                                                  # noqa: E402
from eval_native import Hist, make_map                                            # noqa: E402
from diag30_thin_premise import aupro05                                           # noqa: E402
from dataset import MVTecAD2Dataset                                               # noqa: E402
from utils import get_gaussian_kernel, get_logger                                 # noqa: E402
from backbones_ext import load_backbone                                           # noqa: E402

warnings.filterwarnings('ignore')

RULE_P, RULE_G = 95.0, 1.15          # ĐÓNG BĂNG (fairthr: argmax 3 lần liên tiếp). KHÔNG ĐỔI.
VARIANTS = ['rand1', 'rand3', 'core1', 'core3']


@torch.no_grad()
def coreset(cand, m, device, dproj=128, seed=0):
    """Greedy k-center trên chiếu ngẫu nhiên 128-D (PatchCore-style). cand: (N,C) GPU."""
    N = cand.shape[0]
    if N <= m:
        return cand
    g = torch.Generator().manual_seed(seed)
    P = (torch.randn(cand.shape[1], dproj, generator=g) / dproj ** 0.5).to(device)
    X = cand @ P
    sel = torch.empty(m, dtype=torch.long, device=device)
    sel[0] = int(torch.randint(N, (1,), generator=g))
    mind = torch.cdist(X, X[sel[0]][None]).squeeze(1)
    for i in range(1, m):
        sel[i] = torch.argmax(mind)
        mind = torch.minimum(mind, torch.cdist(X, X[sel[i]][None]).squeeze(1))
    return cand[sel]


@torch.no_grad()
def nn2_grid(g, bank, device, chunk=2048):
    """1 lượt cdist -> cả 2 score: d1 = min, d3 = mean top-3. Trả 2 mảng (G,G)."""
    G = g.shape[0]; C = g.shape[-1]
    q = g.reshape(-1, C).to(device)
    d1 = torch.empty(q.shape[0], device=device)
    d3 = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        v = torch.cdist(q[s:s + chunk], bank).topk(3, dim=1, largest=False).values
        d1[s:s + chunk] = v[:, 0]
        d3[s:s + chunk] = v.mean(1)
    return d1.reshape(G, G).cpu().numpy(), d3.reshape(G, G).cpu().numpy()


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))   # per-cat, độc lập thứ tự cat
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr
    va = tr[args.max_train:args.max_train + args.n_val] if args.max_train else []
    overlap = False
    if len(va) < 3:
        va, overlap = tr_use[-args.n_val:], True

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={G} heldout={len(va)}{" [TRÙNG BANK]" if overlap else ""}')

    cand = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.cand_size, device)
    banks = {'rand': subsample(cand, args.bank_size).to(device)}
    banks['core'] = coreset(cand, args.bank_size, device, seed=args.seed)
    p(f'    [{cat}] cand={cand.shape[0]} -> bank rand={banks["rand"].shape[0]} core={banks["core"].shape[0]}')
    del cand; torch.cuda.empty_cache()

    # ---- heldout -> ngưỡng per-variant theo RULE đóng băng ----
    tr_px = {v: [] for v in VARIANTS}
    for path in tqdm(va, ncols=70, desc=f'    {cat} heldout', leave=False):
        pil = Image.open(path)
        W, H = pil.size
        g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
        for bn in ('rand', 'core'):
            d1, d3 = nn2_grid(g, banks[bn], device)
            for kk, s in (('1', d1), ('3', d3)):
                m = make_map(s, args.canvas, gk, (H, W), device).cpu().numpy()
                tr_px[bn + kk].append(m.ravel()[::4])
        del g
    thr = {v: float(np.percentile(np.concatenate(tr_px[v]), RULE_P)) * RULE_G for v in VARIANTS}
    del tr_px

    # ---- test: featgrid 1 lần / ảnh -> 4 map ----
    s_grids = {v: [] for v in VARIANTS}
    for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
        g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
        for bn in ('rand', 'core'):
            d1, d3 = nn2_grid(g, banks[bn], device)
            s_grids[bn + '1'].append(d1); s_grids[bn + '3'].append(d3)
        del g
    del banks; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        gmin = min(float(s.min()) for s in s_grids[v])
        gmax = max(float(s.max()) for s in s_grids[v])
        h = Hist(gmin - 0.05, gmax + 0.05)
        ap_preds, ap_gts = [], []
        mst = np.zeros(3, np.float64)
        for s, i in zip(s_grids[v], idx):
            W, H = Image.open(ds.img_paths[i]).size
            m_t = make_map(s, args.canvas, gk, (H, W), device)
            if ds.labels[i] == 0:
                g_nat = np.zeros((H, W), np.uint8)
                g_ap = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
            else:
                gpil = Image.open(ds.gt_paths[i]).convert('L')
                g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
                g_ap = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
            h.add(m_t.cpu().numpy().reshape(-1), g_nat.reshape(-1))
            ap_preds.append(make_map(s, args.canvas, gk, (args.aupro_res, args.aupro_res), device).cpu().numpy())
            ap_gts.append(g_ap)
            r = max(1, round(min(H, W) / G))
            g_t = torch.from_numpy(g_nat).to(device) > 0
            pred = closing(m_t > thr[v], 2 * r + 1)
            mst += ((pred & g_t).sum().item(), (pred & ~g_t).sum().item(), ((~pred) & g_t).sum().item())
            del m_t, g_t
        tp, fp, fn = mst
        out[v] = {'aupro': aupro05(ap_preds, ap_gts, rng), 'f1_max': h.f1_max(),
                  'f1': float(2 * tp / (2 * tp + fp + fn + 1e-9))}
        del ap_preds, ap_gts
        p(f'    [{cat}] {v:5s}: AUPRO0.05={out[v]["aupro"]:.4f}  trần={out[v]["f1_max"]:.4f}  F1@rule={out[v]["f1"]:.4f}')
    del s_grids
    return out


def main():
    ap = argparse.ArgumentParser('eval_bankmap: coreset vs random bank x 1nn vs knn3 (rule threshold đóng băng)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000, help='pool ứng viên trước khi chọn bank')
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split. Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./bankmap')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('bankmap', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} bank={args.bank_size} cand={args.cand_size} '
      f'heldout={args.n_val} | RULE ĐÓNG BĂNG p={RULE_P} gain={RULE_G} morph=CÓ')
    p('  variants: rand1 (=baseline fairthr) | rand3 | core1 | core3. FAIR toàn bộ, không per-cat.')

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
        p(f'  {v:5s}: AUPRO0.05={mean[v]["aupro"]:.4f}  trần={mean[v]["f1_max"]:.4f}  F1@rule={mean[v]["f1"]:.4f}')
    dc = {k: (mean['core1'][k] + mean['core3'][k] - mean['rand1'][k] - mean['rand3'][k]) / 2 for k in mean['rand1']}
    dk = {k: (mean['rand3'][k] + mean['core3'][k] - mean['rand1'][k] - mean['core1'][k]) / 2 for k in mean['rand1']}
    p(f'\n  Δ TRỤC BANK  (core-rand): AUPRO{dc["aupro"]:+.4f}  trần{dc["f1_max"]:+.4f}  F1{dc["f1"]:+.4f}')
    p(f'  Δ TRỤC SCORE (knn3-1nn) : AUPRO{dk["aupro"]:+.4f}  trần{dk["f1_max"]:+.4f}  F1{dk["f1"]:+.4f}')
    bestv = max(VARIANTS, key=lambda v: mean[v]['aupro'])
    p(f'  variant tốt nhất theo AUPRO: {bestv} ({mean[bestv]["aupro"]:.4f})')
    p('\nĐỌC (pre-registered): trục vào config nền nếu ΔAUPRO>=+0.010 hoặc Δtrần>=+0.020. '
      'Cả hai ~0/âm -> trục bank/score chết ở quick harness -> còn overlap-averaging + guided upsampling.')


if __name__ == '__main__':
    main()
