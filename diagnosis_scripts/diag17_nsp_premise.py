# diag17_nsp_premise.py
# -----------------------------------------------------------------------------
# Test TIỀN ĐỀ của novelty NSP (Nuisance-Subspace Projection), KHÔNG dùng nhãn:
#   - Đào rare-normal (từ validation/good hoặc train/good held-out): patch normal xa bank.
#   - Dựng nuisance subspace = PCA của (rare_normal - mean_bank) -> hướng "biến thiên normal hợp lệ".
#   - CHIẾU BỎ subspace đó khỏi feature (cả bank lẫn query) -> đo lại distance.
#   - So HARD separation defect-vs-rare-normal (diag10 = 0.45 ~ random) TRƯỚC vs SAU khi chiếu bỏ.
#
# ĐỌC:
#   HARD_proj >> HARD_orig (mà KHÔNG dùng nhãn) -> rare-normal nằm trong subspace tách được,
#       NSP phá được trần low-FPR unsupervised -> NOVELTY thật, đáng build full.
#   HARD_proj ~ HARD_orig -> defect & rare-normal cùng subspace -> NSP vô vọng -> bỏ.
#   (Kiểm FULL_proj không tụt nhiều so FULL_orig -> không phá tín hiệu dễ.)
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag17_nsp_premise.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --rn_q 95 --ranks 1 2 5 10 20 40 --out_dir ./diag17
# -----------------------------------------------------------------------------

import os
import sys
import glob
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

from dataset import MVTecAD2Dataset
from utils import get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    return [pil.crop((round(j * w / T), round(i * h / T), round((j + 1) * w / T), round((i + 1) * h / T)))
            for i in range(T) for j in range(T)]


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def img_featflat(bb, pil, T, R, gt, layers, eb):
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), eb):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + eb]])
        fl.append(bb.extract(b, layers))
    f = torch.cat(fl, 0)                      # [T*T, gt*gt, C]
    C = f.shape[-1]
    grid = torch.zeros(T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * gt:(i + 1) * gt, j * gt:(j + 1) * gt] = f[k, :gt * gt].reshape(gt, gt, C)
    return grid.reshape(-1, C)                # [G*G, C]


def gt_flat(gpath, label, G):
    if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros(G * G, dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8).reshape(-1)


@torch.no_grad()
def min_dist(q, bank, chunk=2048):
    out = torch.empty(q.shape[0], device=q.device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(1)[0]
    return out


def main():
    ap = argparse.ArgumentParser('Diag17: NSP premise (project out rare-normal subspace)')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--rn_q', type=float, default=95.0, help='percentile distance -> rare-normal')
    ap.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 5, 10, 20, 40])
    ap.add_argument('--hard_q', type=float, default=95.0, help='top-x%% normal theo d1 = hard-normal')
    ap.add_argument('--max_norm', type=int, default=40000)
    ap.add_argument('--probe_frac', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag17')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag17', args.out_dir).info
    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    T = args.tiles; gt = args.grid_tile
    rng = np.random.default_rng(args.seed)
    p('=' * 92)
    p(f'DIAG17 NSP premise | model={args.model} eff_grid={T*gt} layers={layers} | rn_q={args.rn_q} ranks={args.ranks}')
    p('HARD = defect vs top-5% normal theo d1 (diag10=0.45). So d1-AUROC TRƯỚC vs SAU chiếu bỏ rare-normal subspace.')
    p('=' * 92)

    rows = []
    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        rng.shuffle(tr)
        # bank từ train/good
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            for pth in tr:
                f = img_featflat(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                acc.append(subsample(f, keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        C = bank.shape[-1]; mu = bank.mean(0, keepdim=True)

        # probe để đào rare-normal: validation/good nếu có, else 30% train held-out
        val_dir = os.path.join(args.data_path, cat, 'validation', 'good')
        probe = sorted(glob.glob(os.path.join(val_dir, '*.png')) + glob.glob(os.path.join(val_dir, '*.jpg')))
        if not probe:
            probe = tr[:max(1, int(len(tr) * args.probe_frac))]
        rn = []
        with torch.no_grad():
            for pth in probe:
                f = img_featflat(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                d = min_dist(f, bank)
                thr = torch.quantile(d, args.rn_q / 100.0)
                rn.append(f[d >= thr].cpu())
        rn = torch.cat(rn, 0)
        rn = subsample(rn, args.max_norm).to(device)

        # nuisance subspace = PCA của (rare_normal - mu)
        Xc = (rn - mu)
        # SVD -> hướng chính
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)   # Vh: [min(N,C), C]
        Vfull = Vh                                              # rows = components

        # test patches: feature + label + d1_orig (để chọn hard-normal)
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        feats, labs = [], []
        with torch.no_grad():
            for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  {cat}'):
                f = img_featflat(bb, Image.open(ds.img_paths[idx]), T, R, gt, layers, args.enc_batch)
                G = int(round(f.shape[0] ** 0.5))
                lab = gt_flat(ds.gt_paths[idx], ds.labels[idx], G)
                feats.append(f.cpu()); labs.append(lab)
        F = torch.cat(feats, 0); y = np.concatenate(labs)
        # subsample normal, giữ hết defect
        di = np.where(y == 1)[0]; ni = np.where(y == 0)[0]
        if len(ni) > args.max_norm:
            ni = rng.choice(ni, args.max_norm, replace=False)
        sel = np.concatenate([di, ni]); Fs = F[sel].to(device); ys = y[sel]

        d1_orig = min_dist(Fs, bank).cpu().numpy()
        # hard-normal theo d1_orig
        norm_mask = ys == 0
        thr_h = np.percentile(d1_orig[norm_mask], args.hard_q)
        hard_norm = norm_mask & (d1_orig >= thr_h)
        hard_idx = (ys == 1) | hard_norm
        yh = ys[hard_idx]

        def auc(d):
            if yh.sum() < 3 or (1 - yh).sum() < 3:
                return float('nan')
            return roc_auc_score(yh, d[hard_idx])

        full_orig = roc_auc_score(ys, d1_orig) if ys.sum() > 0 else float('nan')
        hard_orig = auc(d1_orig)

        res = {'cat': cat, 'FULL_orig': full_orig, 'HARD_orig': hard_orig, 'proj': {}}
        for r in args.ranks:
            if r > Vfull.shape[0]:
                continue
            V = Vfull[:r]                                  # [r, C]
            P = V.t() @ V                                  # [C,C] projector onto nuisance
            bank_p = bank - (bank - mu) @ P
            Fs_p = Fs - (Fs - mu) @ P
            d1p = min_dist(Fs_p, bank_p).cpu().numpy()
            res['proj'][r] = (roc_auc_score(ys, d1p) if ys.sum() > 0 else float('nan'), auc(d1p))
        rows.append(res)
        best_r = max(res['proj'], key=lambda rr: (res['proj'][rr][1] if not np.isnan(res['proj'][rr][1]) else -1))
        bp = res['proj'][best_r]
        p(f'  [{cat:<11}] HARD orig={hard_orig:.3f} -> proj best(r={best_r})={bp[1]:.3f}  '
          f'| FULL orig={full_orig:.3f} -> {bp[0]:.3f}  | Δhard={bp[1]-hard_orig:+.3f}')

    p('\n' + '=' * 92)
    p('{:<13}{:>11}{:>12}{:>10}{:>11}{:>11}'.format('cat', 'HARD_orig', 'HARD_proj*', 'Δhard', 'FULL_orig', 'FULL_proj*'))
    H0, H1 = [], []
    for res in rows:
        best_r = max(res['proj'], key=lambda rr: (res['proj'][rr][1] if not np.isnan(res['proj'][rr][1]) else -1))
        bp = res['proj'][best_r]
        H0.append(res['HARD_orig']); H1.append(bp[1])
        p('{:<13}{:>11.3f}{:>12.3f}{:>10.3f}{:>11.3f}{:>11.3f}'.format(
            res['cat'], res['HARD_orig'], bp[1], bp[1] - res['HARD_orig'], res['FULL_orig'], bp[0]))
    p('{:<13}{:>11.3f}{:>12.3f}{:>10.3f}'.format('MEAN', np.nanmean(H0), np.nanmean(H1), np.nanmean(H1) - np.nanmean(H0)))

    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('cat,HARD_orig,HARD_proj_best,best_rank,FULL_orig,FULL_proj_best\n')
        for res in rows:
            br = max(res['proj'], key=lambda rr: (res['proj'][rr][1] if not np.isnan(res['proj'][rr][1]) else -1))
            bp = res['proj'][br]
            f.write(f'{res["cat"]},{res["HARD_orig"]:.4f},{bp[1]:.4f},{br},{res["FULL_orig"]:.4f},{bp[0]:.4f}\n')
    p('\nĐỌC: Δhard >> 0 (HARD_proj cao hẳn) MÀ KHÔNG dùng nhãn -> NSP phá trần rare-normal -> NOVELTY thật, build full.')
    p('     Δhard ~ 0 -> defect & rare-normal cùng subspace -> NSP vô vọng, bỏ.')


if __name__ == '__main__':
    main()
