# diag14_sota_errors.py
# -----------------------------------------------------------------------------
# MỔ LỖI CÒN LẠI của SOTA hiện tại (v3_large few-shot FUSE, global-norm) ở low-FPR.
# Mục đích: quyết định novelty mới là CÁI GÌ (có bằng chứng, không đoán).
#
# Với mỗi category (eval-split test_public, giữ k shot):
#   - AUPRO0.05 của FUSE (SOTA) và ORACLE (full-label CV) -> GAP còn lại.
#   - Đặt ngưỡng tại FPR=0.05 (trên pixel normal). Ở đó:
#       * FP: phân rã "rare-normal-driven" (distance đẩy: (1-hw)*dr > hw*pr)
#                    vs "head-overfire" (head đẩy)  -> quyết RNM hay calibrate head.
#       * FN (defect bị miss): mean distance & mean head-prob -> defect "trông như normal"?
#
# ĐỌC:
#   - FP phần lớn rare-normal-driven -> RNM đúng thuốc.
#   - FP phần lớn head-overfire      -> cần regularize/calibrate head, KHÔNG phải RNM.
#   - FN nhiều & dr,pr đều thấp       -> defect in-distribution -> cần sensitivity/contrastive.
#   - GAP-to-oracle lớn              -> còn headroom cho mechanism mới.
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag14_sota_errors.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --head_w 0.6 --out_dir ./diag14
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
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
HARD = {'can', 'sheet_metal', 'wallplugs'}
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
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
def img_featmap(bb, pil, T, R, gt, layers, enc_batch):
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), enc_batch):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + enc_batch]])
        fl.append(bb.extract(b, layers))
    f = torch.cat(fl, 0)
    C = f.shape[-1]
    grid = torch.zeros(T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * gt:(i + 1) * gt, j * gt:(j + 1) * gt] = f[k, :gt * gt].reshape(gt, gt, C)
    return grid


@torch.no_grad()
def nn_map(grid, bank, device, chunk=4096):
    G = grid.shape[0]; C = grid.shape[-1]
    q = grid.reshape(-1, C)
    out = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(1)[0]
    return out.reshape(G, G).cpu().numpy()


def gt_grid(gpath, label, G):
    if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((G, G), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


def upmap(arr2d, size, gk, device):
    t = torch.tensor(arr2d, device=device)[None, None].float()
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return gk(t)[0, 0].cpu().numpy()


def evaluate_set(maps, gts, gk, device, resize=256, r=0.01):
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * r))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    return ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)


def main():
    ap = argparse.ArgumentParser('Diag14: mổ lỗi SOTA few-shot FUSE ở low-FPR')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--fpr', type=float, default=0.05, help='FPR điểm vận hành để mổ lỗi')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max_patches', type=int, default=40000)
    ap.add_argument('--resize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag14')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag14', args.out_dir).info

    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile; hw = args.head_w
    rng = np.random.default_rng(args.seed)
    p('=' * 88)
    p(f'DIAG14 mổ lỗi SOTA | model={args.model} eff_grid={T*gt} layers={layers} | k={args.shots} '
      f'head_w={hw} | FPR={args.fpr}')
    p('=' * 88)

    rows = []
    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            buf = []
            for pth in tr:
                buf.extend(tile_pils(Image.open(pth), T))
                while len(buf) >= args.enc_batch:
                    b = torch.stack([to_tensor(t, R) for t in buf[:args.enc_batch]]); buf = buf[args.enc_batch:]
                    f = bb.extract(b, layers)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), args.enc_batch * keep).cpu())
            if buf:
                f = bb.extract(torch.stack([to_tensor(t, R) for t in buf]), layers)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        Cdim = bank.shape[-1]

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        shot_pool = bad[:args.shots]
        eval_idx = bad[args.shots:] + good

        def prep(idx):
            grid = img_featmap(bb, Image.open(ds.img_paths[idx]), T, R, gt, layers, args.enc_batch)
            d = nn_map(grid, bank, device)
            g = gt_grid(ds.gt_paths[idx], ds.labels[idx], grid.shape[0])
            return grid.cpu().numpy(), d, g
        ev = [prep(i) for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval')]
        sp_feat = [img_featmap(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch).cpu().numpy()
                   for i in shot_pool]
        sp_gt = [gt_grid(ds.gt_paths[i], 1, ev[0][0].shape[0]) for i in shot_pool]
        ev_feat = [e[0] for e in ev]; ev_d = [e[1] for e in ev]; ev_gt = [e[2] for e in ev]

        # global-norm distance
        dall = np.stack(ev_d, 0); lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)
        ev_dr = [(d - lo) / (hi - lo + 1e-8) for d in ev_d]

        # few-shot head
        Xs, ys = [], []
        for f, g in zip(sp_feat, sp_gt):
            Xs.append(f.reshape(-1, Cdim)); ys.append(g.reshape(-1))
        X = np.concatenate(Xs); y = np.concatenate(ys).astype(int)
        if y.sum() < 3:
            p(f'  [{cat}] thiếu defect, bỏ'); continue
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0] - 1)),
                            LogisticRegression(max_iter=2000, class_weight='balanced'))
        clf.fit(X, y)
        ev_pr = [clf.predict_proba(f.reshape(-1, Cdim))[:, 1].reshape(f.shape[0], f.shape[0]) for f in ev_feat]
        fuse_grid = [(1 - hw) * dr + hw * pr for dr, pr in zip(ev_dr, ev_pr)]

        # AUPRO few-shot
        au_fs = evaluate_set(fuse_grid, ev_gt, gk, device, resize=args.resize)[7]

        # ORACLE (full-label CV) -> headroom
        N = len(ev_feat)
        oof = [None] * N
        kf = KFold(n_splits=min(args.folds, N), shuffle=True, random_state=args.seed)
        for tr_i, va_i in kf.split(range(N)):
            Xo = np.concatenate([ev_feat[i].reshape(-1, Cdim) for i in tr_i])
            yo = np.concatenate([ev_gt[i].reshape(-1) for i in tr_i]).astype(int)
            if yo.sum() < 3 or (1 - yo).sum() < 3:
                continue
            if Xo.shape[0] > args.max_patches:
                pos = np.where(yo == 1)[0]; neg = np.where(yo == 0)[0]
                nn_ = max(1, min(len(neg), args.max_patches - len(pos)))
                sel = np.concatenate([pos, rng.choice(neg, nn_, replace=False)])
                Xo, yo = Xo[sel], yo[sel]
            co = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, Xo.shape[1], Xo.shape[0] - 1)),
                               LogisticRegression(max_iter=2000, class_weight='balanced'))
            co.fit(Xo, yo)
            for i in va_i:
                oof[i] = co.predict_proba(ev_feat[i].reshape(-1, Cdim))[:, 1].reshape(ev_feat[i].shape[0], ev_feat[i].shape[0])
        oracle_grid = [oof[i] if oof[i] is not None else ev_dr[i] for i in range(N)]
        au_or = evaluate_set(oracle_grid, ev_gt, gk, device, resize=args.resize)[7]

        # ---- mổ lỗi ở FPR = args.fpr (trên map 256, đã gaussian) ----
        S = args.resize
        dr256 = [upmap(m, S, gk, device) for m in ev_dr]
        pr256 = [upmap(m, S, gk, device) for m in ev_pr]
        fz256 = [(1 - hw) * a + hw * b for a, b in zip(dr256, pr256)]
        gt256 = [np.asarray(Image.fromarray(g).resize((S, S), Image.NEAREST)) > 0 for g in ev_gt]

        norm_scores = np.concatenate([fz[~g].reshape(-1) for fz, g in zip(fz256, gt256)])
        thr = np.quantile(norm_scores, 1 - args.fpr)

        fp_dist, fp_head, n_fp, n_fn, n_tp, n_def = 0, 0, 0, 0, 0, 0
        fn_dr, fn_pr = [], []
        for a, b, fz, g in zip(dr256, pr256, fz256, gt256):
            fp = (~g) & (fz >= thr)
            tp = g & (fz >= thr)
            fn = g & (fz < thr)
            n_fp += int(fp.sum()); n_tp += int(tp.sum()); n_fn += int(fn.sum()); n_def += int(g.sum())
            if fp.any():
                dist_driven = ((1 - hw) * a[fp]) > (hw * b[fp])   # distance term > head term
                fp_dist += int(dist_driven.sum()); fp_head += int((~dist_driven).sum())
            if fn.any():
                fn_dr.append(a[fn]); fn_pr.append(b[fn])
        fp_rare_frac = fp_dist / max(1, fp_dist + fp_head)
        recall = n_tp / max(1, n_def)
        fn_dr_m = float(np.concatenate(fn_dr).mean()) if fn_dr else float('nan')
        fn_pr_m = float(np.concatenate(fn_pr).mean()) if fn_pr else float('nan')

        tag = '*' if cat in HARD else ' '
        p(f'{tag}[{cat:<11}] AUPRO05 fs={au_fs:.3f} oracle={au_or:.3f} gap={au_or-au_fs:+.3f} | '
          f'@FPR{args.fpr}: recall={recall:.3f} FP_rare={fp_rare_frac:.2f} FP_head={1-fp_rare_frac:.2f} | '
          f'FN dr={fn_dr_m:.2f} pr={fn_pr_m:.2f}')
        rows.append((cat, au_fs, au_or, recall, fp_rare_frac, fn_dr_m, fn_pr_m))

    p('\n' + '=' * 88)
    p('{:<13}{:>9}{:>9}{:>8}{:>10}{:>10}{:>9}{:>9}'.format(
        'cat', 'fs05', 'orac05', 'gap', 'recall', 'FP_rare', 'FNdr', 'FNpr'))
    arr = []
    for r in rows:
        p('{:<13}{:>9.3f}{:>9.3f}{:>8.3f}{:>10.3f}{:>10.2f}{:>9.2f}{:>9.2f}'.format(
            r[0], r[1], r[2], r[2] - r[1], r[3], r[4], r[5], r[6]))
        arr.append(r[1:])
    arr = np.array(arr)
    p('{:<13}{:>9.3f}{:>9.3f}{:>8.3f}{:>10.3f}{:>10.2f}{:>9.2f}{:>9.2f}'.format(
        'MEAN', arr[:, 0].mean(), arr[:, 1].mean(), (arr[:, 1] - arr[:, 0]).mean(),
        arr[:, 2].mean(), arr[:, 3].mean(), np.nanmean(arr[:, 4]), np.nanmean(arr[:, 5])))

    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('cat,fs_AUPRO05,oracle_AUPRO05,recall@fpr,FP_rare_frac,FN_mean_dr,FN_mean_pr\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n')
    p('\nĐỌC: FP_rare cao -> RNM đúng thuốc. FP_rare thấp (head-overfire) -> calibrate head. '
      'recall thấp + FN dr,pr thấp -> defect in-distribution (cần sensitivity). gap lớn -> còn headroom.')


if __name__ == '__main__':
    main()
