# diag15_lowfpr_mismatch.py
# -----------------------------------------------------------------------------
# Xác nhận METRIC–OBJECTIVE MISMATCH: head train bằng BCE/cross-entropy (logistic) xếp hạng
# TỔNG THỂ tốt (pixel-AUROC cao) NHƯNG hỏng đúng vùng LOW-FPR (partial-AUROC@0.05 thấp).
# Nếu đúng -> có dư địa cho objective tối ưu low-FPR (SoftPRO@0.05) = justify novelty mới.
#
# Per category (v3_large few-shot, eval-split test_public, global-norm), cho HEAD và FUSE:
#   - AUROC      : pixel-AUROC toàn cục (xếp hạng tổng thể)
#   - pAUROC0.05 : partial pixel-AUROC tới FPR=0.05 (McClish-normalized) = chất lượng LOW-FPR
#   - AUPRO0.05  : region overlap ở low-FPR (metric đích)
#   gap = AUROC - pAUROC0.05 (lớn -> low-FPR là điểm yếu -> objective low-FPR còn dư địa)
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag15_lowfpr_mismatch.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --head_w 0.6 --out_dir ./diag15
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
from sklearn.metrics import roc_auc_score

from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
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


def region_metrics(maps, gts, gk, device, resize=256):
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * 0.01))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    r = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return r[7], r[5]   # AUPRO0.05, P-F1max (SegF1)


def pooled_pixels(maps, gts, gk, device, resize=256, max_neg=600000, seed=0):
    yp, ys = [], []
    for m, g in zip(maps, gts):
        s = upmap(m, resize, gk, device).reshape(-1)
        y = (np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) > 0).reshape(-1).astype(np.uint8)
        yp.append(s); ys.append(y)
    s = np.concatenate(yp); y = np.concatenate(ys)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    if len(neg) > max_neg:
        neg = rng.choice(neg, max_neg, replace=False)
    idx = np.concatenate([pos, neg])
    return y[idx], s[idx]


def main():
    ap = argparse.ArgumentParser('Diag15: metric-objective mismatch (AUROC vs pAUROC@0.05)')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--resize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag15')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag15', args.out_dir).info

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
    p('=' * 86)
    p(f'DIAG15 mismatch | model={args.model} eff_grid={T*gt} layers={layers} | k={args.shots} head_w={hw}')
    p('AUROC=xếp hạng tổng thể | pAUROC0.05=chất lượng low-FPR | gap lớn -> objective low-FPR có dư địa')
    p('=' * 86)

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

        dall = np.stack(ev_d, 0); lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)
        ev_dr = [(d - lo) / (hi - lo + 1e-8) for d in ev_d]

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
        head_maps = ev_pr
        fuse_maps = [(1 - hw) * dr + pr_ * hw for dr, pr_ in zip(ev_dr, ev_pr)]

        for tag, maps in [('HEAD', head_maps), ('FUSE', fuse_maps)]:
            yv, sv = pooled_pixels(maps, ev_gt, gk, device, resize=args.resize, seed=args.seed)
            try:
                full = roc_auc_score(yv, sv)
                p005 = roc_auc_score(yv, sv, max_fpr=0.05)
            except Exception:
                full, p005 = float('nan'), float('nan')
            au, segf1 = region_metrics(maps, ev_gt, gk, device, resize=args.resize)
            p(f'  [{cat:<11}] {tag:<4} AUROC={full:.4f} pAUROC0.05={p005:.4f} gap={full-p005:+.4f} '
              f'AUPRO0.05={au:.4f} SegF1={segf1:.4f}')
            rows.append((cat, tag, full, p005, au, segf1))

    p('\n' + '=' * 86)
    p('{:<6}{:>10}{:>14}{:>10}{:>12}{:>10}'.format('branch', 'AUROC', 'pAUROC0.05', 'gap', 'AUPRO0.05', 'SegF1'))
    for tag in ['HEAD', 'FUSE']:
        sub = [r for r in rows if r[1] == tag]
        if not sub:
            continue
        a = np.array([[r[2], r[3], r[4], r[5]] for r in sub])
        m = np.nanmean(a, 0)
        p('{:<6}{:>10.4f}{:>14.4f}{:>10.4f}{:>12.4f}{:>10.4f}'.format(tag, m[0], m[1], m[0] - m[1], m[2], m[3]))

    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('cat,branch,AUROC,pAUROC0.05,AUPRO0.05,SegF1\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n')
    p('\nĐỌC: AUROC cao mà pAUROC0.05 thấp (gap lớn) -> BCE head xếp hạng tốt tổng thể nhưng HỎNG low-FPR')
    p('     -> objective tối ưu trực tiếp low-FPR (SoftPRO@0.05) CÓ dư địa -> justify novelty.')
    p('     Nếu pAUROC0.05 ~ AUROC (gap nhỏ) -> head đã tốt ở low-FPR -> loss mới khó ăn.')


if __name__ == '__main__':
    main()
