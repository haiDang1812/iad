# eval_nsp_fuse.py
# -----------------------------------------------------------------------------
# Test: NSP có đẩy SOTA nội bộ (few-shot FUSE) lên không?
# SOTA = (1-hw)*norm(distance_PLAIN) + hw*head.  Ứng viên = thay distance_PLAIN bằng distance_NSP
# (chiếu bỏ subspace rare-normal). Head GIỮ NGUYÊN (trên feature gốc, đã train trên 10 shot).
#
# So PLAIN-fuse (=SOTA) vs NSP-fuse(rank r), cùng head_w, trên test_public eval-split.
# Báo AUPRO0.05 & SegF1, per-cat + mean.
#
# Chạy:
#   HF_HUB_OFFLINE=1 python eval_nsp_fuse.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --head_w 0.6 --ranks 0 20 40 --out_dir ./diag_nspfuse
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
def img_grid(bb, pil, T, R, gt, layers, eb):
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), eb):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + eb]])
        fl.append(bb.extract(b, layers))
    f = torch.cat(fl, 0)
    C = f.shape[-1]
    grid = torch.zeros(T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * gt:(i + 1) * gt, j * gt:(j + 1) * gt] = f[k, :gt * gt].reshape(gt, gt, C)
    return grid


@torch.no_grad()
def nnmap(qflat, bank, G, device, chunk=4096):
    out = torch.empty(qflat.shape[0], device=device)
    for s in range(0, qflat.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(qflat[s:s + chunk], bank).min(1)[0]
    return out.reshape(G, G).cpu().numpy()


def gt_grid(gpath, label, G):
    if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((G, G), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


def upmap(a, size, gk, device):
    t = torch.tensor(a, device=device)[None, None].float()
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return gk(t)[0, 0].cpu().numpy()


def evalset(maps, gts, gk, device, resize=256, r=0.01):
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * r))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    o = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return o[7], o[5]


def norm01_stack(dlist):
    dall = np.stack(dlist, 0); lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)
    return [(d - lo) / (hi - lo + 1e-8) for d in dlist]


def main():
    ap = argparse.ArgumentParser('NSP trong few-shot FUSE: có đẩy SOTA không')
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
    ap.add_argument('--rn_q', type=float, default=95.0)
    ap.add_argument('--max_rn', type=int, default=40000)
    ap.add_argument('--ranks', type=int, nargs='+', default=[0, 20, 40], help='0 = PLAIN (=SOTA)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag_nspfuse')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('nspfuse', args.out_dir).info
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
    p(f'NSP-FUSE | model={args.model} eff_grid={T*gt} | k={args.shots} head_w={hw} ranks={args.ranks}')
    p('PLAIN-fuse (rank0 = SOTA) vs NSP-fuse(r). AUPRO0.05 & SegF1.')
    p('=' * 88)

    agg = {r: [] for r in args.ranks}
    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        rng.shuffle(tr)
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            for pth in tr:
                fgg = img_grid(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                acc.append(subsample(fgg.reshape(-1, fgg.shape[-1]), keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        C = bank.shape[-1]; mu = bank.mean(0, keepdim=True)

        # nuisance subspace từ validation/good
        val_dir = os.path.join(args.data_path, cat, 'validation', 'good')
        probe = sorted(glob.glob(os.path.join(val_dir, '*.png')) + glob.glob(os.path.join(val_dir, '*.jpg')))
        if not probe:
            probe = tr[:max(1, len(tr) // 3)]
        rnf = []
        with torch.no_grad():
            for pth in probe:
                fg = img_grid(bb, Image.open(pth), T, R, gt, layers, args.enc_batch).reshape(-1, C)
                d = torch.cdist(fg, bank).min(1)[0]
                rnf.append(fg[d >= torch.quantile(d, args.rn_q / 100.0)].cpu())
        rnf = subsample(torch.cat(rnf, 0), args.max_rn).to(device)
        _, _, Vh = torch.linalg.svd(rnf - mu, full_matrices=False)

        # test split + shots
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        shot = bad[:args.shots]; eval_idx = bad[args.shots:] + good

        Xs, ys = [], []
        with torch.no_grad():
            for i in shot:
                g = img_grid(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch).cpu().numpy()
                G = g.shape[0]
                Xs.append(g.reshape(-1, C)); ys.append(gt_grid(ds.gt_paths[i], 1, G).reshape(-1))
        X = np.concatenate(Xs); yy = np.concatenate(ys).astype(int)
        if yy.sum() < 3:
            p(f'  [{cat}] thiếu defect, bỏ'); continue
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0] - 1)),
                            LogisticRegression(max_iter=2000, class_weight='balanced'))
        clf.fit(X, yy)

        # test grids + head prob (head trên feature GỐC)
        grids, gts, pr_list = [], [], []
        with torch.no_grad():
            for idx in tqdm(eval_idx, ncols=80, desc=f'  {cat}'):
                g = img_grid(bb, Image.open(ds.img_paths[idx]), T, R, gt, layers, args.enc_batch)
                G = g.shape[0]
                grids.append(g); gts.append(gt_grid(ds.gt_paths[idx], ds.labels[idx], G))
                pr_list.append(clf.predict_proba(g.reshape(-1, C).cpu().numpy())[:, 1].reshape(G, G))

        for r in args.ranks:
            if r == 0:
                dlist = [nnmap(g.reshape(-1, C), bank, g.shape[0], device) for g in grids]
            else:
                if r > Vh.shape[0]:
                    continue
                V = Vh[:r]; P = V.t() @ V
                bank_r = bank - (bank - mu) @ P
                dlist = []
                for g in grids:
                    q = g.reshape(-1, C); qp = q - (q - mu) @ P
                    dlist.append(nnmap(qp, bank_r, g.shape[0], device))
            drn = norm01_stack(dlist)
            maps = [(1 - hw) * dr + hw * pr for dr, pr in zip(drn, pr_list)]
            au, f1 = evalset(maps, gts, gk, device)
            agg[r].append((au, f1))
            tag = 'PLAIN(SOTA)' if r == 0 else f'NSP r={r}'
            p(f'  [{cat:<11}] {tag:<12} AUPRO05={au:.4f} SegF1={f1:.4f}')

    p('\n' + '=' * 88)
    p('{:<14}{:>12}{:>12}'.format('config', 'AUPRO0.05', 'SegF1'))
    rows = []
    for r in args.ranks:
        if not agg[r]:
            continue
        m = np.array(agg[r]).mean(0)
        tag = 'PLAIN(SOTA)' if r == 0 else f'NSP-{r}'
        rows.append((tag, m[0], m[1]))
        p('{:<14}{:>12.4f}{:>12.4f}'.format(tag, m[0], m[1]))
    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('config,AUPRO0.05,SegF1\n')
        for t, a, s in rows:
            f.write(f'{t},{a:.4f},{s:.4f}\n')
    p('\nĐỌC: NSP-r > PLAIN(SOTA) ở CẢ AUPRO0.05 & SegF1 -> novelty đẩy được SOTA. '
      'NSP ≈ PLAIN -> head đã bao phủ (NSP thừa) -> SOTA chỉ lên bằng resolution.')


if __name__ == '__main__':
    main()
