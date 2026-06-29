# eval_nsp.py
# -----------------------------------------------------------------------------
# NOVELTY end-to-end: Nuisance-Subspace Projection (NSP), UNSUPERVISED (k=0, không nhãn defect).
# Premise đã xác nhận (diag17): chiếu bỏ subspace rare-normal -> tách defect-vs-rare-normal tốt hơn.
# Giờ đo TÁC ĐỘNG THẬT lên AUPRO0.05 & SegF1: PLAIN distance vs NSP distance, sweep rank.
#
#   - bank từ train/good; đào rare-normal từ validation/good (held-out normal, KHÔNG nhãn).
#   - nuisance subspace V = PCA của (rare_normal - mean_bank).
#   - NSP: chiếu bỏ V khỏi (bank & query) rồi NN-distance -> anomaly map.
#   - so PLAIN (rank 0) vs NSP(rank r) ở AUPRO0.05 + SegF1 (mean 8 cat, test_public).
#
# Chạy:
#   HF_HUB_OFFLINE=1 python eval_nsp.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --rn_q 95 --ranks 0 20 40 80 160 --out_dir ./diag_nsp
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
def nn_map_from(qflat, bank, G, device, chunk=4096):
    out = torch.empty(qflat.shape[0], device=device)
    for s in range(0, qflat.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(qflat[s:s + chunk], bank).min(1)[0]
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
    out = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return out[7], out[5]  # AUPRO0.05, SegF1


def main():
    ap = argparse.ArgumentParser('Eval NSP end-to-end: PLAIN vs nuisance-subspace-projected distance')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--rn_q', type=float, default=95.0)
    ap.add_argument('--max_rn', type=int, default=40000)
    ap.add_argument('--probe_frac', type=float, default=0.3)
    ap.add_argument('--ranks', type=int, nargs='+', default=[0, 20, 40, 80, 160], help='0 = PLAIN')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag_nsp')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('nsp', args.out_dir).info
    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile
    rng = np.random.default_rng(args.seed)
    p('=' * 88)
    p(f'EVAL NSP (unsup) | model={args.model} eff_grid={T*gt} layers={layers} | rn_q={args.rn_q} ranks={args.ranks}')
    p('PLAIN (rank 0) vs NSP(rank r). Báo AUPRO0.05 & SegF1, mean 8 cat.')
    p('=' * 88)

    agg = {r: [] for r in args.ranks}
    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        rng.shuffle(tr)
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            for pth in tr:
                fg = img_grid(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                acc.append(subsample(fg.reshape(-1, fg.shape[-1]), keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        C = bank.shape[-1]; mu = bank.mean(0, keepdim=True)

        # đào rare-normal từ validation/good (held-out) -> nuisance subspace
        val_dir = os.path.join(args.data_path, cat, 'validation', 'good')
        probe = sorted(glob.glob(os.path.join(val_dir, '*.png')) + glob.glob(os.path.join(val_dir, '*.jpg')))
        if not probe:
            probe = tr[:max(1, int(len(tr) * args.probe_frac))]
        rnf = []
        with torch.no_grad():
            for pth in probe:
                fg = img_grid(bb, Image.open(pth), T, R, gt, layers, args.enc_batch).reshape(-1, C)
                d = torch.cdist(fg, bank).min(1)[0]
                thr = torch.quantile(d, args.rn_q / 100.0)
                rnf.append(fg[d >= thr].cpu())
        rnf = subsample(torch.cat(rnf, 0), args.max_rn).to(device)
        _, _, Vh = torch.linalg.svd(rnf - mu, full_matrices=False)   # [min, C]

        # test grids + gt
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        grids, gts = [], []
        with torch.no_grad():
            for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  {cat}'):
                fg = img_grid(bb, Image.open(ds.img_paths[idx]), T, R, gt, layers, args.enc_batch)
                grids.append(fg); gts.append(gt_grid(ds.gt_paths[idx], ds.labels[idx], fg.shape[0]))

        for r in args.ranks:
            if r == 0:
                bank_r = bank
                maps = [nn_map_from(g.reshape(-1, C), bank_r, g.shape[0], device) for g in grids]
            else:
                if r > Vh.shape[0]:
                    continue
                V = Vh[:r]; P = V.t() @ V
                bank_r = bank - (bank - mu) @ P
                maps = []
                for g in grids:
                    q = g.reshape(-1, C)
                    qp = q - (q - mu) @ P
                    maps.append(nn_map_from(qp, bank_r, g.shape[0], device))
            au, f1 = evaluate_set(maps, gts, gk, device)
            agg[r].append((au, f1))
            tag = 'PLAIN' if r == 0 else f'NSP r={r}'
            p(f'  [{cat:<11}] {tag:<9} AUPRO05={au:.4f} SegF1={f1:.4f}')

    p('\n' + '=' * 88)
    p('{:<12}{:>12}{:>12}'.format('rank', 'AUPRO0.05', 'SegF1'))
    rows = []
    for r in args.ranks:
        if not agg[r]:
            continue
        m = np.array(agg[r]).mean(0)
        tag = 'PLAIN' if r == 0 else f'NSP-{r}'
        rows.append((tag, m[0], m[1]))
        p('{:<12}{:>12.4f}{:>12.4f}'.format(tag, m[0], m[1]))
    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('config,AUPRO0.05,SegF1\n')
        for t, a, s in rows:
            f.write(f'{t},{a:.4f},{s:.4f}\n')
    p('\nĐỌC: NSP-r > PLAIN ở AUPRO0.05/SegF1 (unsup, KHÔNG nhãn) -> NOVELTY end-to-end. '
      'Best rank để build full + nộp track unsup VAND.')


if __name__ == '__main__':
    main()
