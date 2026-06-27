# diag13_backbone_sweep.py
# -----------------------------------------------------------------------------
# SWEEP backbone: DINOv2 + DINOv3 × {base, large, huge/giant} — đo trực tiếp (không đoán)
# backbone ảnh hưởng định vị low-FPR thế nào.
#
# Mỗi backbone (qua backbones_ext, HF transformers):
#   - input = GRID × patch (v2 patch14, v3 patch16) -> CÙNG cỡ lưới GRID×GRID giữa mọi model
#     => so sánh CÔNG BẰNG về granularity, chênh lệch = thuần do encoder.
#   - UNSUP  : memory-bank NN distance AUPRO0.05 (định vị KHÔNG nhãn)
#   - ORACLE : logistic CÓ NHÃN (k-fold theo ảnh) AUPRO0.05 = TRẦN định vị của backbone
#   - báo per-category (* = cat khó can/sheet_metal/wallplugs) + mean + meanHARD.
#
# ĐỌC: so UNSUP & ORACLE giữa các backbone, NHẤT là meanHARD:
#   - ORACLE_HARD thấp ở mọi backbone -> trần do supervision/method, swap ít giúp.
#   - backbone X nâng ORACLE_HARD rõ  -> X đáng dùng (semantics/feature tốt hơn thật sự).
#   - v3 ≈ v2 cùng size             -> semantics bão hòa, đòn bẩy là granularity (Swin/CNN).
#
# Chạy (subset cho nhẹ trước, rồi full):
#   python diag13_backbone_sweep.py --data_path ../data --models v2_base v3_base --grid 28 --out_dir ./diag13
#   python diag13_backbone_sweep.py --data_path ../data \
#     --models v2_base v2_large v2_giant v3_base v3_large v3_huge --grid 28 --out_dir ./diag13
# -----------------------------------------------------------------------------

import os
import sys
import glob
import argparse
import warnings

# cho phép chạy từ thư mục con (vd diagnosis_scripts/): thêm repo-root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def featgrid(bb, pil, R, layers, enc_batch):
    f = bb.extract(to_tensor(pil, R).unsqueeze(0), layers)   # [1,N,C]
    g = int(round(f.shape[1] ** 0.5))
    C = f.shape[-1]
    return f[0, :g * g].reshape(g, g, C)                      # [G,G,C] (cuda)


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
    ap = argparse.ArgumentParser('Diag13: backbone sweep (v2/v3 × size) localization ceiling')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--models', type=str, nargs='+',
                    default=['v2_base', 'v2_large', 'v2_giant', 'v3_base', 'v3_large', 'v3_huge'])
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--grid', type=int, default=28, help='cỡ lưới patch chung (input = grid×patch)')
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=8)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max_patches', type=int, default=60000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag13')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag13', args.out_dir).info
    p('=' * 88)
    p(f'DIAG13 BACKBONE SWEEP | grid={args.grid} (cùng granularity) | layers={args.layers}')
    p(f'models={args.models} | UNSUP=distance, ORACLE=logistic CÓ NHÃN (k-fold ảnh) | metric AUPRO0.05')
    p('=' * 88)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_rows = {}
    for mname in args.models:
        try:
            bb = load_backbone(mname, device)
        except SystemExit as e:
            p(f'[SKIP] {mname}: {e}')
            continue
        R = args.grid * bb.patch
        p(f'\n########## {mname} | input={R} (grid {args.grid}×{args.grid}, patch {bb.patch}) ##########')
        rows = []
        for cat in args.categories:
            tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                        glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
            # bank
            acc, keep = [], max(64, args.bank_size * 4 // max(1, len(tr)))
            with torch.no_grad():
                for s in range(0, len(tr), args.enc_batch):
                    batch = torch.stack([to_tensor(Image.open(pth), R) for pth in tr[s:s + args.enc_batch]])
                    f = bb.extract(batch, args.layers)          # [B,N,C]
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), batch.shape[0] * keep).cpu())
            bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)

            ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                                 transform=None, gt_transform=None, phase='test')
            N = len(ds.img_paths)
            feats, dist, gts = [], [], []
            with torch.no_grad():
                for idx in tqdm(range(N), ncols=80, desc=f'  {mname}/{cat}'):
                    grid = featgrid(bb, Image.open(ds.img_paths[idx]), R, args.layers, args.enc_batch)
                    feats.append(grid.cpu().numpy())
                    dist.append(nn_map(grid, bank, device))
                    gts.append(gt_grid(ds.gt_paths[idx], ds.labels[idx], grid.shape[0]))
            C = feats[0].shape[-1]; G = feats[0].shape[0]

            ru = evaluate_set(dist, gts, gk, device)            # UNSUP

            oof = [None] * N
            kf = KFold(n_splits=min(args.folds, N), shuffle=True, random_state=args.seed)
            for tr_i, va_i in kf.split(range(N)):
                X = np.concatenate([feats[i].reshape(-1, C) for i in tr_i])
                y = np.concatenate([gts[i].reshape(-1) for i in tr_i]).astype(int)
                if y.sum() < 3 or (1 - y).sum() < 3:
                    continue
                if X.shape[0] > args.max_patches:
                    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
                    rng = np.random.default_rng(args.seed)
                    nneg = max(1, min(len(neg), args.max_patches - len(pos)))
                    sel = np.concatenate([pos, rng.choice(neg, size=nneg, replace=False)])
                    X, y = X[sel], y[sel]
                clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0] - 1)),
                                    LogisticRegression(max_iter=2000, class_weight='balanced'))
                clf.fit(X, y)
                for i in va_i:
                    oof[i] = clf.predict_proba(feats[i].reshape(-1, C))[:, 1].reshape(G, G)
            oracle_maps = [oof[i] if oof[i] is not None else dist[i] for i in range(N)]
            ro = evaluate_set(oracle_maps, gts, gk, device)     # ORACLE

            tag = '*' if cat in HARD else ' '
            p(f'  {tag}[{cat:<11}] UNSUP05={ru[7]:.4f}  ORACLE05={ro[7]:.4f}  gap={ro[7]-ru[7]:+.4f}')
            rows.append((cat, ru[7], ro[7]))
        all_rows[mname] = rows
        del bb
        torch.cuda.empty_cache()

    p('\n' + '=' * 88)
    p('{:<12}{:>11}{:>11}{:>13}{:>13}'.format('model', 'UNSUP05', 'ORACLE05', 'HARD-UNSUP', 'HARD-ORACLE'))
    summary = []
    for mname in args.models:
        if mname not in all_rows:
            continue
        arr = np.array([[r[1], r[2]] for r in all_rows[mname]])
        harr = np.array([[r[1], r[2]] for r in all_rows[mname] if r[0] in HARD])
        mu, mo = arr.mean(0); hu, ho = harr.mean(0)
        p('{:<12}{:>11.4f}{:>11.4f}{:>13.4f}{:>13.4f}'.format(mname, mu, mo, hu, ho))
        summary.append((mname, mu, mo, hu, ho))

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('model,category,UNSUP_AUPRO05,ORACLE_AUPRO05\n')
        for mname in all_rows:
            for r in all_rows[mname]:
                f.write(f'{mname},{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
        f.write('\nmodel,mean_UNSUP05,mean_ORACLE05,meanHARD_UNSUP05,meanHARD_ORACLE05\n')
        for s in summary:
            f.write(f'{s[0]},{s[1]:.4f},{s[2]:.4f},{s[3]:.4f},{s[4]:.4f}\n')
    p(f'\nĐã lưu: {csv}')
    p('ĐỌC: HARD-ORACLE là then chốt. Backbone nào nâng nó rõ -> đáng dùng. '
      'Mọi backbone đều thấp -> trần do supervision. v3≈v2 -> đòn bẩy là granularity (Swin).')


if __name__ == '__main__':
    main()
