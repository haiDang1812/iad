# diag16_segf1_rootcause.py
# -----------------------------------------------------------------------------
# MỔ vì sao SegF1 thấp per-category (dù AUROC/AUPRO cao) -> để biết novelty cần làm gì.
# Giả thuyết: SegF1 bị giới hạn bởi BIÊN/độ sắc (map mờ -> over-segment -> precision sụp),
# KHÔNG phải ranking. SegF1 là pixel + có ngưỡng nên rất nhạy với mờ; AUPRO (region) thì không.
#
# Per category (v3_large fuse, eval-split, global-norm), ở res cao (mặc định 512):
#   - tại ngưỡng F1-opt: Precision, Recall, F1, và pred_area/gt_area (>1 = over-segment).
#   - SegF1 với GAUSSIAN (mờ, hiện tại) vs KHÔNG gaussian (sắc hơn) -> mờ có hại SegF1 không.
#   - kích thước defect trung bình (% diện tích) -> cat defect nhỏ có phải cat SegF1 thấp.
#
# ĐỌC:
#   - precision THẤP + pred/gt>>1 + gaussian hại -> SegF1 nghẽn ở BIÊN/độ sắc
#       -> novelty = boundary refinement (vd feature-affinity của DINO, không cần SAM).
#   - recall THẤP (miss) -> nghẽn ở phát hiện, không phải biên.
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag16_segf1_rootcause.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --head_w 0.6 --eval_res 512 --out_dir ./diag16
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
from sklearn.metrics import precision_recall_curve

from dataset import MVTecAD2Dataset
from utils import get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
HARD = {'can', 'sheet_metal', 'wallplugs'}
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


def up(arr2d, size, device, gk=None):
    t = torch.tensor(arr2d, device=device)[None, None].float()
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    if gk is not None:
        t = gk(t)
    return t[0, 0].cpu().numpy()


def gt_at(gpath, label, size):
    if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((size, size), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((size, size), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


def f1_breakdown(maps, gts, max_neg=4000000, seed=0):
    s = np.concatenate([m.reshape(-1) for m in maps])
    y = np.concatenate([g.reshape(-1) for g in gts]).astype(np.uint8)
    if y.sum() == 0:
        return dict(F1=0, P=0, R=0, area_ratio=0)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    negs = rng.choice(neg, min(len(neg), max_neg), replace=False)
    idx = np.concatenate([pos, negs])
    ys, ss = y[idx], s[idx]
    prec, rec, thr = precision_recall_curve(ys, ss)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1))
    t = thr[min(i, len(thr) - 1)]
    # area ratio trên TOÀN bộ (không subsample) để đo over-seg đúng
    pred = sum(int((m >= t).sum()) for m in maps)
    gtsum = sum(int(g.sum()) for g in gts)
    return dict(F1=float(f1[i]), P=float(prec[i]), R=float(rec[i]),
                area_ratio=pred / max(1, gtsum))


def main():
    ap = argparse.ArgumentParser('Diag16: SegF1 root-cause per category')
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
    ap.add_argument('--eval_res', type=int, default=512, help='res đo SegF1 (cao -> lộ vấn đề biên)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag16')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag16', args.out_dir).info
    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile; hw = args.head_w; S = args.eval_res
    rng = np.random.default_rng(args.seed)
    p('=' * 96)
    p(f'DIAG16 SegF1 root-cause | model={args.model} eff_grid={T*gt} eval_res={S} | k={args.shots} head_w={hw}')
    p('=' * 96)

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

        sp_feat, sp_gt = [], []
        for i in shot_pool:
            g = img_featmap(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch).cpu().numpy()
            G = g.shape[0]
            sp_feat.append(g.reshape(-1, Cdim)); sp_gt.append(gt_grid(ds.gt_paths[i], 1, G).reshape(-1))
        X = np.concatenate(sp_feat); y = np.concatenate(sp_gt).astype(int)
        if y.sum() < 3:
            p(f'  [{cat}] thiếu defect, bỏ'); continue
        clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0] - 1)),
                            LogisticRegression(max_iter=2000, class_weight='balanced'))
        clf.fit(X, y)

        # eval: distance + head -> fuse grid; gt grid; defect size
        ev_d, ev_fuse_grid, ev_lab, ev_gpath = [], [], [], []
        with torch.no_grad():
            for idx in tqdm(eval_idx, ncols=80, desc=f'  {cat}'):
                grid = img_featmap(bb, Image.open(ds.img_paths[idx]), T, R, gt, layers, args.enc_batch)
                G = grid.shape[0]
                d = nn_map(grid, bank, device)
                prob = clf.predict_proba(grid.cpu().numpy().reshape(-1, Cdim))[:, 1].reshape(G, G)
                ev_d.append(d); ev_fuse_grid.append((d, prob))
                ev_lab.append(ds.labels[idx]); ev_gpath.append(ds.gt_paths[idx])
        dall = np.stack(ev_d, 0); lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)

        gts = [gt_at(gp, lb, S) for gp, lb in zip(ev_gpath, ev_lab)]
        maps_g, maps_s = [], []  # gaussian vs sharp
        for (d, prob) in ev_fuse_grid:
            fz = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * prob
            maps_g.append(up(fz, S, device, gk=gk))
            maps_s.append(up(fz, S, device, gk=None))

        bg = f1_breakdown(maps_g, gts, seed=args.seed)
        bs = f1_breakdown(maps_s, gts, seed=args.seed)
        # defect size %
        dsz = np.mean([g.mean() for g, lb in zip(gts, ev_lab) if lb == 1]) * 100

        tag = '*' if cat in HARD else ' '
        p(f'{tag}[{cat:<11}] GAUSS F1={bg["F1"]:.3f} P={bg["P"]:.3f} R={bg["R"]:.3f} area×{bg["area_ratio"]:.2f}'
          f'  | SHARP F1={bs["F1"]:.3f}  | defect={dsz:.2f}%')
        rows.append((cat, bg, bs, dsz))

    p('\n' + '=' * 96)
    p('{:<13}{:>8}{:>8}{:>8}{:>9}{:>10}{:>9}'.format('cat', 'F1g', 'P', 'R', 'area×', 'F1sharp', 'defect%'))
    for cat, bg, bs, dsz in rows:
        p('{:<13}{:>8.3f}{:>8.3f}{:>8.3f}{:>9.2f}{:>10.3f}{:>9.2f}'.format(
            cat, bg['F1'], bg['P'], bg['R'], bg['area_ratio'], bs['F1'], dsz))
    arr = np.array([[r[1]['F1'], r[1]['P'], r[1]['R'], r[1]['area_ratio'], r[2]['F1'], r[3]] for r in rows])
    m = arr.mean(0)
    p('{:<13}{:>8.3f}{:>8.3f}{:>8.3f}{:>9.2f}{:>10.3f}{:>9.2f}'.format('MEAN', *m))

    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('cat,F1_gauss,P,R,area_ratio,F1_sharp,defect_pct\n')
        for cat, bg, bs, dsz in rows:
            f.write(f'{cat},{bg["F1"]:.4f},{bg["P"]:.4f},{bg["R"]:.4f},{bg["area_ratio"]:.4f},{bs["F1"]:.4f},{dsz:.4f}\n')
    p('\nĐỌC: P thấp + area×>>1 -> over-segment/biên mờ -> novelty=boundary refinement. '
      'F1sharp > F1gauss -> gaussian đang hại SegF1. R thấp -> nghẽn phát hiện (khác hướng).')


if __name__ == '__main__':
    main()
