# eval_fewshot_bb.py
# -----------------------------------------------------------------------------
# Few-shot FUSE (như eval_fewshot.py) NHƯNG backbone qua backbones_ext -> chạy được
# DINOv2 / DINOv3 mọi size. Mục đích: xem v3 có vượt 0.7181 (v2) ở pipeline THẬT không.
#
# Khớp granularity công bằng: mỗi tile resize về (grid_tile × patch) -> lưới grid_tile×grid_tile
# /tile bất kể patch (v2=14, v3=16). tiles=2 -> eff lưới (2*grid_tile)². grid_tile=28 ~ eff 56 (như 0.7181).
# Layer auto-scale theo độ sâu model (base 12-layer là tham chiếu).
#
# Chạy (cache, offline):
#   HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 0 10 --out_dir ./diag_fewshot_v3large
#   # đối chứng v2 cùng pipeline:
#   HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v2_base \
#     --tiles 2 --grid_tile 28 --shots 0 10 --out_dir ./diag_fewshot_v2base
# -----------------------------------------------------------------------------

import os
import sys
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
def img_featmap(bb, pil, T, R, gt, layers, enc_batch):
    # ghép T*T tile -> lưới [T*gt, T*gt, C]
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), enc_batch):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + enc_batch]])
        fl.append(bb.extract(b, layers))
    f = torch.cat(fl, 0)                              # [T*T, gt*gt, C]
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
    return out.reshape(G, G)


def gt_grid(gpath, label, G):
    if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((G, G), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


def upmap(arr2d, size, gk, device):
    t = torch.tensor(arr2d, device=device)[None, None].float()
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return gk(t)[0, 0].cpu().numpy()


def evaluate_set(maps, gts, gk, device, resize=256, r=0.01, morph=0):
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    if morph > 0:
        from scipy import ndimage
        pr = np.stack([ndimage.grey_closing(m, size=(morph, morph)) for m in pr], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * r))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    return ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)


def find_train(data_path, cat):
    import glob
    return sorted(glob.glob(os.path.join(data_path, cat, 'train', 'good', '*.png')) +
                  glob.glob(os.path.join(data_path, cat, 'train', 'good', '*.jpg')))


def main():
    ap = argparse.ArgumentParser('Few-shot FUSE trên backbone bất kỳ (v2/v3) qua backbones_ext')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v2_base', help='tên backbones_ext: v2_base/v3_large/...')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9],
                    help='layer tham chiếu trên model 12-layer; tự scale theo độ sâu')
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28, help='lưới patch mỗi tile (input/tile = grid_tile×patch)')
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, nargs='+', default=[0, 10])
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--morph_close', type=int, default=0)
    ap.add_argument('--global_norm', action='store_true',
                    help='khớp submission: dist chuẩn hoá TOÀN CỤC + head raw, sweep --head_w')
    ap.add_argument('--head_w', type=float, nargs='+', default=[0.5, 0.7],
                    help='(global_norm) trọng số head trong fuse, có thể list để sweep')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag_fewshot_bb')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fewshot_bb', args.out_dir).info

    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch                          # input mỗi tile
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))

    p('=' * 80)
    p(f'FEW-SHOT BB | model={args.model} patch={patch} | tiles={args.tiles} grid_tile={args.grid_tile} '
      f'(tile_res={R}, eff_grid={args.tiles*args.grid_tile}) | layers={layers} | shots={args.shots}')
    p('UNSUP=distance | HEAD=logistic(k) | FUSE=0.5 rank(d)+0.5 head | FMULT=rank(d)*head')
    p('=' * 80)

    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile
    rng = np.random.default_rng(args.seed)
    agg = {}

    for cat in args.categories:
        tr = find_train(args.data_path, cat)
        # bank
        acc = []
        keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            buf = []
            for pth in tr:
                buf.extend(tile_pils(Image.open(pth), T))
                while len(buf) >= args.enc_batch:
                    b = torch.stack([to_tensor(t, R) for t in buf[:args.enc_batch]]); buf = buf[args.enc_batch:]
                    f = bb.extract(b, layers)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), args.enc_batch * keep).cpu())
            if buf:
                b = torch.stack([to_tensor(t, R) for t in buf])
                f = bb.extract(b, layers)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        maxk = max(args.shots)
        shot_pool = bad[:maxk]
        eval_idx = bad[maxk:] + good

        def prep(idx):
            grid = img_featmap(bb, Image.open(ds.img_paths[idx]), T, R, gt, layers, args.enc_batch)
            d = nn_map(grid, bank, device).cpu().numpy()
            g = gt_grid(ds.gt_paths[idx], ds.labels[idx], grid.shape[0])
            return grid.cpu().numpy(), d, g
        ev = [prep(i) for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval')]
        sp_feat = [img_featmap(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch).cpu().numpy()
                   for i in shot_pool]
        sp_gt = [gt_grid(ds.gt_paths[i], 1, ev[0][0].shape[0]) for i in shot_pool]
        ev_feat = [e[0] for e in ev]; ev_d = [e[1] for e in ev]; ev_gt = [e[2] for e in ev]
        Cdim = ev[0][0].shape[-1]

        # global-norm distance (khớp submission) nếu bật
        ev_dr_global = None
        if args.global_norm:
            dall = np.stack(ev_d, 0); lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)
            ev_dr_global = [(d - lo) / (hi - lo + 1e-8) for d in ev_d]

        for k in args.shots:
            if k == 0:
                umaps = ev_dr_global if args.global_norm else ev_d
                ru = evaluate_set(umaps, ev_gt, gk, device, morph=args.morph_close)
                agg.setdefault((0, 'UNSUP'), []).append(ru)
                p(f'  [{cat}] k=0 UNSUP AUPRO05={ru[7]:.4f} F1={ru[5]:.4f}')
                continue
            Xs, ys = [], []
            for f, g in zip(sp_feat[:k], sp_gt[:k]):
                Xs.append(f.reshape(-1, Cdim)); ys.append(g.reshape(-1))
            X = np.concatenate(Xs); y = np.concatenate(ys).astype(int)
            if y.sum() < 3 or (1 - y).sum() < 3:
                continue
            clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0] - 1)),
                                LogisticRegression(max_iter=2000, class_weight='balanced'))
            clf.fit(X, y)
            if args.global_norm:
                # KHỚP SUBMISSION: dist global + head raw, sweep head_w
                ev_pr = [clf.predict_proba(f.reshape(-1, Cdim))[:, 1].reshape(f.shape[0], f.shape[0]) for f in ev_feat]
                for hw in args.head_w:
                    maps = [(1 - hw) * dr + hw * pr for dr, pr in zip(ev_dr_global, ev_pr)]
                    rr = evaluate_set(maps, ev_gt, gk, device, morph=args.morph_close)
                    agg.setdefault((k, f'hw{hw:.2f}'), []).append(rr)
                    p(f'  [{cat}] k={k} hw{hw:.2f} AUPRO05={rr[7]:.4f} F1={rr[5]:.4f}')
            else:
                head, fuse, fmult = [], [], []
                for f, d in zip(ev_feat, ev_d):
                    G = f.shape[0]
                    prob = clf.predict_proba(f.reshape(-1, Cdim))[:, 1].reshape(G, G)
                    dr = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    pr = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
                    head.append(pr); fuse.append(0.5 * dr + 0.5 * pr); fmult.append(dr * pr)
                for bn, mp in [('HEAD', head), ('FUSE', fuse), ('FMULT', fmult)]:
                    rr = evaluate_set(mp, ev_gt, gk, device, morph=args.morph_close)
                    agg.setdefault((k, bn), []).append(rr)
                    p(f'  [{cat}] k={k} {bn:<5} AUPRO05={rr[7]:.4f} F1={rr[5]:.4f}')

    p('\n' + '=' * 80)
    p('{:<14}{:>12}{:>12}'.format('shot/branch', 'AUPRO0.05', 'P-F1max'))
    rows = []
    for key in sorted(agg.keys(), key=lambda x: (x[0], x[1])):
        m = np.array(agg[key]).mean(0)
        rows.append((f'{key[0]}-{key[1]}', m[7], m[5]))
        p('{:<14}{:>12.4f}{:>12.4f}'.format(f'{key[0]}-{key[1]}', m[7], m[5]))
    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('shot_branch,AUPRO0.05,P-F1max\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
    p(f'\nĐã lưu: {csv}  | model={args.model}')
    p('ĐỌC: so 10-FUSE của model này với v2_base 0.7181. Cao hơn rõ -> swap backbone cho method.')


if __name__ == '__main__':
    main()
