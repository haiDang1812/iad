# eval_fewshot_rnm.py
# -----------------------------------------------------------------------------
# NOVELTY (cơ chế mới, gắn diag10/11): RARE-NORMAL MINING.
# Diag10: FP low-FPR = rare-normal (patch normal nhưng xa bank), unsup không tách nổi.
# Diag11: có giám sát thì tách được. Few-shot thường chỉ label DEFECT.
# -> Ý mới: head học phân biệt defect (k nhãn, positive) VỚI rare-normal (negative),
#    mà rare-normal ĐÀO MIỄN PHÍ từ train/good (toàn normal): patch distance cao nhất.
#    => dạy model "lạ-mà-bình-thường KHÔNG phải lỗi" -> dập đúng FP low-FPR.
#
# So trực tiếp:  BASE (head defect-vs-normal-pixels) vs RNM (+ rare-normal mined negatives).
# Chuẩn hoá TOÀN CỤC (khớp submission) + sweep head_w. Backbone bất kỳ qua backbones_ext.
#
# Chạy:
#   HF_HUB_OFFLINE=1 python eval_fewshot_rnm.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --head_w 0.5 0.7 --rn_q 97 --out_dir ./diag_rnm
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
def nn_dist(feats, bank, chunk=4096):
    q = feats.reshape(-1, feats.shape[-1])
    out = torch.empty(q.shape[0], device=q.device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(1)[0]
    return out


def nn_map(grid, bank, device):
    G = grid.shape[0]
    return nn_dist(grid, bank).reshape(G, G).cpu().numpy()


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


def fit_head(X, y, w, pca, seed=0):
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(pca, X.shape[1], X.shape[0] - 1)),
                        LogisticRegression(max_iter=2000, class_weight='balanced'))
    if w is None:
        clf.fit(X, y)
    else:
        clf.fit(X, y, logisticregression__sample_weight=w)
    return clf


def main():
    ap = argparse.ArgumentParser('Few-shot + RARE-NORMAL MINING (novelty gắn diag10/11)')
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
    ap.add_argument('--head_w', type=float, nargs='+', default=[0.5, 0.7])
    ap.add_argument('--morph', type=int, default=0)
    # rare-normal mining
    ap.add_argument('--rn_q', type=float, default=97.0, help='percentile distance để coi là rare-normal')
    ap.add_argument('--rn_max', type=int, default=20000, help='số rare-normal negative tối đa')
    ap.add_argument('--rn_weight', type=float, default=1.0, help='sample_weight cho rare-normal negative')
    ap.add_argument('--probe_frac', type=float, default=0.2, help='tỉ lệ train/good giữ làm probe để đào rare-normal')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag_rnm')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('rnm', args.out_dir).info

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
    p('=' * 84)
    p(f'RNM | model={args.model} tile_res={R} eff_grid={T*gt} layers={layers} | k={args.shots} '
      f'head_w={args.head_w} | rn_q={args.rn_q} rn_max={args.rn_max} rn_w={args.rn_weight}')
    p('So BASE (head thường) vs RNM (+rare-normal mined negative). Global-norm. Báo AUPRO0.05 & P-F1max.')
    p('=' * 84)
    agg = {}

    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        rng.shuffle(tr)
        nprobe = max(1, int(len(tr) * args.probe_frac))
        probe_imgs, bank_imgs = tr[:nprobe], tr[nprobe:]

        # bank từ bank_imgs
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(bank_imgs) * T * T))
        with torch.no_grad():
            buf = []
            for pth in bank_imgs:
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

        # ĐÀO rare-normal từ probe_imgs: patch distance cao -> rare-normal
        rn_feats = []
        with torch.no_grad():
            for pth in probe_imgs:
                g = img_featmap(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                fl = g.reshape(-1, Cdim)
                d = nn_dist(fl, bank)
                thr = torch.quantile(d, args.rn_q / 100.0)
                sel = fl[d >= thr]
                rn_feats.append(sel.cpu().numpy())
        rn_feats = np.concatenate(rn_feats, 0) if rn_feats else np.zeros((0, Cdim), np.float32)
        if rn_feats.shape[0] > args.rn_max:
            idx = rng.choice(rn_feats.shape[0], args.rn_max, replace=False)
            rn_feats = rn_feats[idx]

        # test set + split shots
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

        # nhãn từ shots
        Xs, ys = [], []
        for f, g in zip(sp_feat, sp_gt):
            Xs.append(f.reshape(-1, Cdim)); ys.append(g.reshape(-1))
        Xsh = np.concatenate(Xs); ysh = np.concatenate(ys).astype(int)
        if ysh.sum() < 3:
            p(f'  [{cat}] thiếu defect pixel, bỏ'); continue

        # global-norm distance (khớp submission)
        dall = np.stack(ev_d, 0); lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)
        ev_dr = [(d - lo) / (hi - lo + 1e-8) for d in ev_d]

        # ---- BASE head (chỉ shots) ----
        clf_base = fit_head(Xsh, ysh, None, args.pca, args.seed)
        # ---- RNM head (shots + rare-normal mined negatives) ----
        Xr = np.concatenate([Xsh, rn_feats], 0)
        yr = np.concatenate([ysh, np.zeros(rn_feats.shape[0], int)], 0)
        wr = np.concatenate([np.ones(Xsh.shape[0]), np.full(rn_feats.shape[0], args.rn_weight)])
        clf_rnm = fit_head(Xr, yr, wr, args.pca, args.seed)
        p(f'  [{cat}] rare-normal mined: {rn_feats.shape[0]} | defect px: {ysh.sum()}')

        for tag, clf in [('BASE', clf_base), ('RNM', clf_rnm)]:
            ev_pr = [clf.predict_proba(f.reshape(-1, Cdim))[:, 1].reshape(f.shape[0], f.shape[0]) for f in ev_feat]
            for hw in args.head_w:
                maps = [(1 - hw) * dr + hw * pr for dr, pr in zip(ev_dr, ev_pr)]
                rr = evaluate_set(maps, ev_gt, gk, device, morph=args.morph)
                agg.setdefault((tag, hw), []).append(rr)
                p(f'  [{cat}] {tag} hw{hw:.2f} AUPRO05={rr[7]:.4f} F1={rr[5]:.4f}')

    p('\n' + '=' * 84)
    p('{:<16}{:>12}{:>12}'.format('head/head_w', 'AUPRO0.05', 'P-F1max'))
    rows = []
    for key in sorted(agg.keys(), key=lambda x: (x[0], x[1])):
        m = np.array(agg[key]).mean(0)
        rows.append((f'{key[0]}-hw{key[1]:.2f}', m[7], m[5]))
        p('{:<16}{:>12.4f}{:>12.4f}'.format(f'{key[0]}-hw{key[1]:.2f}', m[7], m[5]))
    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('head_headw,AUPRO0.05,P-F1max\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
    p(f'\nĐã lưu: {csv} | model={args.model}')
    p('ĐỌC: RNM > BASE ở AUPRO0.05/F1 (nhất là can/wallplugs) -> rare-normal mining = NOVELTY method thật.')


if __name__ == '__main__':
    main()
