# eval_fewshot_adaptive.py
# -----------------------------------------------------------------------------
# CATEGORY-ADAPTIVE operating point (từ phát hiện: branch tốt nhất KHÁC nhau theo cat —
# rice cần UNSUP, fabric cần HEAD, ...). Mỗi category TỰ chọn branch bằng k-fold trên
# chính k ảnh shot (KHÔNG đụng eval -> không leak), rồi áp branch đó lên eval.
#
# Candidate branches: UNSUP / HEAD / FUSE / MIX(head_w) / FMULT.
# Selection: k-fold trên shot -> out-of-fold pixel P-F1max (rẻ, không cần AUPRO) -> argmax.
#   (--select_by f1  : chọn theo SegF1  [mặc định]
#    --select_by aupro: chọn theo AUPRO0.05 trên OOF — chậm hơn)
#
# Báo: từng branch cố định (mean) + ADAPTIVE (mỗi cat lấy branch tự chọn) — AUPRO0.05 & P-F1max.
#
# Chạy:
#   python eval_fewshot_adaptive.py --data_path ../data --shots 10 --head_w 0.7 --morph_close 3 --out_dir ./diag_adaptive
# -----------------------------------------------------------------------------

import os
import glob
import argparse
import warnings

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
from sklearn.metrics import precision_recall_curve

from models import vit_encoder
from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
CANDIDATES = ['UNSUP', 'HEAD', 'FUSE', 'MIX', 'FMULT']
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


@torch.no_grad()
def extract(encoder, imgs, layers, n_reg, device):
    x = encoder.prepare_tokens(imgs.to(device))
    feats, last = [], max(layers)
    for i, blk in enumerate(encoder.blocks):
        if i <= last:
            x = blk(x)
        if i in layers:
            feats.append(x[:, 1 + n_reg:, :])
    return torch.stack(feats, dim=1).mean(dim=1)


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def img_featmap(encoder, pil, T, args, n_reg, device):
    ts = args.tile_res // 14
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), args.enc_batch):
        b = torch.stack([to_tensor(t, args.tile_res) for t in tiles[s:s + args.enc_batch]])
        fl.append(extract(encoder, b, args.layers, n_reg, device))
    f = torch.cat(fl, 0)
    C = f.shape[-1]
    grid = torch.zeros(T * ts, T * ts, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = f[k].reshape(ts, ts, C)
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


def _proc(maps, gts, gk, device, resize, morph):
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    if morph > 0:
        from scipy import ndimage
        pr = np.stack([ndimage.grey_closing(m, size=(morph, morph)) for m in pr], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    return pr, gt


def evaluate_set(maps, gts, gk, device, resize=256, r=0.01, morph=0):
    pr, gt = _proc(maps, gts, gk, device, resize, morph)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * r))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    return ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)


def f1max_pix(maps, gts, gk, device, resize=256, morph=0):
    """P-F1max rẻ (không tính AUPRO) — dùng cho selection trên OOF shot."""
    pr, gt = _proc(maps, gts, gk, device, resize, morph)
    yp = pr.reshape(-1); yt = (gt.reshape(-1) > 0).astype(int)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return 0.0
    prec, rec, _ = precision_recall_curve(yt, yp)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(np.nanmax(f1))


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def combine(branch, dr, pr, head_w):
    if branch == 'UNSUP':
        return dr
    if branch == 'HEAD':
        return pr
    if branch == 'FUSE':
        return 0.5 * dr + 0.5 * pr
    if branch == 'MIX':
        return head_w * pr + (1 - head_w) * dr
    if branch == 'FMULT':
        return dr * pr
    raise ValueError(branch)


def fit_head(feats, gts, Cdim, args):
    Xs, ys = [], []
    for f, g in zip(feats, gts):
        Xs.append(f.reshape(-1, Cdim)); ys.append(g.reshape(-1))
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(int)
    if y.sum() < 3 or (1 - y).sum() < 3:
        return None
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0] - 1)),
                        LogisticRegression(max_iter=2000, class_weight='balanced'))
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser('Category-adaptive few-shot operating point')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--tile_res', type=int, default=392)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=32)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--head_w', type=float, default=0.7)
    ap.add_argument('--morph_close', type=int, default=3)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--select_by', type=str, default='f1', choices=['f1', 'aupro'])
    ap.add_argument('--candidates', type=str, nargs='+', default=CANDIDATES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag_adaptive')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('adaptive', args.out_dir).info
    p('=' * 80)
    p(f'CATEGORY-ADAPTIVE | tiles={args.tiles} | k={args.shots} | head_w={args.head_w} | morph={args.morph_close}')
    p(f'candidates={args.candidates} | select_by={args.select_by} | folds={args.folds}')
    p('=' * 80)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles
    rng = np.random.default_rng(args.seed)
    k = args.shots

    fixed = {}            # branch -> list of (aupro05, f1) per cat
    adaptive = []         # list of (aupro05, f1) per cat (selected branch)
    sel_log = []          # (cat, selected_branch)

    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        buf, acc, keep = [], [], max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            for pth in tr:
                for t in tile_pils(Image.open(pth), T):
                    buf.append(to_tensor(t, args.tile_res))
                    if len(buf) >= args.enc_batch:
                        f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
                        acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu()); buf.clear()
            if buf:
                f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        shot_pool = bad[:k]
        eval_idx = bad[k:] + good

        def prep(idx, lab):
            grid = img_featmap(encoder, Image.open(ds.img_paths[idx]), T, args, n_reg, device)
            d = nn_map(grid, bank, device).cpu().numpy()
            g = gt_grid(ds.gt_paths[idx], lab, grid.shape[0])
            return grid.cpu().numpy(), d, g

        ev = [prep(i, ds.labels[i]) for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval')]
        sp = [prep(i, 1) for i in shot_pool]
        Cdim = ev[0][0].shape[-1]
        sp_feat = [s[0] for s in sp]; sp_dr = [norm01(s[1]) for s in sp]; sp_gt = [s[2] for s in sp]
        ev_feat = [e[0] for e in ev]; ev_d = [e[1] for e in ev]; ev_gt = [e[2] for e in ev]

        # ---- SELECTION: k-fold OOF trên shot ----
        nf = min(args.folds, k)
        sel = 'FUSE'
        if nf >= 2:
            oof_prob = [None] * k
            kf = KFold(n_splits=nf, shuffle=True, random_state=args.seed)
            for tr_i, va_i in kf.split(range(k)):
                clf = fit_head([sp_feat[i] for i in tr_i], [sp_gt[i] for i in tr_i], Cdim, args)
                if clf is None:
                    continue
                for i in va_i:
                    G = sp_feat[i].shape[0]
                    oof_prob[i] = clf.predict_proba(sp_feat[i].reshape(-1, Cdim))[:, 1].reshape(G, G)
            ok = [i for i in range(k) if oof_prob[i] is not None]
            scores = {}
            for bn in args.candidates:
                maps = []
                for i in ok:
                    dr = sp_dr[i]
                    pr = norm01(oof_prob[i]) if oof_prob[i] is not None else np.zeros_like(dr)
                    maps.append(combine(bn, dr, pr, args.head_w))
                gts = [sp_gt[i] for i in ok]
                if args.select_by == 'f1':
                    scores[bn] = f1max_pix(maps, gts, gk, device, morph=args.morph_close)
                else:
                    scores[bn] = evaluate_set(maps, gts, gk, device, morph=args.morph_close)[7]
            sel = max(scores, key=scores.get)
            p(f'  [{cat}] OOF {args.select_by}: ' +
              ' '.join(f'{b}={scores[b]:.3f}' for b in args.candidates) + f'  -> chọn {sel}')
        else:
            p(f'  [{cat}] k<2, fallback FUSE')
        sel_log.append((cat, sel))

        # ---- EVAL: head cuối (train all k) + tính mọi branch ----
        clf = fit_head(sp_feat, sp_gt, Cdim, args)
        ev_branch = {}
        for bn in args.candidates:
            if bn == 'UNSUP':
                ev_branch[bn] = list(ev_d)            # raw distance (như baseline UNSUP)
                continue
            maps = []
            for f, d in zip(ev_feat, ev_d):
                G = f.shape[0]
                prob = clf.predict_proba(f.reshape(-1, Cdim))[:, 1].reshape(G, G)
                maps.append(combine(bn, norm01(d), norm01(prob), args.head_w))
            ev_branch[bn] = maps
        p(f'  [{cat}] computing AUPRO eval (CPU)...')
        cat_res = {}
        for bn in args.candidates:
            rr = evaluate_set(ev_branch[bn], ev_gt, gk, device, morph=args.morph_close)
            cat_res[bn] = (rr[7], rr[5])
            fixed.setdefault(bn, []).append((rr[7], rr[5]))
            mark = ' <== ADAPTIVE' if bn == sel else ''
            p(f'  [{cat}] {bn:<5} AUPRO05={rr[7]:.4f} F1={rr[5]:.4f}{mark}')
        adaptive.append(cat_res[sel])

    # ---- TỔNG HỢP ----
    p('\n' + '=' * 80)
    p('{:<18}{:>12}{:>12}'.format('branch', 'AUPRO0.05', 'P-F1max'))
    rows = []
    for bn in args.candidates:
        m = np.array(fixed[bn]).mean(0)
        rows.append((f'fixed-{bn}', m[0], m[1]))
        p('{:<18}{:>12.4f}{:>12.4f}'.format(f'fixed-{bn}', m[0], m[1]))
    ma = np.array(adaptive).mean(0)
    rows.append(('ADAPTIVE', ma[0], ma[1]))
    p('{:<18}{:>12.4f}{:>12.4f}   <== category-adaptive'.format('ADAPTIVE', ma[0], ma[1]))
    p('\nBranch chọn theo cat: ' + ', '.join(f'{c}:{b}' for c, b in sel_log))

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as fcsv:
        fcsv.write('branch,AUPRO0.05,P-F1max\n')
        for r in rows:
            fcsv.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
        fcsv.write('\ncat,selected_branch\n')
        for c, b in sel_log:
            fcsv.write(f'{c},{b}\n')
    p(f'\nĐã lưu: {csv}')
    p('ĐỌC: ADAPTIVE có > mọi fixed branch ở P-F1max (và giữ AUPRO) không. '
      'Nếu có -> category-adaptive là đòn bẩy + contribution.')


if __name__ == '__main__':
    main()
