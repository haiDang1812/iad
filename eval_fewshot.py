# eval_fewshot.py
# -----------------------------------------------------------------------------
# NOVELTY (grounded từ diag10+diag11): unsup distance có TRẦN FP ở rare-normal
# (HARD AUROC 0.45) mà SUPERVISED gỡ được (oracle HARD 0.93). -> dùng ÍT nhãn defect
# (weakly-supervised) học hướng phân biệt rare-normal/defect -> phá trần -> đẩy SegF1 (+AUPRO).
#
# Đo k-shot curve: với k = 0 (unsup) / 1 / 5 / 10 ảnh defect có mask:
#   - UNSUP   : memory-bank distance (baseline)
#   - HEAD    : logistic head học trên k ảnh nhãn (frozen feature)
#   - FUSE    : kết hợp (distance rank + head prob)
# Báo AUPRO0.05 + P-F1max (proxy SegF1) trên tập eval (disjoint với shot).
#
# Split: test_public/bad chia k ảnh "shot" (train head) + phần còn lại + good = eval.
# (Đây là thí nghiệm chứng minh giá trị ít-nhãn; KHÔNG phải submission unsup.)
#
# Chạy:
#   python eval_fewshot.py --data_path ../data --tiles 2 --shots 0 1 5 10 --out_dir ./diagnosis_fewshot
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

from models import vit_encoder
from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger

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
    # trả feature theo lưới [T*ts, T*ts, C] (đã ghép tile) + side
    ts = args.tile_res // 14
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), args.enc_batch):
        b = torch.stack([to_tensor(t, args.tile_res) for t in tiles[s:s + args.enc_batch]])
        fl.append(extract(encoder, b, args.layers, n_reg, device))
    f = torch.cat(fl, 0)                              # [T*T, ts*ts, C]
    C = f.shape[-1]
    grid = torch.zeros(T * ts, T * ts, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = f[k].reshape(ts, ts, C)
    return grid                                       # [G, G, C]


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
    if morph > 0:                                   # dọn FP lẻ / làm liền vùng -> precision (SegF1)
        from scipy import ndimage
        pr = np.stack([ndimage.grey_closing(m, size=(morph, morph)) for m in pr], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * r))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    return ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)


def main():
    ap = argparse.ArgumentParser('Few-shot weakly-supervised head (grounded từ diag11)')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--tile_res', type=int, default=392)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=32)
    ap.add_argument('--shots', type=int, nargs='+', default=[0, 1, 5, 10])
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--morph_close', type=int, default=0, help='kernel morphological closing (đẩy precision/SegF1)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diagnosis_fewshot')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fewshot', args.out_dir).info
    p('=' * 78)
    p(f'FEW-SHOT | tiles={args.tiles} (eff {args.tiles*args.tile_res}) | shots={args.shots}')
    p('UNSUP=distance | HEAD=logistic(k ảnh nhãn) | FUSE=rank(d)+prob. Báo AUPRO0.05 & P-F1max.')
    p('=' * 78)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles
    rng = np.random.default_rng(args.seed)

    agg = {}  # (shot, branch) -> list of metric rows
    for cat in args.categories:
        # bank unsup
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

        # test set
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        maxk = max(args.shots)
        shot_pool = bad[:maxk]               # ảnh defect dùng làm shot (tối đa)
        eval_bad = bad[maxk:]                # eval trên bad còn lại + good (cố định giữa các k)
        eval_idx = eval_bad + good

        # precompute feature grid + unsup map + gt cho eval + shot pool
        def prep(idx):
            grid = img_featmap(encoder, Image.open(ds.img_paths[idx]), T, args, n_reg, device)
            d = nn_map(grid, bank, device).cpu().numpy()
            g = gt_grid(ds.gt_paths[idx], ds.labels[idx], grid.shape[0])
            return grid.cpu().numpy(), d, g
        ev = [prep(i) for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval')]
        sp_feat = [img_featmap(encoder, Image.open(ds.img_paths[i]), T, args, n_reg, device).cpu().numpy()
                   for i in shot_pool]
        sp_gt = [gt_grid(ds.gt_paths[i], 1, ev[0][0].shape[0]) for i in shot_pool]

        ev_d = [e[1] for e in ev]
        ev_gt = [e[2] for e in ev]
        Cdim = ev[0][0].shape[-1]

        for k in args.shots:
            # UNSUP (k irrelevant nhưng báo ở k=0)
            if k == 0:
                agg.setdefault((0, 'UNSUP'), []).append(evaluate_set(ev_d, ev_gt, gk, device, morph=args.morph_close))
                continue
            # train head trên k ảnh shot
            Xs, ys = [], []
            for f, g in zip(sp_feat[:k], sp_gt[:k]):
                Xs.append(f.reshape(-1, Cdim)); ys.append(g.reshape(-1))
            X = np.concatenate(Xs); y = np.concatenate(ys).astype(int)
            if y.sum() < 3 or (1 - y).sum() < 3:
                continue
            clf = make_pipeline(StandardScaler(), PCA(n_components=min(args.pca, X.shape[1], X.shape[0]-1)),
                                LogisticRegression(max_iter=2000, class_weight='balanced'))
            clf.fit(X, y)
            # head score map + fuse trên eval
            head_maps, fuse_maps, fmult_maps = [], [], []
            for f, d in zip([e[0] for e in ev], ev_d):
                G = f.shape[0]
                prob = clf.predict_proba(f.reshape(-1, Cdim))[:, 1].reshape(G, G)
                head_maps.append(prob)
                dr = (d - d.min()) / (d.max() - d.min() + 1e-8)
                pr = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
                fuse_maps.append(0.5 * dr + 0.5 * pr)
                fmult_maps.append(dr * pr)                       # precision-focused (cả 2 phải đồng ý)
            agg.setdefault((k, 'HEAD'), []).append(evaluate_set(head_maps, ev_gt, gk, device, morph=args.morph_close))
            agg.setdefault((k, 'FUSE'), []).append(evaluate_set(fuse_maps, ev_gt, gk, device, morph=args.morph_close))
            agg.setdefault((k, 'FMULT'), []).append(evaluate_set(fmult_maps, ev_gt, gk, device, morph=args.morph_close))

    p('\n' + '=' * 78)
    p('{:<14}{:>12}{:>12}'.format('shot/branch', 'AUPRO0.05', 'P-F1max'))
    keys = sorted(agg.keys(), key=lambda x: (x[0], x[1]))
    rows = []
    for key in keys:
        m = np.array(agg[key]).mean(0)
        rows.append((f'{key[0]}-{key[1]}', m[7], m[5]))
        p('{:<14}{:>12.4f}{:>12.4f}'.format(f'{key[0]}-{key[1]}', m[7], m[5]))
    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as fcsv:
        fcsv.write('shot_branch,AUPRO0.05,P-F1max\n')
        for r in rows:
            fcsv.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')
    p(f'\nĐã lưu: {csv}')
    p('ĐỌC: HEAD/FUSE ở k nhỏ có vượt 0-UNSUP không (nhất là P-F1max=proxy SegF1). '
      'Lift mạnh -> few-shot phá trần FP = NOVELTY.')


if __name__ == '__main__':
    main()
