# diag18_defect_synth_premise.py
# -----------------------------------------------------------------------------
# Test PREMISE của novelty "subspace-guided defect feature synthesis":
#   Defect nhỏ/lạ bị miss vì few-shot quá ít ví dụ defect (diag16: can=4 px).
#   Ý: từ ít defect thật, NGOẠI SUY thêm defect TỔNG HỢP trong feature-space,
#      dẫn hướng bởi defect-PCA directions, RÀNG buộc trong complement (bỏ nuisance subspace NSP)
#      -> head thấy nhiều defect đa dạng -> nhận ra defect chưa thấy -> recall lên.
#
# Premise test (leave-out): chia 10 shot -> train-shots vs held-out-shots.
#   real-only head: train trên defect của train-shots.
#   synth head    : train trên defect train-shots + defect TỔNG HỢP.
#   Đo: nhận diện defect của HELD-OUT shots (AUROC vs normal, recall@FPR5).
#   synth > real-only -> synthesis giúp generalize -> build full. Không -> bỏ.
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag18_defect_synth_premise.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --holdout 3 --n_synth 5000 --out_dir ./diag18
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
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from dataset import MVTecAD2Dataset
from utils import get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
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
def img_flat(bb, pil, T, R, gt, layers, eb):
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
    return grid.reshape(-1, C)


def gt_flat(gpath, G):
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8).reshape(-1)


def fit_head(Xpos, Xneg, pca):
    X = np.concatenate([Xpos, Xneg]); y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))]).astype(int)
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(pca, X.shape[1], X.shape[0] - 1)),
                        LogisticRegression(max_iter=2000, class_weight='balanced'))
    clf.fit(X, y)
    return clf


def evalrec(clf, Xheld, Xnorm):
    if len(Xheld) < 1:
        return float('nan'), float('nan')
    ph = clf.predict_proba(Xheld)[:, 1]
    pn = clf.predict_proba(Xnorm)[:, 1]
    y = np.concatenate([np.ones(len(ph)), np.zeros(len(pn))])
    s = np.concatenate([ph, pn])
    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan')
    thr = np.percentile(pn, 95)            # FPR=5%
    rec = float((ph >= thr).mean())
    return auc, rec


def main():
    ap = argparse.ArgumentParser('Diag18: defect feature-synthesis premise')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=128)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--holdout', type=int, default=3, help='số shot giữ làm held-out defect')
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--n_synth', type=int, default=5000)
    ap.add_argument('--defect_dirs', type=int, default=10, help='số hướng defect-PCA để ngoại suy')
    ap.add_argument('--synth_scale', type=float, default=1.0)
    ap.add_argument('--nuisance_rank', type=int, default=40, help='rank subspace rare-normal để bỏ (0=tắt)')
    ap.add_argument('--rn_q', type=float, default=95.0)
    ap.add_argument('--n_neg', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag18')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag18', args.out_dir).info
    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    T = args.tiles; gt = args.grid_tile
    rng = np.random.default_rng(args.seed)
    p('=' * 84)
    p(f'DIAG18 defect-synth premise | model={args.model} eff_grid={T*gt} | shots={args.shots} holdout={args.holdout} '
      f'| n_synth={args.n_synth} dirs={args.defect_dirs} scale={args.synth_scale} nui_rank={args.nuisance_rank}')
    p('real-only vs synth-augmented head; recall defect HELD-OUT (AUROC vs normal, recall@FPR5)')
    p('=' * 84)

    rows = []
    for cat in args.categories:
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            for pth in tr:
                fg = img_flat(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                acc.append(subsample(fg, keep).cpu())
        bank = torch.cat(acc, 0)
        bank = subsample(bank, args.bank_size).numpy()
        C = bank.shape[-1]; mu = bank.mean(0, keepdims=True)

        # nuisance subspace (rare-normal) từ validation/good
        Vn = None
        if args.nuisance_rank > 0:
            val_dir = os.path.join(args.data_path, cat, 'validation', 'good')
            probe = sorted(glob.glob(os.path.join(val_dir, '*.png')) + glob.glob(os.path.join(val_dir, '*.jpg')))
            if not probe:
                probe = tr[:max(1, len(tr) // 3)]
            bt = torch.tensor(bank, device=device)
            rnf = []
            with torch.no_grad():
                for pth in probe:
                    fg = img_flat(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
                    d = torch.cdist(fg, bt).min(1)[0]
                    rnf.append(fg[d >= torch.quantile(d, args.rn_q / 100.)].cpu())
            rnf = subsample(torch.cat(rnf, 0), 40000).numpy()
            _, _, Vh = np.linalg.svd(rnf - mu, full_matrices=False)
            Vn = Vh[:args.nuisance_rank]                       # [r, C]

        # shots defect patches theo shot
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng.shuffle(bad)
        shot = bad[:args.shots]
        per_shot = []
        with torch.no_grad():
            for i in shot:
                fg = img_flat(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch).cpu().numpy()
                G = int(round(fg.shape[0] ** 0.5))
                m = gt_flat(ds.gt_paths[i], G)
                df = fg[m == 1]
                if len(df) > 0:
                    per_shot.append(df)
        if len(per_shot) < args.holdout + 2:
            p(f'  [{cat}] quá ít shot có defect, bỏ'); continue
        # split shots
        train_def = np.concatenate(per_shot[:-args.holdout])
        held_def = np.concatenate(per_shot[-args.holdout:])
        if len(train_def) < 3 or len(held_def) < 1:
            p(f'  [{cat}] defect px quá ít, bỏ'); continue

        neg = bank[rng.choice(len(bank), min(args.n_neg, len(bank)), replace=False)]

        # defect-PCA directions (trong complement nếu có nuisance)
        Dc = train_def - mu
        if Vn is not None:
            Dc = Dc - Dc @ Vn.T @ Vn          # bỏ thành phần nuisance
        # PCA của defect-complement
        _, Sd, Vd = np.linalg.svd(Dc - Dc.mean(0, keepdims=True), full_matrices=False)
        k = min(args.defect_dirs, Vd.shape[0])
        Vd = Vd[:k]; sig = (Sd[:k] / np.sqrt(max(1, len(Dc))))   # std mỗi hướng

        # synth: lấy defect thật + nhiễu dọc defect-dirs, rồi bỏ nuisance
        idx = rng.integers(0, len(train_def), args.n_synth)
        base = train_def[idx]
        eps = rng.normal(0, args.synth_scale, size=(args.n_synth, k)) * sig[None, :]
        synth = base + eps @ Vd
        if Vn is not None:
            synth = synth - (synth - mu) @ Vn.T @ Vn          # giữ trong complement

        clf_real = fit_head(train_def, neg, args.pca)
        clf_syn = fit_head(np.concatenate([train_def, synth]), neg, args.pca)

        neg_eval = bank[rng.choice(len(bank), min(20000, len(bank)), replace=False)]
        a_r, r_r = evalrec(clf_real, held_def, neg_eval)
        a_s, r_s = evalrec(clf_syn, held_def, neg_eval)
        p(f'  [{cat:<11}] real AUROC={a_r:.3f} rec@5={r_r:.3f} | synth AUROC={a_s:.3f} rec@5={r_s:.3f} '
          f'| Δrec={r_s-r_r:+.3f}')
        rows.append((cat, a_r, r_r, a_s, r_s))

    p('\n' + '=' * 84)
    p('{:<13}{:>10}{:>10}{:>11}{:>11}{:>9}'.format('cat', 'AUROC_re', 'rec_re', 'AUROC_syn', 'rec_syn', 'Δrec'))
    A = []
    for cat, ar, rr, as_, rs in rows:
        p('{:<13}{:>10.3f}{:>10.3f}{:>11.3f}{:>11.3f}{:>9.3f}'.format(cat, ar, rr, as_, rs, rs - rr))
        A.append([ar, rr, as_, rs])
    A = np.array(A); m = np.nanmean(A, 0)
    p('{:<13}{:>10.3f}{:>10.3f}{:>11.3f}{:>11.3f}{:>9.3f}'.format('MEAN', m[0], m[1], m[2], m[3], m[3] - m[1]))
    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('cat,AUROC_real,rec_real,AUROC_synth,rec_synth\n')
        for cat, ar, rr, as_, rs in rows:
            f.write(f'{cat},{ar:.4f},{rr:.4f},{as_:.4f},{rs:.4f}\n')
    p('\nĐỌC: synth rec@5 / AUROC > real-only (defect HELD-OUT) -> feature-synthesis giúp generalize -> build full.')
    p('     Không hơn -> synthetic feature vô ích -> bỏ ý này.')


if __name__ == '__main__':
    main()
