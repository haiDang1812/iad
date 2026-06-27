# diagnosis11_oracle_separability.py
# -----------------------------------------------------------------------------
# DIAGNOSIS: rare-normal vs defect có TÁCH ĐƯỢC bằng GIÁM SÁT không (oracle)?
#
# Diag10 cho thấy: distance & relative-isolation KHÔNG tách được "rare-normal" (FP d1 cao)
# khỏi defect (HARD AUROC < 0.5). Câu hỏi: liệu CÓ signal nào (cần nhãn) tách được?
#   - Oracle = fit logistic regression CÓ NHÃN trên feature DINOv2 (PCA), 5-fold CV AUROC.
#   - Đo trên: FULL (mọi normal vs defect) và HARD (top-5% normal theo d1 vs defect).
#
# Kết luận:
#   - HARD oracle AUROC CAO  -> signal tồn tại, chỉ thiếu nhãn -> mở đường few-shot/weakly-sup (NOVELTY).
#   - HARD oracle AUROC ~0.5 -> defect ≡ rare-normal ở patch-level -> fundamental -> analysis paper thuần.
#
# Chạy:
#   python diagnosis11_oracle_separability.py --data_path ../data --out_dir ./diagnosis/diag11
# -----------------------------------------------------------------------------

import os
import glob
import argparse
import warnings

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import vit_encoder
from dataset import MVTecAD2Dataset
from utils import get_logger

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
def nn_d1(feats, bank, chunk=4096):
    q = feats.reshape(-1, feats.shape[-1])
    out = torch.empty(q.shape[0], device=q.device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(1)[0]
    return out


def cv_auroc(X, y, seed=0):
    if y.sum() < 5 or (1 - y).sum() < 5:
        return float('nan')
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    clf = make_pipeline(StandardScaler(), PCA(n_components=n_comp),
                        LogisticRegression(max_iter=2000, class_weight='balanced'))
    try:
        sc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        return float(np.mean(sc))
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser('Diag11: oracle separability rare-normal vs defect')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--crop_res', type=int, default=392)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=32)
    ap.add_argument('--hard_q', type=float, default=95.0)
    ap.add_argument('--max_normal', type=int, default=20000, help='subsample normal patch cho oracle fit')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diagnosis/diag11')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag11', args.out_dir).info
    p('=' * 70)
    p('DIAGNOSIS 11 — ORACLE (logistic, CV) tách rare-normal vs defect?')
    p('FULL = normal vs defect | HARD = top-5% normal theo d1 vs defect')
    p('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    side = args.crop_res // 14

    rows = []
    for cat in args.categories:
        tr = (glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
              glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')) +
              glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.JPG')))
        buf, acc, keep = [], [], max(64, args.bank_size * 4 // max(1, len(tr)))
        with torch.no_grad():
            for pth in tr:
                buf.append(to_tensor(Image.open(pth), args.crop_res))
                if len(buf) >= args.enc_batch:
                    f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu()); buf.clear()
            if buf:
                f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        FE, D1, LAB = [], [], []
        with torch.no_grad():
            for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  {cat}'):
                ip, gp, lb = ds.img_paths[idx], ds.gt_paths[idx], ds.labels[idx]
                fe = extract(encoder, to_tensor(Image.open(ip), args.crop_res).unsqueeze(0),
                             args.layers, n_reg, device)
                d1 = nn_d1(fe, bank)
                if lb == 0 or not (isinstance(gp, str) and os.path.exists(gp)):
                    g = np.zeros((side, side), dtype=np.uint8)
                else:
                    gi = Image.open(gp).convert('L').resize((side, side), Image.NEAREST)
                    g = (np.asarray(gi) > 127).astype(np.uint8)
                FE.append(fe[0].cpu().numpy()); D1.append(d1.cpu().numpy()); LAB.append(g.reshape(-1))
        FE = np.concatenate(FE, 0); d1 = np.concatenate(D1); lab = np.concatenate(LAB)

        dfct = lab == 1; norm = lab == 0
        # FULL oracle (subsample normal)
        ni = np.where(norm)[0]
        rng = np.random.default_rng(0)
        ni = rng.choice(ni, size=min(args.max_normal, len(ni)), replace=False)
        di = np.where(dfct)[0]
        idx_full = np.concatenate([ni, di])
        au_full = cv_auroc(FE[idx_full], lab[idx_full].astype(int))
        # HARD oracle: top-5% normal theo d1 vs defect
        thr = np.percentile(d1[norm], args.hard_q)
        hard_ni = np.where(norm & (d1 >= thr))[0]
        idx_hard = np.concatenate([hard_ni, di])
        au_hard = cv_auroc(FE[idx_hard], lab[idx_hard].astype(int))

        p(f'\n=== {cat.upper()} === defect%={100*dfct.mean():.2f}')
        p(f'  ORACLE  FULL={au_full:.4f}   HARD={au_hard:.4f}   '
          f'{"-> signal TỒN TẠI (cần nhãn)" if au_hard > 0.7 else "-> HARD khó cả với oracle"}')
        rows.append((cat, au_full, au_hard))

    p('\n' + '=' * 70)
    p('{:<13}{:>14}{:>14}'.format('cat', 'ORACLE_FULL', 'ORACLE_HARD'))
    arr = np.array([[r[1], r[2]] for r in rows])
    for r in rows:
        p('{:<13}{:>14.4f}{:>14.4f}'.format(*r))
    p('{:<13}{:>14.4f}{:>14.4f}'.format('MEAN', *np.nanmean(arr, 0)))
    p('\nĐỌC: ORACLE_HARD cao (>0.7) -> rare-normal vs defect TÁCH ĐƯỢC bằng giám sát')
    p('     -> mở đường few-shot/weakly-supervised (dùng ít nhãn) = NOVELTY có cơ sở.')
    p('     ORACLE_HARD ~0.5 -> fundamental, không tách được -> analysis paper thuần.')


if __name__ == '__main__':
    main()
