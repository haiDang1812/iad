# diagnosis10_bank_failure.py
# -----------------------------------------------------------------------------
# DIAGNOSIS (không phải method): memory-bank NN distance FAIL ở đâu tại low-FPR?
#
# Giả thuyết: FP ở low-FPR đến từ "normal HIẾM" (patch normal nhưng d_1 cao) — bank
# không phân biệt được nó với defect. Và tín hiệu relative-isolation d_1/d_k (BOUNDED)
# có thể tách defect khỏi đám "normal-d-cao" đó không?
#
# Đo trên patch-level (frozen DINOv2, 1 scale @crop_res), pool toàn test mỗi category:
#   - AUROC(label ; d_1)            : khả năng tách của NN distance thô (baseline)
#   - AUROC(label ; d_1/d_k)        : relative isolation (anomaly = isolated -> d1/dk cao)
#   - HARD set = defect vs top-q% normal theo d_1 (chính là các FP ở low-FPR):
#       AUROC_hard(d_1)  ~ 0.5 (chồng nhau, đó là lý do FP)
#       AUROC_hard(d1/dk) > 0.5 ?  -> nếu CÓ: relative-isolation gỡ đúng cái fail
#
# Chạy:
#   python diagnosis10_bank_failure.py --data_path ../data --out_dir ./diagnosis/diag10
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
from sklearn.metrics import roc_auc_score

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
def knn(feats, bank, k, chunk=4096):
    # trả d_1 (min) và d_k (k-th nhỏ nhất) cho từng patch
    q = feats.reshape(-1, feats.shape[-1])
    d1 = torch.empty(q.shape[0], device=q.device)
    dk = torch.empty(q.shape[0], device=q.device)
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s:s + chunk], bank)
        vals, _ = torch.topk(d, k, dim=1, largest=False)   # [cs, k] tăng dần
        d1[s:s + chunk] = vals[:, 0]
        dk[s:s + chunk] = vals[:, -1]
    return d1, dk


def main():
    ap = argparse.ArgumentParser('Diagnosis: where does memory-bank distance fail at low-FPR')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--crop_res', type=int, default=392)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=32)
    ap.add_argument('--hard_q', type=float, default=95.0, help='top (100-q)%% normal theo d1 = FP set')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diagnosis/diag10')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger('diag10', args.out_dir)
    p = logger.info
    p('=' * 78)
    p(f'DIAGNOSIS 10 — memory-bank failure @ low-FPR | crop={args.crop_res} k={args.k}')
    p('AUROC patch-level: d1 (NN distance) vs d1/dk (relative isolation, bounded)')
    p('HARD = defect vs top-{:.0f}% normal theo d1 (= các FP ở low-FPR)'.format(100 - args.hard_q))
    p('=' * 78)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    side = args.crop_res // 14

    rows = []
    for cat in args.categories:
        train_paths = (glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                       glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')) +
                       glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.JPG')))
        # build bank
        buf, acc = [], []
        keep = max(64, args.bank_size * 4 // max(1, len(train_paths)))
        with torch.no_grad():
            for pth in train_paths:
                buf.append(to_tensor(Image.open(pth), args.crop_res))
                if len(buf) >= args.enc_batch:
                    f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu()); buf.clear()
            if buf:
                f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)

        # test: pool patch-level d1, dk, label
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                             transform=None, gt_transform=None, phase='test')
        D1, DK, LAB = [], [], []
        with torch.no_grad():
            for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  {cat}'):
                ip, gp, lb = ds.img_paths[idx], ds.gt_paths[idx], ds.labels[idx]
                feats = extract(encoder, to_tensor(Image.open(ip), args.crop_res).unsqueeze(0),
                                args.layers, n_reg, device)
                d1, dk = knn(feats, bank, args.k)
                if lb == 0 or not (isinstance(gp, str) and os.path.exists(gp)):
                    g = np.zeros((side, side), dtype=np.uint8)
                else:
                    gi = Image.open(gp).convert('L').resize((side, side), Image.NEAREST)
                    g = (np.asarray(gi) > 127).astype(np.uint8)
                D1.append(d1.cpu().numpy()); DK.append(dk.cpu().numpy()); LAB.append(g.reshape(-1))
        d1 = np.concatenate(D1); dk = np.concatenate(DK); lab = np.concatenate(LAB)
        reliso = d1 / (dk + 1e-8)

        # full separability
        au_d1 = roc_auc_score(lab, d1)
        au_ri = roc_auc_score(lab, reliso)
        # HARD set: defect vs top-(100-q)% normal theo d1
        norm = lab == 0; dfct = lab == 1
        thr = np.percentile(d1[norm], args.hard_q)
        hard_norm = norm & (d1 >= thr)                       # normal nhưng d1 cao = FP ở low-FPR
        sel = hard_norm | dfct
        y = lab[sel]
        au_d1_h = roc_auc_score(y, d1[sel])
        au_ri_h = roc_auc_score(y, reliso[sel])
        p(f'\n=== {cat.upper()} ===  defect-patch%={100*dfct.mean():.2f}  hard-normal={hard_norm.sum()}')
        p(f'  FULL : AUROC(d1)={au_d1:.4f}  AUROC(d1/dk)={au_ri:.4f}')
        p(f'  HARD : AUROC(d1)={au_d1_h:.4f}  AUROC(d1/dk)={au_ri_h:.4f}   '
          f'{"-> reliso GỠ được FP" if au_ri_h > au_d1_h + 0.02 else "-> reliso KHÔNG giúp"}')
        rows.append((cat, au_d1, au_ri, au_d1_h, au_ri_h))

    p('\n' + '=' * 78)
    p('{:<13}{:>10}{:>12}{:>12}{:>14}'.format('cat', 'AUROC_d1', 'AUROC_d1/dk', 'HARD_d1', 'HARD_d1/dk'))
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    for r in rows:
        p('{:<13}{:>10.4f}{:>12.4f}{:>12.4f}{:>14.4f}'.format(*r))
    p('{:<13}{:>10.4f}{:>12.4f}{:>12.4f}{:>14.4f}'.format('MEAN', *arr.mean(0)))
    p('\nĐỌC: nếu HARD_d1/dk >> HARD_d1 (~0.5) -> relative-isolation tách được FP rare-normal -> đáng build.')
    p('     nếu HARD_d1/dk ~ HARD_d1 ~0.5      -> bank fail không do rare-normal/isolation -> đổi hướng.')


if __name__ == '__main__':
    main()
