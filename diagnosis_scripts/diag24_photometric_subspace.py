# diag24_photometric_subspace.py
# -----------------------------------------------------------------------------
# PREMISE TEST cho method novel "Photometric-invariant feature distance".
#
# GIẢ THUYẾT: light-shift private làm AUPRO tụt vì feature DINOv3 của patch NORMAL
#   trôi theo MỘT nuisance subspace HẠNG THẤP (đổi sáng => dịch chuyển feature nhất
#   quán). Nếu đúng, CHIẾU BỎ subspace đó khỏi bank+test trước khi tính distance ->
#   normal hết phình dưới shift, defect vẫn nổi -> distance robust với shift.
#
# 3 CÂU HỎI (không cần submit, chạy 1-2 category):
#   Q1 LOW-RANK  : dịch chuyển feature dưới photometric-aug có tập trung vào vài chiều?
#                  (explained variance của top-k singular directions của D = Fs - F0)
#   Q2 CONSISTENT: hướng dịch chuyển có NHẤT QUÁN giữa các patch? (|cos| tới mean-dir)
#   Q3 SURVIVES  : chiếu bỏ subspace -> tách defect/normal (patch AUROC) DƯỚI SHIFT có
#                  TỐT HƠN raw không, mà KHÔNG giết tách trên clean?
#
# QUYẾT ĐỊNH: Q1 cao (top-5 >~0.7) + Q2 cao (>~0.6) + Q3 shift-proj > shift-raw
#   => method đứng vững, đáng dựng. Ngược lại => bỏ, và "shift không hạng-thấp" là insight.
#
# Parity: import backbone + feature extractor production (không lệch pipeline).
#   python diag24_photometric_subspace.py --data_path ../data --categories fruit_jelly vial
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

# leo lên từ thư mục script tới folder chứa pipeline (iad/) dù đang ở folder con
_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D); break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                      # noqa: E402
    load_backbone, img_featgrid, build_bank, nn_map, gt_grid, VALID,
)
from dataset import MVTecAD2Dataset                        # noqa: E402
from utils import get_logger                               # noqa: E402

warnings.filterwarnings('ignore')
IMG_EXT = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.PNG', '*.JPG']


def photometric_shift(pil, s):
    """Khớp diag22/eval_segf1_resolution: brightness/contrast/color/gamma mức s>=0."""
    if s <= 0:
        return pil.convert('RGB')
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


def auroc(scores, labels):
    """AUROC = Mann-Whitney U (score cao = anomaly hơn). labels in {0,1}."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def proj_out(x, U):
    """Chiếu BỎ subspace U (C x k, cột trực chuẩn): x - (x U) U^T."""
    return x - (x @ U) @ U.t()


def main():
    ap = argparse.ArgumentParser('diag24: premise test photometric nuisance subspace')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--bank_size', type=int, default=30000)
    ap.add_argument('--n_train', type=int, default=16, help='ảnh train/good: build bank + đo dịch chuyển')
    ap.add_argument('--n_test', type=int, default=24, help='ảnh test defect: đo tách defect/normal')
    ap.add_argument('--patch_per_img', type=int, default=400, help='subsample patch/ảnh khi gom D')
    ap.add_argument('--shift_levels', type=float, nargs='+', default=[0.3, 0.6, 0.9, 1.2])
    ap.add_argument('--test_shift', type=float, default=0.8, help='mức shift dùng cho Q3')
    ap.add_argument('--k_sub', type=int, default=5, help='số chiều subspace chiếu bỏ (Q3)')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag24')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag24', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bb = load_backbone(args.model, device)
    T, gt, R, L = args.tiles, args.grid_tile, args.grid_tile * bb.patch, args.layers
    rng = np.random.default_rng(0)
    p(f'device={device} model={args.model} tiles={T} grid_tile={gt} layers={L} k_sub={args.k_sub}')

    for cat in args.categories:
        p(f'\n===== [{cat}] =====')
        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ'); continue
        tr_disp = tr[:args.n_train]
        bank = build_bank(bb, tr, T, R, gt, L, args.enc_batch, args.bank_size, device)
        Cdim = bank.shape[-1]

        # ---- Q1/Q2: gom dịch chuyển D = Fs - F0 trên patch train/good ----
        disp = []
        for pth in tqdm(tr_disp, ncols=70, desc=f'  {cat}/disp', leave=False):
            pil0 = Image.open(pth).convert('RGB')
            F0 = img_featgrid(bb, pil0, T, R, gt, L, args.enc_batch).reshape(-1, Cdim)  # (P,C) on device
            P = F0.shape[0]
            sel = torch.from_numpy(rng.choice(P, size=min(args.patch_per_img, P), replace=False)).to(device)
            F0s = F0[sel]
            for s in args.shift_levels:
                Fs = img_featgrid(bb, photometric_shift(pil0, s), T, R, gt, L, args.enc_batch).reshape(-1, Cdim)[sel]
                disp.append((Fs - F0s).cpu())
        D = torch.cat(disp, 0).to(device)                          # (M,C) dịch chuyển thô
        M = D.shape[0]

        # SVD hướng dịch chuyển (KHÔNG center: giữ common-mode = hướng đổi-sáng chính)
        A = (D.t() @ D) / M                                        # C x C (giống bậc 2 singular)
        evals, evecs = torch.linalg.eigh(A)                        # tăng dần
        evals = torch.clamp(evals.flip(0), min=0.0)                # giảm dần
        evecs = evecs.flip(1)                                      # cột theo evals giảm dần
        ev_ratio = (evals / (evals.sum() + 1e-12)).cpu().numpy()
        cum = np.cumsum(ev_ratio)
        # Q2: nhất quán hướng — |cos| của từng dịch chuyển (unit) tới mean-direction
        dbar = D.mean(0); dbar = dbar / (dbar.norm() + 1e-8)
        Du = D / (D.norm(dim=1, keepdim=True) + 1e-8)
        cos_mean = float((Du @ dbar).abs().mean().cpu())
        p(f'  Q1 LOW-RANK   explvar top1={ev_ratio[0]:.3f} top3={cum[2]:.3f} '
          f'top5={cum[4]:.3f} top10={cum[9]:.3f}  (M={M}, C={Cdim})')
        p(f'  Q2 CONSISTENT mean|cos(disp, mean-dir)|={cos_mean:.3f}')

        # basis subspace nhiễu (trực chuẩn từ eigh) -> chiếu bỏ
        U = evecs[:, :args.k_sub].contiguous()                     # C x k
        bank_p = proj_out(bank, U)

        # ---- Q3: tách defect/normal patch, {clean,shift} x {raw,proj} ----
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1][:args.n_test]
        acc = {kk: {'d': [], 'y': []} for kk in ('clean_raw', 'clean_proj', 'shift_raw', 'shift_proj')}
        for i in tqdm(bad, ncols=70, desc=f'  {cat}/test', leave=False):
            pil0 = Image.open(ds.img_paths[i]).convert('RGB')
            for cond, sft in (('clean', 0.0), ('shift', args.test_shift)):
                grid = img_featgrid(bb, photometric_shift(pil0, sft), T, R, gt, L, args.enc_batch)
                G = grid.shape[0]
                y = gt_grid(ds.gt_paths[i], 1, G).reshape(-1)
                d_raw = nn_map(grid, bank, device).reshape(-1)
                gp = proj_out(grid.reshape(-1, Cdim), U).reshape(G, G, Cdim)
                d_proj = nn_map(gp, bank_p, device).reshape(-1)
                acc[f'{cond}_raw']['d'].append(d_raw); acc[f'{cond}_raw']['y'].append(y)
                acc[f'{cond}_proj']['d'].append(d_proj); acc[f'{cond}_proj']['y'].append(y)

        res = {}
        for kk, v in acc.items():
            res[kk] = auroc(np.concatenate(v['d']), np.concatenate(v['y']))
        p(f'  Q3 patch-AUROC(defect vs normal):')
        p(f'       clean:  raw={res["clean_raw"]:.4f}  proj={res["clean_proj"]:.4f}  '
          f'(Δ={res["clean_proj"]-res["clean_raw"]:+.4f})')
        p(f'       shift:  raw={res["shift_raw"]:.4f}  proj={res["shift_proj"]:.4f}  '
          f'(Δ={res["shift_proj"]-res["shift_raw"]:+.4f})')

        # verdict tự động
        q1 = cum[4] >= 0.70
        q2 = cos_mean >= 0.60
        q3 = (res['shift_proj'] > res['shift_raw'] + 0.005) and (res['clean_proj'] >= res['clean_raw'] - 0.01)
        verd = 'ĐÁNG DỰNG' if (q1 and q2 and q3) else 'YẾU'
        p(f'  => VERDICT [{cat}]: Q1={q1} Q2={q2} Q3={q3}  -> {verd}')

    p('\nXONG. Q1 top5>=0.70 + Q2>=0.60 + Q3(shift-proj>shift-raw, clean không tụt) => method novel đứng vững.')


if __name__ == '__main__':
    main()
