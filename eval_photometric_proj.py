# eval_photometric_proj.py
# -----------------------------------------------------------------------------
# ĐO METRIC THẬT (AUPRO0.05) cho method novel "Photometric-invariant distance".
# diag24 đã xác nhận (proxy patch-AUROC): shift làm feature normal trôi theo nuisance
#   subspace hạng-thấp; CHIẾU BỎ subspace -> phục hồi tách defect dưới shift (vial
#   0.30->0.89, can 0.87->0.98). Script này đo trên metric server (AUPRO0.05) để:
#     (1) xác nhận projection NÂNG AUPRO dưới sim-shift, không giết clean,
#     (2) SWEEP k_sub chọn số chiều bỏ (k lớn cứu vial nhưng giết fruit_jelly/rice).
#
# Đo nhánh DISTANCE (bank NN) raw vs projected -> cô lập đúng cơ chế. global-norm là
#   đơn điệu nên KHÔNG đổi AUPRO -> bỏ, dùng thẳng distance map (up_to gaussian y hệt
#   production). Không head/fuse ở đây (đo sạch tác dụng của projection).
#
#   python eval_photometric_proj.py --data_path ../data --sim_shift --k_list 1 2 3 5
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

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                      # noqa: E402
    build_bank, img_featgrid, nn_map, gt_grid, up_to, VALID, IMG_EXT,
)
from dataset import MVTecAD2Dataset                        # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator  # noqa: E402
from backbones_ext import load_backbone                    # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def photometric_shift(pil, s):
    if s <= 0:
        return pil.convert('RGB')
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


def proj_out(x, U):
    """Chiếu BỎ subspace U (C x k trực chuẩn): x - (x U) U^T."""
    return x - (x @ U) @ U.t()


def nuisance_basis(bb, tr_disp, T, R, gt, layers, eb, shift_levels, patch_per_img, kmax, device, rng):
    """Dựng nuisance subspace: SVD hướng dịch chuyển feature dưới photometric-aug (KHÔNG center)."""
    disp = []
    for pth in tqdm(tr_disp, ncols=70, desc='    basis', leave=False):
        pil0 = Image.open(pth).convert('RGB')
        F0 = img_featgrid(bb, pil0, T, R, gt, layers, eb)
        C = F0.shape[-1]; F0 = F0.reshape(-1, C)
        P = F0.shape[0]
        sel = torch.from_numpy(rng.choice(P, size=min(patch_per_img, P), replace=False)).to(device)
        F0s = F0[sel]
        for s in shift_levels:
            Fs = img_featgrid(bb, photometric_shift(pil0, s), T, R, gt, layers, eb).reshape(-1, C)[sel]
            disp.append((Fs - F0s).cpu())
    D = torch.cat(disp, 0).to(device)
    M, C = D.shape
    A = (D.t() @ D) / M
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals.flip(0), min=0.0)
    evecs = evecs.flip(1)
    ev_ratio = (evals / (evals.sum() + 1e-12)).cpu().numpy()
    return evecs[:, :kmax].contiguous(), ev_ratio


def aupro05(preds, gts):
    """AUPRO0.05 pooled (khớp eval_segf1_resolution)."""
    sp = np.array([float(p.max()) for p in preds])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(preds), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def run_cat(bb, cat, args, gk, device):
    T, gt, R, L = args.tiles, args.grid_tile, args.grid_tile * bb.patch, args.layers
    er = args.eval_res
    rng = np.random.default_rng(args.seed)
    srng = np.random.default_rng(args.seed + 777)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr, T, R, gt, L, args.enc_batch, args.bank_size, device)
    Cdim = bank.shape[-1]
    kmax = max(args.k_list)
    U, ev_ratio = nuisance_basis(bb, tr[:args.n_disp], T, R, gt, L, args.enc_batch,
                                 args.shift_levels, args.patch_per_img, kmax, device, rng)
    bank_pk = {k: proj_out(bank, U[:, :k].contiguous()) for k in args.k_list}

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    eval_idx = bad + good

    conds = [('clean', False)] + ([('shift', True)] if args.sim_shift else [])
    out = {'ev_top': (float(ev_ratio[0]), float(np.cumsum(ev_ratio)[min(kmax, len(ev_ratio)) - 1]))}
    for cname, do_shift in conds:
        methods = ['raw'] + [f'k{k}' for k in args.k_list]
        preds = {m: [] for m in methods}; gts = []
        for i in tqdm(eval_idx, ncols=70, desc=f'    {cat}/{cname}', leave=False):
            pil = Image.open(ds.img_paths[i])
            if do_shift:
                pil = photometric_shift(pil, float(srng.uniform(args.shift_lo, args.shift_hi)))
            grid = img_featgrid(bb, pil, T, R, gt, L, args.enc_batch)
            G = grid.shape[0]
            g0 = gt_grid(ds.gt_paths[i], ds.labels[i], G)
            gg = np.asarray(Image.fromarray(g0).resize((er, er), Image.NEAREST)).astype(np.uint8)
            gts.append(gg)
            preds['raw'].append(up_to(nn_map(grid, bank, device), (er, er), gk, device))
            flat = grid.reshape(-1, Cdim)
            for k in args.k_list:
                gp = proj_out(flat, U[:, :k].contiguous()).reshape(G, G, Cdim)
                preds[f'k{k}'].append(up_to(nn_map(gp, bank_pk[k], device), (er, er), gk, device))
        out[cname] = {m: float(aupro05(preds[m], gts)) for m in methods}
    return out


def main():
    ap = argparse.ArgumentParser('eval AUPRO0.05: photometric-invariant projected distance')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--bank_size', type=int, default=30000)
    ap.add_argument('--n_disp', type=int, default=16, help='ảnh train dựng nuisance basis')
    ap.add_argument('--patch_per_img', type=int, default=400)
    ap.add_argument('--shift_levels', type=float, nargs='+', default=[0.3, 0.6, 0.9, 1.2])
    ap.add_argument('--k_list', type=int, nargs='+', default=[1, 2, 3, 5], help='sweep số chiều nuisance bỏ')
    ap.add_argument('--sim_shift', action='store_true', help='đo thêm điều kiện sim-shift (hetero/ảnh)')
    ap.add_argument('--shift_lo', type=float, default=0.3)
    ap.add_argument('--shift_hi', type=float, default=1.2)
    ap.add_argument('--eval_res', type=int, default=256)
    ap.add_argument('--max_eval', type=int, default=40, help='số ảnh defect và good mỗi bên (AUPRO chậm)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./photproj')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('photproj', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bb = load_backbone(args.model, device)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} model={args.model} k_list={args.k_list} sim_shift={args.sim_shift} '
      f'max_eval={args.max_eval} layers={args.layers}')

    rows = []
    for cat in args.categories:
        r = run_cat(bb, cat, args, gk, device)
        if r is None:
            p(f'  [{cat}] thiếu data -> bỏ'); continue
        rows.append((cat, r))
        line = f'  [{cat}] ev_top1={r["ev_top"][0]:.3f} | clean: ' + \
               ' '.join(f'{m}={r["clean"][m]:.4f}' for m in r['clean'])
        if 'shift' in r:
            line += ' || shift: ' + ' '.join(f'{m}={r["shift"][m]:.4f}' for m in r['shift'])
        p(line)

    if rows:
        p('\n===== MEAN qua category =====')
        methods = ['raw'] + [f'k{k}' for k in args.k_list]
        for cname in (['clean', 'shift'] if any('shift' in r for _, r in rows) else ['clean']):
            valid = [r for _, r in rows if cname in r]
            mean = {m: float(np.mean([r[cname][m] for r in valid])) for m in methods}
            best = max(methods, key=lambda m: mean[m])
            p(f'  {cname}: ' + ' '.join(f'{m}={mean[m]:.4f}' for m in methods) + f'   -> best={best}')
        p('\nĐỌC: raw = pipeline hiện tại. Muốn thấy shift-AUPRO(kX) > shift-AUPRO(raw) mà '
          'clean-AUPRO(kX) không tụt. Chọn k thắng nhất trên shift -> cắm vào submission.')


if __name__ == '__main__':
    main()
