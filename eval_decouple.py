# eval_decouple.py
# -----------------------------------------------------------------------------
# PHÁT HIỆN HỘI TỤ (resolution turnover + multi-scale + overlap): AUPRO0.05 và SegF1 có
#   SỞ THÍCH MAP NGƯỢC NHAU — AUPRO thích MƯỢT (ngữ cảnh, low-FPR ranking), SegF1 thích
#   ĐỈNH SẮC (precision pixel). Không map đơn nào tối ưu cả hai.
#
# UNLOCK: server chấm 2 metric từ 2 FILE RIÊNG (tiff liên tục -> AUPRO; png nhị phân TA cắt
#   -> SegF1). => DECOUPLE: tiff dùng map MƯỢT (sigma lớn), png dùng map SẮC (sigma nhỏ) +
#   test_ksig. Mỗi metric lấy đúng map nó thích -> CẢ HAI đạt đỉnh, không đánh đổi.
#
# Kiểm rẻ (thay overlap scorer đắt bằng Gaussian sweep): trên test_public (có GT), quét
#   sigma smoothing, đo AUPRO0.05 & SegF1@test_ksig ở mỗi sigma. Đọc:
#   - AUPRO cực đại ở sigma LỚN, SegF1 cực đại ở sigma NHỎ => xung khắc xác nhận.
#   - Cặp decoupled (AUPRO@best_sigma_au , SegF1@best_sigma_f1) so single (cùng sigma=4)
#     => lợi ích tách đầu ra. Nếu heavy-smooth đạt ~overlap-AUPRO => khỏi cần overlap.
#
#   python eval_decouple.py --data_path ../data --out_dir ./decouple --tiles 3 --grid_tile 24
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, gt_grid, VALID, IMG_EXT, SMOOTH_RES,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def aupro05(maps, gts):
    sp = np.array([float(m.max()) for m in maps])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(maps), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def segf1_ksig(maps, gts, k):
    P = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    thr = float(P.mean() + k * P.std())
    TP = FP = FN = 0.0
    for m, g in zip(maps, gts):
        pred = m >= thr; gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
    return 2 * TP / (2 * TP + FP + FN + 1e-9)


def run_cat(bb, cat, args, layers, kernels, device):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch; hw = args.head_w; k = args.thr_sigma
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train and len(tr) > args.max_train:
        tr = tr[:args.max_train]
    bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None
    idx = [i for i in bad if i not in set(bad[:args.shots])][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    gts = [gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8) for i in idx]

    # map RAW @SMOOTH_RES (chưa smooth): fuse rồi interpolate, chưa gaussian
    raw = []
    C = bank.shape[-1]
    for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
        g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch)
        d = np.asarray(nn_map(g, bank, device))
        with torch.no_grad():
            pr = torch.sigmoid(head(g.reshape(-1, C))).reshape(g.shape[0], g.shape[0]).cpu().numpy()
        raw.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in raw]), [1, 99])
    fused = []
    for d, pr in raw:
        fm = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr
        t = torch.tensor(fm, device=device)[None, None].float()
        fused.append(F.interpolate(t, size=SMOOTH_RES, mode='bilinear', align_corners=False))  # [1,1,S,S]

    out = {}
    for sig, gk in kernels.items():
        maps = [gk(t)[0, 0].cpu().numpy().astype(np.float32) for t in fused]
        out[sig] = (aupro05(maps, gts), segf1_ksig(maps, gts, k))
    return out


def main():
    ap = argparse.ArgumentParser('eval_decouple: AUPRO thích mượt, SegF1 thích sắc -> tách đầu ra')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--sigmas', type=float, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./decouple')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('decouple', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    kernels = {float(s): get_gaussian_kernel(kernel_size=int(6 * s) | 1, sigma=float(s)).to(device)
               for s in args.sigmas}
    p(f'device={device} eff_grid={args.tiles*args.grid_tile} layers={layers} head_w={args.head_w} '
      f'sigmas={args.sigmas} k={args.thr_sigma}')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, kernels, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        s = '  '.join(f'σ{int(sg)}=({r[sg][0]:.3f}/{r[sg][1]:.3f})' for sg in kernels)
        p(f'  [{cat:11s}] {s}')
    if not res:
        return

    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig) theo sigma =====')
    m = {sg: (float(np.mean([res[c][sg][0] for c in res])), float(np.mean([res[c][sg][1] for c in res])))
         for sg in kernels}
    for sg in kernels:
        p(f'  σ={sg:>4}: AUPRO0.05={m[sg][0]:.4f}  SegF1={m[sg][1]:.4f}')

    best_au = max(kernels, key=lambda s: m[s][0])
    best_f1 = max(kernels, key=lambda s: m[s][1])
    p('\n----- DECOUPLE -----')
    p(f'  tiff (AUPRO)  -> σ={best_au}:  AUPRO0.05={m[best_au][0]:.4f}')
    p(f'  png  (SegF1)  -> σ={best_f1}:  SegF1={m[best_f1][1]:.4f}')
    single = min(kernels, key=lambda s: abs(s - 4.0))                     # baseline: cùng σ≈4 cho cả hai
    p(f'  single-map σ={single}: AUPRO0.05={m[single][0]:.4f}  SegF1={m[single][1]:.4f}')
    p(f'  => DECOUPLED cặp = ({m[best_au][0]:.4f} / {m[best_f1][1]:.4f})  '
      f'vs single ({m[single][0]:.4f} / {m[single][1]:.4f})  '
      f'Δ=+{m[best_au][0]-m[single][0]:.4f}/+{m[best_f1][1]-m[single][1]:.4f}')

    p('\nĐỌC: nếu best_sigma_AUPRO > best_sigma_SegF1 (mượt cho tiff, sắc cho png) và cặp decoupled')
    p('  > single trên CẢ HAI => tách đầu ra là both-lift miễn phí (chỉ 2 sigma khác nhau).')
    p('  Build: infer_submit lưu tiff@σ_lớn + png@(σ_nhỏ, test_ksig). Nếu heavy-smooth ~ overlap-AUPRO')
    p('  thì khỏi cần overlap scorer.')


if __name__ == '__main__':
    main()
