# eval_multiscale.py
# -----------------------------------------------------------------------------
# NOVELTY PREMISE: "self-calibrated multi-scale PEAK fusion" phá đánh đổi fine-vs-context.
#
# Chẩn đoán (eval_resfuse): res cao giúp defect-nhỏ (rice/sheet_metal SegF1↑) nhưng hại
#   cần-ngữ-cảnh (fabric/vial AUPRO↓); đường cong PEAK rồi TỤT. Không res đơn nào tối ưu,
#   và tension này đúng CẢ TRONG từng ảnh (defect nhỏ chỗ này, lệch-texture chỗ kia).
#
# Ý tưởng: mỗi scale s chuẩn hóa map về z SO VỚI NORMAL CỦA CHÍNH SCALE (z_s=(M_s-μ_s)/σ_s),
#   rồi hợp nhất PER-PIXEL bằng PEAK (max-z) thay vì TRUNG BÌNH. Mỗi pixel định tuyến tới
#   scale mà nó là outlier mạnh nhất -> giữ đỉnh sắc (AUPRO0.05 low-FPR) + giữ defect nhỏ
#   (SegF1) ĐỒNG THỜI. Khác SuperADD (mean-overlap làm TÙ đỉnh).
#
# Kiểm PREMISE (chưa build vào infer): trên test_public (CÓ GT), so — cho CẢ 2 metric —
#   single*  : scale đơn tốt nhất (trần hiện tại)
#   mean_z   : trung bình z qua scale (đại diện SuperADD mean-fuse)
#   max_z    : PEAK fusion (method đề xuất, unsupervised)
#   soft_z   : softmax-weighted z (bản mềm của max)
#   ORACLE   : định tuyến per-pixel hoàn hảo bằng GT (defect->max_s, normal->min_s) = TRẦN
#
# PASS = max_z/soft_z > single* VÀ > mean_z trên CẢ HAI metric, và ORACLE >> single* (còn đất).
# FAIL = ORACLE ~ single* (scale không bổ khuyết per-pixel) HOẶC max_z !> mean_z (peak vô ích)
#        -> bỏ ý multi-scale, quay lại (A) uniform 3:24.
#
#   python eval_multiscale.py --data_path ../data --out_dir ./multiscale --scales 2:28 3:24 4:24
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, score_grid, gt_grid, up_to, VALID, IMG_EXT, SMOOTH_RES,
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


def metrics(maps, gts, k):
    return aupro05(maps, gts), segf1_ksig(maps, gts, k)


def scale_maps(bb, cat, T, gt, args, layers, gk, device, idx, ds):
    """Trả list per-image fused-map @SMOOTH_RES ở một scale (T:gt). None nếu head fail."""
    args.tiles, args.grid_tile = T, gt
    R = gt * bb.patch; hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train and len(tr) > args.max_train:
        tr = tr[:args.max_train]
    bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None
    recs = []
    for i in idx:
        d, pr = score_grid(bb, Image.open(ds.img_paths[i]), bank, head, args, layers, device)
        recs.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in recs]), [1, 99])
    out = []
    for d, pr in recs:
        fused = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + (hw * pr if pr is not None else 0)
        out.append(up_to(fused, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32))
    return out


def zstack(maps_by_scale):
    """[S] list per-image maps -> per scale z-chuẩn hóa (μ,σ pooled của scale đó, transductive)."""
    S = len(maps_by_scale); N = len(maps_by_scale[0])
    Z = []                                                     # Z[s] = list per-image z-map
    for s in range(S):
        P = np.concatenate([m.reshape(-1) for m in maps_by_scale[s]])
        mu, sd = float(P.mean()), float(P.std() + 1e-8)
        Z.append([(m - mu) / sd for m in maps_by_scale[s]])
    return Z, S, N


def run_cat(bb, cat, args, layers, gk, device):
    rng = np.random.default_rng(args.seed)
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shots = set(bad[:args.shots])
    idx = [i for i in bad if i not in shots][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    gts = [gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8) for i in idx]

    cfgs = [(int(c.split(':')[0]), int(c.split(':')[1])) for c in args.scales]
    maps_by_scale = []
    kept_scales = []
    for cstr, (T, gt) in zip(args.scales, cfgs):
        m = scale_maps(bb, cat, T, gt, args, layers, gk, device, idx, ds)
        if m is not None:
            maps_by_scale.append(m); kept_scales.append(cstr)
    if len(maps_by_scale) < 2:
        return None                                           # cần ≥2 scale để fuse

    k = args.thr_sigma
    Z, S, N = zstack(maps_by_scale)

    out = {'scales': kept_scales}
    # single* : scale đơn tốt nhất (theo tổng 2 metric chuẩn hóa thô)
    singles = [metrics(maps_by_scale[s], gts, k) for s in range(S)]
    out['single'] = max(singles, key=lambda ab: ab[0] + ab[1])

    # mean_z : trung bình z (đại diện mean-overlap SuperADD)
    mean_maps = [np.mean([Z[s][n] for s in range(S)], axis=0) for n in range(N)]
    out['mean_z'] = metrics(mean_maps, gts, k)

    # max_z : PEAK fusion (method)
    max_maps = [np.max(np.stack([Z[s][n] for s in range(S)]), axis=0) for n in range(N)]
    out['max_z'] = metrics(max_maps, gts, k)

    # soft_z : softmax-weighted z (bản mềm)
    T_soft = args.soft_temp
    soft_maps = []
    for n in range(N):
        st = np.stack([Z[s][n] for s in range(S)])            # [S,H,W]
        w = np.exp(st / T_soft); w /= w.sum(0, keepdims=True)
        soft_maps.append((w * st).sum(0))
    out['soft_z'] = metrics(soft_maps, gts, k)

    # ORACLE per-pixel: defect->max_s z, normal->min_s z (TRẦN định tuyến hoàn hảo)
    orc_maps = []
    for n in range(N):
        st = np.stack([Z[s][n] for s in range(S)])
        gb = gts[n].astype(bool)
        orc = np.where(gb, st.max(0), st.min(0))
        orc_maps.append(orc)
    out['oracle'] = metrics(orc_maps, gts, k)
    return out


def main():
    ap = argparse.ArgumentParser('eval_multiscale: peak-fusion đa-scale có phá đánh đổi fine-vs-context?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--scales', type=str, nargs='+', default=['2:28', '3:24', '4:24'],
                    help='các scale tiles:grid_tile để fuse')
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
    ap.add_argument('--soft_temp', type=float, default=1.0, help='nhiệt softmax cho soft_z')
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./multiscale')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('multiscale', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} scales={args.scales} layers={layers} head_w={args.head_w} '
      f'SegF1@test_ksig k={args.thr_sigma}')

    variants = ['single', 'mean_z', 'max_z', 'soft_z', 'oracle']
    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ (thiếu scale/head)'); continue
        res[cat] = r
        s = '  '.join(f'{v}=({r[v][0]:.3f}/{r[v][1]:.3f})' for v in variants)
        p(f'  [{cat:11s}] scales={r["scales"]}  {s}')
    if not res:
        return

    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig) qua category =====')
    m = {}
    for v in variants:
        au = float(np.mean([res[c][v][0] for c in res]))
        f1 = float(np.mean([res[c][v][1] for c in res]))
        m[v] = (au, f1)
    for v in variants:
        d = '' if v == 'single' else (f'   ΔAUPRO={m[v][0]-m["single"][0]:+.4f}  '
                                      f'ΔSegF1={m[v][1]-m["single"][1]:+.4f}  vs single*')
        tag = {'single': '  <- scale đơn tốt nhất', 'mean_z': '  (SuperADD-style)',
               'max_z': '  <- METHOD (peak)', 'oracle': '  <- TRẦN (GT-routing)'}.get(v, '')
        p(f'  {v:7s}: AUPRO0.05={m[v][0]:.4f}  SegF1={m[v][1]:.4f}{d}{tag}')

    p('\nĐỌC:')
    p(' - max_z/soft_z > single* VÀ > mean_z trên CẢ HAI => peak-fusion phá được đánh đổi')
    p('   fine-vs-context => NOVELTY nhấc cả hai. Build vào infer (multi-scale + max-z fuse).')
    p(' - oracle >> single* => scale bổ khuyết per-pixel, còn nhiều đất (khoảng cách = trần).')
    p(' - Nếu max_z ~ mean_z ~ single* HOẶC oracle ~ single* => scale KHÔNG bổ khuyết => bỏ,')
    p('   quay lại (A) uniform 3:24.')


if __name__ == '__main__':
    main()
