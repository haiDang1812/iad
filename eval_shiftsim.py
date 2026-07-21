# eval_shiftsim.py
# -----------------------------------------------------------------------------
# VALIDATE offline "cái gì nhấc CẢ HAI metric". private không có GT -> mô phỏng
#   light-shift trên test_public (CÓ GT): train bank+head trên train/shots SẠCH (nguồn),
#   rồi chấm AUPRO0.05 + SegF1 trên public ĐÃ SHIFT. So các điều kiện:
#
#   clean/none  : không shift, head gốc          -> TRẦN (không có domain-gap)
#   shift/none  : có shift, head gốc             -> baseline suy giảm (hành vi hiện tại)
#   shift/image : có shift, re-center head TỪNG ẢNH (trừ mean của chính ảnh)
#   shift/split : có shift, re-center head POOLED (trừ mean gộp toàn eval)
#
# GIẢ THUYẾT (diag25/26): shift = offset δ ~uniform trong feature-space, cắn nhánh HEAD
#   (drift +2.79σ) chứ không cắn distance (shift-gap~1.0). Re-center mu head -> đích khử δ,
#   GIỮ tín hiệu -> shift/image|split hồi phục về gần clean/none trên CẢ HAI metric.
#
# PASS = shift/image (hoặc split) >> shift/none trên CẢ AUPRO LẪN SegF1, tiến gần clean/none.
# FAIL = không hồi phục -> δ không uniform / không tách được -> đổi hướng (per-image residual...).
#
#   python eval_shiftsim.py --data_path ../data --out_dir ./shiftsim --bright 0.6 --jitter 0.2
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
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, score_grid, gt_grid, up_to, recenter_c, apply_recenter,
    VALID, IMG_EXT, SMOOTH_RES,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def shift_image(pil, bright, contrast, gamma, jitter, rng):
    """Mô phỏng private light-shift: brightness (hệ thống + jitter mỗi ảnh) + contrast + gamma."""
    b = bright * (1.0 + rng.uniform(-jitter, jitter))
    im = pil.convert('RGB')
    if abs(b - 1.0) > 1e-6:
        im = ImageEnhance.Brightness(im).enhance(b)
    if abs(contrast - 1.0) > 1e-6:
        im = ImageEnhance.Contrast(im).enhance(contrast)
    if abs(gamma - 1.0) > 1e-6:
        a = np.clip((np.asarray(im).astype(np.float32) / 255.0) ** gamma, 0, 1)
        im = Image.fromarray((a * 255).astype(np.uint8))
    return im


def aupro05(maps, gts):
    sp = np.array([float(m.max()) for m in maps])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(maps), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def pooled_f1(maps, gts, thr):
    TP = FP = FN = 0.0
    for m, g in zip(maps, gts):
        pred = m >= thr; gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
    return 2 * TP / (2 * TP + FP + FN + 1e-9)


def segf1_ksig(maps, gts, k):
    P = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    return pooled_f1(maps, gts, float(P.mean() + k * P.std()))       # test_ksig (diag27/28)


def run_cat(bb, cat, args, layers, gk, device):
    T, gt = args.tiles, args.grid_tile
    hw = args.head_w
    rng_np = np.random.default_rng(args.seed)
    rng_sh = np.random.default_rng(args.seed + 1)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr, T, gt * bb.patch, gt, layers, args.enc_batch, args.bank_size, device)

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng_np.shuffle(bad)
    shot_pool = bad[:args.shots]
    head = build_head(bb, ds, shot_pool, bank, args, layers, device)     # head SẠCH (nguồn)
    if head is None:
        return None

    shots = set(shot_pool)
    idx = [i for i in bad if i not in shots][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]

    out = {}
    for cond in ['clean', 'shift']:
        recs = []          # (d, pr_raw, fmean, gt)
        for i in tqdm(idx, ncols=70, desc=f'    {cat}/{cond}', leave=False):
            pil = Image.open(ds.img_paths[i])
            if cond == 'shift':
                pil = shift_image(pil, args.bright, args.contrast, args.gamma, args.jitter, rng_sh)
            d, pr, fmean, _ = score_grid(bb, pil, bank, head, args, layers, device, return_feat=True)
            recs.append((d, pr, fmean, gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8)))

        lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _, _, _ in recs]), [1, 99])
        gts = [g for _, _, _, g in recs]

        # pooled feat_mean của điều kiện này (cho re-center 'split')
        fmean_pool = sum(fm for _, _, fm, _ in recs) / len(recs)
        c_split = recenter_c(head, fmean_pool)

        variants = ['none'] if cond == 'clean' else ['none', 'image', 'split']
        for v in variants:
            maps = []
            for d, pr, fm, _ in recs:
                if v == 'image':
                    pr2 = apply_recenter(pr, recenter_c(head, fm))
                elif v == 'split':
                    pr2 = apply_recenter(pr, c_split)
                else:
                    pr2 = pr
                fused = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr2
                maps.append(up_to(fused, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32))
            out[f'{cond}/{v}'] = (aupro05(maps, gts), segf1_ksig(maps, gts, args.thr_sigma))
    return out


def main():
    ap = argparse.ArgumentParser('eval_shiftsim: re-center head có nhấc CẢ HAI metric dưới shift?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
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
    ap.add_argument('--thr_sigma', type=float, default=4.5, help='k của test_ksig (diag27/28)')
    # tham số shift quang học
    ap.add_argument('--bright', type=float, default=0.6, help='hệ số brightness hệ thống (private tối hơn)')
    ap.add_argument('--contrast', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--jitter', type=float, default=0.2, help='dao động brightness ±jitter mỗi ảnh')
    ap.add_argument('--max_eval', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./shiftsim')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('shiftsim', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} layers={layers} head_w={args.head_w}')
    p(f'shift: bright={args.bright} contrast={args.contrast} gamma={args.gamma} jitter=±{args.jitter} | '
      f'SegF1@test_ksig k={args.thr_sigma}')

    conds = ['clean/none', 'shift/none', 'shift/image', 'shift/split']
    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        s = '  '.join(f'{c.split("/")[1]:>5}=({r[c][0]:.3f}/{r[c][1]:.3f})' for c in conds)
        p(f'  [{cat:11s}] {s}')

    if not res:
        return
    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1) qua category =====')
    ref = None
    for c in conds:
        au = float(np.mean([res[k][c][0] for k in res]))
        f1 = float(np.mean([res[k][c][1] for k in res]))
        tag = ''
        if c == 'clean/none':
            ref = (au, f1); tag = '   <- TRẦN (no shift)'
        elif c == 'shift/none':
            tag = '   <- baseline suy giảm'
        else:
            tag = f'   ΔAUPRO={au-res_sn[0]:+.3f} ΔSegF1={f1-res_sn[1]:+.3f} vs shift/none'
        p(f'  {c:12s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}{tag}')
        if c == 'shift/none':
            res_sn = (au, f1)
    if ref is not None:
        p(f'\n  gap do shift (clean/none - shift/none): '
          f'AUPRO={ref[0]-res_sn[0]:+.3f}  SegF1={ref[1]-res_sn[1]:+.3f}  <- phần re-center cần hồi phục')

    p('\nĐỌC:')
    p('  PASS: shift/image (hoặc split) nhấc CẢ AUPRO LẪN SegF1 so shift/none, tiến gần clean/none')
    p('    => re-center head khử được drift -> đây là lever "nhấc cả hai". Port sang private:')
    p('       infer_submit --head_recenter image --thr_mode test_ksig --thr_sigma 4.5.')
    p('  FAIL: không hồi phục => δ không tách được bằng dịch mean => cần bậc cao hơn (rescale sd,')
    p('    hoặc chiếu bỏ subspace shift trên input head). eval này tái dùng để thử tiếp.')


if __name__ == '__main__':
    main()
