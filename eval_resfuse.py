# eval_resfuse.py
# -----------------------------------------------------------------------------
# "Nhấc CẢ HAI metric" đúng nghĩa = cải thiện MAP, không phải khử shift (eval_shiftsim
#   đã cho thấy AUPRO bất biến với shift -> AUPRO ở TRẦN MAP, không phải trần shift).
#   Đòn bẩy MAP có bằng chứng dương duy nhất = ĐỘ PHÂN GIẢI (sweep cũ +0.045 AUPRO @2:36,
#   nhưng đo trên nhánh distance-only + SegF1 buggy). Ở đây đo LẠI cho ĐÚNG:
#
#   - Pipeline FUSED đầy đủ (distance + head few-shot + fuse), y hệt submit.
#   - AUPRO0.05 (threshold-free) VÀ SegF1@test_ksig (k=4.5, ngưỡng đã sửa ở diag27/28).
#   - Kèm cột split-recenter (add-on SegF1 miễn phí, AUPRO trung tính - eval_shiftsim).
#   - Trên test_public SẠCH (AUPRO bất biến shift nên clean là proxy công bằng; SegF1@test_ksig
#     chính là đại lượng diag27/28 tối ưu, 2:28 -> 0.558).
#
# Config nào nâng CẢ AUPRO0.05 LẪN SegF1 so 2:28 -> lượt submit đáng đốt
#   (res + test_ksig + split-recenter gộp 1 phát). Nếu res cao mà 2 số bão hòa/giảm
#   -> trần là feature, không phải res -> quay lại backbone/layer (diag12/13).
#
#   python eval_resfuse.py --data_path ../data --out_dir ./resfuse --configs 2:28 2:36 3:24
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
    build_bank, build_head, score_grid, gt_grid, up_to, recenter_c, apply_recenter,
    VALID, IMG_EXT, SMOOTH_RES,
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
    thr = float(P.mean() + k * P.std())                        # test_ksig
    TP = FP = FN = 0.0
    for m, g in zip(maps, gts):
        pred = m >= thr; gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
    return 2 * TP / (2 * TP + FP + FN + 1e-9)


def run_cat(bb, cat, T, gt, args, layers, gk, device):
    args.tiles, args.grid_tile = T, gt                          # build_head/score_grid đọc từ args
    R = gt * bb.patch
    hw = args.head_w
    rng_np = np.random.default_rng(args.seed)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train and len(tr) > args.max_train:      # cap ảnh dựng bank -> nhanh hơn, bank vẫn đủ lấp bank_size
        tr = tr[:args.max_train]
    bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng_np.shuffle(bad)
    shot_pool = bad[:args.shots]
    head = build_head(bb, ds, shot_pool, bank, args, layers, device)
    if head is None:
        return None

    shots = set(shot_pool)
    idx = [i for i in bad if i not in shots][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]

    recs = []                                                   # (d, pr, fmean, gt)
    for i in tqdm(idx, ncols=70, desc=f'    {cat} {T}:{gt}', leave=False):
        d, pr, fmean, _ = score_grid(bb, Image.open(ds.img_paths[i]), bank, head, args, layers, device, return_feat=True)
        recs.append((d, pr, fmean, gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8)))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _, _, _ in recs]), [1, 99])
    gts = [g for _, _, _, g in recs]
    c_split = recenter_c(head, sum(fm for _, _, fm, _ in recs) / len(recs))

    out = {}
    for v in ['none', 'split']:
        maps = []
        for d, pr, _, _ in recs:
            pr2 = apply_recenter(pr, c_split) if v == 'split' else pr
            fused = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr2
            maps.append(up_to(fused, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32))
        out[v] = (aupro05(maps, gts), segf1_ksig(maps, gts, args.thr_sigma))
    return out


def main():
    ap = argparse.ArgumentParser('eval_resfuse: sweep resolution, đo AUPRO0.05 + SegF1@test_ksig (fused)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--configs', type=str, nargs='+', default=['2:28', '2:36', '3:24'],
                    help='tiles:grid_tile ; production = 2:28')
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
    ap.add_argument('--thr_sigma', type=float, default=4.5, help='k của test_ksig')
    ap.add_argument('--max_train', type=int, default=80, help='cap ảnh train dựng bank (0=hết). Bank vẫn subsample về bank_size nên đủ')
    ap.add_argument('--max_eval', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./resfuse')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('resfuse', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    cfgs = [(int(c.split(':')[0]), int(c.split(':')[1])) for c in args.configs]

    p(f'device={device} model={args.model} configs={args.configs} (eff-grid={[t*g for t, g in cfgs]}) '
      f'layers={layers} head_w={args.head_w} SegF1@test_ksig k={args.thr_sigma}')

    # res[cstr][cat] = {'none':(au,f1),'split':(au,f1)}
    res = {c: {} for c in args.configs}
    for cat in args.categories:
        for cstr, (T, gt) in zip(args.configs, cfgs):
            r = run_cat(bb, cat, T, gt, args, layers, gk, device)
            if r is None:
                continue
            res[cstr][cat] = r
            p(f'  [{cat:11s}] {cstr}: none=({r["none"][0]:.3f}/{r["none"][1]:.3f})  '
              f'split=({r["split"][0]:.3f}/{r["split"][1]:.3f})')

    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig) qua category =====')
    # CÔNG BẰNG: chỉ trung bình trên cat có mặt ở MỌI config (cat rớt ở 1 config -> loại khắp nơi,
    # nếu không 2:28 gồm can (kém) mà 2:36/3:24 không -> Δ bị thổi).
    common = set.intersection(*[set(res[c].keys()) for c in args.configs]) if all(res[c] for c in args.configs) else set()
    dropped = sorted(set().union(*[set(res[c].keys()) for c in args.configs]) - common)
    if dropped:
        p(f'  (LOẠI khỏi MEAN vì thiếu ở ≥1 config: {dropped} — build_head None ở res cao)')
    common = sorted(common)
    base = args.configs[0]
    b_au = float(np.mean([res[base][c]['none'][0] for c in common])) if common else float('nan')
    b_f1 = float(np.mean([res[base][c]['none'][1] for c in common])) if common else float('nan')
    for cstr in args.configs:
        if not common:
            p(f'  {cstr}: (không có cat chung)'); continue
        for v in ['none', 'split']:
            au = float(np.mean([res[cstr][c][v][0] for c in common]))
            f1 = float(np.mean([res[cstr][c][v][1] for c in common]))
            tag = '  <- production' if (cstr == base and v == 'none') else \
                  f'   ΔAUPRO={au-b_au:+.4f}  ΔSegF1={f1-b_f1:+.4f}  vs 2:28/none'
            p(f'  {cstr:>6}/{v:5s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}{tag}  (n={len(common)} cat)')

    p('\nĐỌC:')
    p(' - Config nâng CẢ AUPRO0.05 LẪN SegF1 so 2:28/none => lever "nhấc cả hai" thật.')
    p('   Submit gộp: infer_submit --tiles T --grid_tile gt --thr_mode test_ksig --thr_sigma 4.5')
    p('              [--head_recenter split nếu cột split thắng none].')
    p(' - split thắng none về SegF1 mà AUPRO ~ngang => bật --head_recenter split (add-on miễn phí).')
    p(' - Res cao mà 2 số bão hòa/giảm => trần là FEATURE, không phải res => diag12/13 (backbone/layer).')


if __name__ == '__main__':
    main()
