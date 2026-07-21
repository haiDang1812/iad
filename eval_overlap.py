# eval_overlap.py
# -----------------------------------------------------------------------------
# SUBSTRATE lever (readout đã cạn): cửa sổ hi-res CHỒNG LẤN + trung bình overlap.
#
# Chẩn đoán turnover (eval_resfuse): res tile-grid TỤT sau đỉnh vì tile KHÔNG chồng lấn ->
#   zoom -> mất ngữ cảnh (fabric/vial AUPRO sập). Overlap-averaging giữ ngữ cảnh ĐỒNG THỜI
#   dày pixel -> phá ngòi nổ turnover -> đẩy res cao hơn mà AUPRO không sập + SegF1 lên.
#   (Đây là cơ chế SuperADD/ISVL đạt SegF1 0.54-0.57.)
#
# So cùng WINDOW SCALE: overlap ov=0 (≈ tile non-overlap hiện tại) vs ov=0.5 (chồng nửa).
#   bank/head build ở scale window (T=win). Test-time: trượt cửa sổ 1/win với stride
#   (1-ov)/win, mỗi cửa sổ encode riêng (giữ NGỮ CẢNH đầy đủ trong cửa sổ), dán vào canvas
#   SMOOTH_RES với sum/count -> trung bình vùng overlap. Đo AUPRO0.05 + SegF1@test_ksig.
#
# Config = "win:gt:ov". PASS = ov=0.5 nâng CẢ HAI so ov=0 cùng win:gt => overlap là lever
#   substrate nhấc cả hai => build vào infer (thay tiling bằng overlapping window).
#
#   python eval_overlap.py --data_path ../data --out_dir ./overlap --configs 3:24:0.0 3:24:0.5
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
from sklearn.metrics import precision_recall_curve
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


def segf1_max(maps, gts):
    """TRẦN của map: F1 tốt nhất trên TOÀN dải ngưỡng. Bỏ biến 'quy tắc cắt' => đo chất lượng
    BẢN ĐỒ. (Đo trên v3_large: trần 0.676 vs đạt 0.616 => threshold chỉ còn +0.06 => map là
    bottleneck. Mọi config mới phải nâng CỘT NÀY, không thì vô nghĩa.)"""
    y = np.concatenate([g.reshape(-1) for g in gts]).astype(np.uint8)
    s = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    if y.max() == 0:
        return float('nan')
    pr, rc, _ = precision_recall_curve(y, s)
    return float(np.nanmax(2 * pr * rc / (pr + rc + 1e-12)))


def metrics(maps, gts, k):
    return aupro05(maps, gts), segf1_ksig(maps, gts, k), segf1_max(maps, gts)


def _resize(a, hw, device):
    t = torch.tensor(a, device=device)[None, None].float()
    return F.interpolate(t, size=hw, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def positions(win, ov):
    side = 1.0 / win
    stride = side * (1.0 - ov)
    ps, x = [], 0.0
    while x <= 1.0 - side + 1e-6:
        ps.append(x); x += stride
    if ps[-1] < 1.0 - side - 1e-6:
        ps.append(1.0 - side)
    return ps, side


@torch.no_grad()
def score_overlap(bb, pil, bank, head, args, layers, device, win, gt, ov, canvas, agg):
    """Trượt cửa sổ 1/win chồng lấn -> (d_canvas, p_canvas) @canvas.
    agg='mean' (trung bình overlap, mượt -> AUPRO) hoặc 'max' (giữ ĐỈNH overlap -> SegF1)."""
    R = gt * bb.patch; C = bank.shape[-1]
    W, H = pil.size
    ps, side = positions(win, ov)
    init = 0.0 if agg == 'mean' else -1e9
    accD = np.full((canvas, canvas), init, np.float32); accP = np.full((canvas, canvas), init, np.float32)
    cnt = np.zeros((canvas, canvas), np.float32)
    for py in ps:
        for px in ps:
            crop = pil.crop((px * W, py * H, (px + side) * W, (py + side) * H)).resize((R, R))
            grid = img_featgrid(bb, crop, 1, R, gt, layers, args.enc_batch)   # 1 cửa sổ = giữ ngữ cảnh
            d = np.asarray(nn_map(grid, bank, device))
            pr = torch.sigmoid(head(grid.reshape(-1, C))).reshape(gt, gt).cpu().numpy() if head is not None \
                else np.zeros((gt, gt), np.float32)
            cx0, cy0 = int(round(px * canvas)), int(round(py * canvas))
            cx1, cy1 = int(round((px + side) * canvas)), int(round((py + side) * canvas))
            dU = _resize(d, (cy1 - cy0, cx1 - cx0), device); pU = _resize(pr, (cy1 - cy0, cx1 - cx0), device)
            if agg == 'mean':
                accD[cy0:cy1, cx0:cx1] += dU; accP[cy0:cy1, cx0:cx1] += pU; cnt[cy0:cy1, cx0:cx1] += 1.0
            else:                                                            # max: giữ đỉnh
                accD[cy0:cy1, cx0:cx1] = np.maximum(accD[cy0:cy1, cx0:cx1], dU)
                accP[cy0:cy1, cx0:cx1] = np.maximum(accP[cy0:cy1, cx0:cx1], pU)
                cnt[cy0:cy1, cx0:cx1] = 1.0
    cnt = np.maximum(cnt, 1.0)
    return accD / cnt, accP / cnt


def run_cat(bb, cat, win, gt, ov, args, layers, gk, device):
    args.tiles, args.grid_tile = win, gt
    R = gt * bb.patch; hw = args.head_w; k = args.thr_sigma
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train and len(tr) > args.max_train:
        tr = tr[:args.max_train]
    bank = build_bank(bb, tr, win, R, gt, layers, args.enc_batch, args.bank_size, device)
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None
    cv = args.canvas
    idx = [i for i in bad if i not in set(bad[:args.shots])][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    gts = [gt_grid(ds.gt_paths[i], ds.labels[i], cv).astype(np.uint8) for i in idx]

    dP, pP = [], []
    for i in tqdm(idx, ncols=70, desc=f'    {cat} {win}:{gt}:{ov}', leave=False):
        d, pr = score_overlap(bb, Image.open(ds.img_paths[i]), bank, head, args, layers, device,
                              win, gt, ov, cv, args.agg)
        dP.append(d); pP.append(pr)
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d in dP]), [1, 99])
    maps = []
    for d, pr in zip(dP, pP):
        fused = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr
        t = gk(torch.tensor(fused, device=device)[None, None].float())[0, 0].cpu().numpy()   # gaussian @canvas
        maps.append(t.astype(np.float32))
    return metrics(maps, gts, k)


def main():
    ap = argparse.ArgumentParser('eval_overlap: cửa sổ chồng lấn có phá turnover + nâng cả hai?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--configs', type=str, nargs='+', default=['3:24:0.0', '3:24:0.5'],
                    help='win:gt:ov (ov=0 ~ tile non-overlap; ov=0.5 chồng nửa)')
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
    ap.add_argument('--canvas', type=int, default=SMOOTH_RES, help='độ phân giải OUTPUT (256 cũ; 512 mịn hơn -> SegF1)')
    ap.add_argument('--agg', type=str, default='mean', choices=['mean', 'max'],
                    help='gộp vùng overlap: mean (mượt->AUPRO) hay max (giữ đỉnh->SegF1)')
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./overlap')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('overlap', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    cfgs = [(int(c.split(':')[0]), int(c.split(':')[1]), float(c.split(':')[2])) for c in args.configs]
    p(f'device={device} configs={args.configs} canvas={args.canvas} agg={args.agg} '
      f'layers={layers} head_w={args.head_w} k={args.thr_sigma}')

    res = {c: {} for c in args.configs}
    for cat in args.categories:
        for cstr, (win, gt, ov) in zip(args.configs, cfgs):
            r = run_cat(bb, cat, win, gt, ov, args, layers, gk, device)
            if r is None:
                continue
            res[cstr][cat] = r
            p(f'  [{cat:11s}] {cstr}: AUPRO0.05={r[0]:.3f}  SegF1={r[1]:.3f}  SegF1_max(trần)={r[2]:.3f}')

    p('\n' + '=' * 86 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig / SegF1_max = TRẦN của map) =====')
    common = set.intersection(*[set(res[c].keys()) for c in args.configs]) if all(res[c] for c in args.configs) else set()
    common = sorted(common)
    base = args.configs[0]
    bm = [float(np.mean([res[base][c][j] for c in common])) if common else float('nan') for j in range(3)]
    for cstr in args.configs:
        if not common:
            p(f'  {cstr}: (không có cat chung)'); continue
        m = [float(np.mean([res[cstr][c][j] for c in common])) for j in range(3)]
        d = '  <- base' if cstr == base else \
            f'   Δ={m[0]-bm[0]:+.4f}/{m[1]-bm[1]:+.4f}/{m[2]-bm[2]:+.4f}'
        p(f'  {cstr:12s}: AUPRO0.05={m[0]:.4f}  SegF1={m[1]:.4f}  TRẦN={m[2]:.4f}{d}  (n={len(common)})')

    p('\nĐỌC — cột quyết định là TRẦN (SegF1_max), không phải SegF1:')
    p('  - TRẦN lên => cửa sổ chồng lấn làm BẢN ĐỒ tốt hơn thật (đây là bottleneck đã đo được:')
    p('    v3_large trần 0.676/đạt 0.616; v3_huge trần 0.694/đạt 0.648 => threshold chỉ còn +0.05).')
    p('  - TRẦN đứng yên mà SegF1 nhúc nhích => chỉ là nhiễu quy tắc cắt, KHÔNG build.')
    p('  - Nhìn riêng sheet_metal: trần 0.30 vs SuperADD 0.557. Nếu overlap không nâng trần cat này')
    p('    thì vấn đề không phải overlap mà là độ phân giải thô (patch 16 nuốt ~34px ảnh gốc).')


if __name__ == '__main__':
    main()
