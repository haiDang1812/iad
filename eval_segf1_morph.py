# eval_segf1_morph.py
# -----------------------------------------------------------------------------
# ĐÒN BẨY RẺ CHO SegF1: hậu xử lý MORPHOLOGY trên binary mask sau threshold.
#   Khác gaussian (diag16 nói gaussian HẠI SegF1 vì nhoè biên): opening bỏ đốm FP
#   LẺ (isolated speckle) mà KHÔNG đụng biên vùng defect thật -> tăng precision ->
#   tăng SegF1 nếu FP hiện dạng đốm rải. closing lấp lỗ -> có thể tăng recall.
#
# Chạy Y HỆT pipeline production (eval_nrs_head): cùng bank/shots/head/fuse/threshold
#   ksig. Chỉ THÊM 1 bước morphology trên binary trước khi tính TP/FP/FN toàn cục.
#   SegF1 = 2TP/(2TP+FP+FN), pool pixel qua toàn test — đúng định nghĩa server.
#   Đo trên CẢ head grid (production dùng cho png/SegF1) VÀ head nrs.
#
#   variant baseline (none) PHẢI trùng F1@ksig của eval_nrs_head -> sanity check.
#
# ĐỌC (pre-register): morph tốt nhất >= baseline +0.01 SegF1 trên >=2/3 cat
#   -> đòn bẩy SỐNG, fold vào infer_submit_uniform (1 dòng, disclose là hậu xử lý
#   chuẩn). < +0.01 hoặc thất thường -> BỎ, không bịa novelty từ nó.
#
#   python eval_segf1_morph.py --data_path ../data --out_dir ./morph \
#       --categories can wallplugs sheet_metal
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
from scipy.ndimage import binary_opening, binary_closing

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, VALID, IMG_EXT,
)
from eval_nrs_head import build_nrs_head                               # noqa: E402
from eval_native import Hist                                           # noqa: E402
from dataset import MVTecAD2Dataset                                    # noqa: E402
from utils import get_gaussian_kernel, get_logger                      # noqa: E402
from backbones_ext import load_backbone                                # noqa: E402

warnings.filterwarnings('ignore')


def morph_variants(r_open, r_close):
    """Danh sách (tên, hàm) áp lên binary 2D. baseline = identity."""
    v = [('none', lambda b: b)]
    for r in r_open:
        v.append((f'open{r}', lambda b, r=r: binary_opening(b, iterations=r)))
    for r in r_close:
        v.append((f'close{r}', lambda b, r=r: binary_closing(b, iterations=r)))
    for ro in r_open:
        for rc in r_close:
            v.append((f'open{ro}+close{rc}',
                      lambda b, ro=ro, rc=rc: binary_closing(binary_opening(b, iterations=ro), iterations=rc)))
    return v


def seg_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def run_cat(bb, cat, args, layers, gk, device, p, variants):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không có train/good -> bỏ')
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    p(f'  [{cat}] build bank từ {len(tr)} ảnh...')
    bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shots = bad[:args.shots]
    torch.manual_seed(args.seed)
    headA = build_head(bb, ds, shots, bank, args, layers, device)          # grid head (production png)
    headB = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)   # nrs head
    if headB is None:
        p(f'  [{cat}] NRS None -> bỏ')
        return None

    idx_bad = bad[args.shots:]
    idx_good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    if args.max_eval:
        idx_bad = idx_bad[:args.max_eval]
        idx_good = idx_good[:args.max_eval]
    idx = idx_bad + idx_good
    p(f'    eval: bad={len(idx_bad)} good={len(idx_good)}')

    # PASS 1: encode 1 lần, giữ grid nhỏ (G×G) + size native -> pass 2 chỉ interpolate
    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]
            d = np.asarray(nn_map(g, bank, device))
            flat = g.reshape(-1, C)
            prA = torch.sigmoid(headA(flat)).reshape(G, G).cpu().numpy() if headA is not None else None
            prB = torch.sigmoid(headB(flat)).reshape(G, G).cpu().numpy()
            grids.append((d, prA, prB))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _, _ in grids]), [1, 99])

    def fuse(d, pr):
        dr = (d - lo) / (hi - lo + 1e-8)
        return dr.astype(np.float32) if pr is None else ((1 - hw) * dr + hw * pr).astype(np.float32)

    out = {}
    for name, k in [('grid', 1), ('nrs', 2)]:
        s_grids = [fuse(gg[0], gg[k]) for gg in grids]
        gmin = min(float(s.min()) for s in s_grids)
        gmax = max(float(s.max()) for s in s_grids)
        # threshold ksig y hệt production (pool mọi pixel native qua Hist)
        h = Hist(gmin - 0.05, gmax + 0.05)
        natmaps, natgts = [], []
        for s, (H, W), i in zip(s_grids, sizes, idx):
            with torch.no_grad():
                t = torch.tensor(s, device=device)[None, None].float()
                t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
                t = gk(t)
                mnat = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            if ds.labels[i] == 0:
                gnat = np.zeros((H, W), np.uint8)
            else:
                gnat = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)
            h.add(mnat.reshape(-1), gnat.reshape(-1))
            natmaps.append(mnat)
            natgts.append(gnat)
        thr = h.ksig(args.thr_sigma)

        # mỗi variant: pool TP/FP/FN toàn cục ở CÙNG threshold, chỉ khác morphology
        acc = {vn: [0, 0, 0] for vn, _ in variants}
        for mnat, gnat in zip(natmaps, natgts):
            base = mnat > thr
            gt = gnat.astype(bool)
            for vn, fn_ in variants:
                b = fn_(base)
                tp = int(np.logical_and(b, gt).sum())
                fp = int(np.logical_and(b, ~gt).sum())
                fnn = int(np.logical_and(~b, gt).sum())
                acc[vn][0] += tp
                acc[vn][1] += fp
                acc[vn][2] += fnn
        f1s = {vn: seg_f1(*acc[vn]) for vn, _ in variants}
        out[name] = f1s
        base_f1 = f1s['none']
        best_vn = max(f1s, key=f1s.get)
        p(f'    [{cat}] head={name:4s} SegF1 baseline={base_f1:.4f} | best={best_vn}={f1s[best_vn]:.4f} '
          f'(Δ{f1s[best_vn] - base_f1:+.4f})')
        p('      ' + '  '.join(f'{vn}={f1s[vn]:.4f}' for vn, _ in variants))
    return out


def main():
    ap = argparse.ArgumentParser('SegF1 morphology post-proc premise')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--n_neg_img', type=int, default=8000)
    ap.add_argument('--pos_per_region', type=int, default=1500)
    ap.add_argument('--max_pos', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split')
    ap.add_argument('--r_open', type=int, nargs='+', default=[1, 2], help='bán kính (iterations) opening')
    ap.add_argument('--r_close', type=int, nargs='+', default=[1], help='bán kính (iterations) closing')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=None, help='mặc định = cả 8 cat AD2')
    ap.add_argument('--out_dir', type=str, default='./morph')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('morph', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    variants = morph_variants(args.r_open, args.r_close)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} k={args.thr_sigma} '
      f'| variants: {[v for v, _ in variants]}')

    cats = args.categories or list(VALID)
    res = {}
    for cat in cats:
        r = run_cat(bb, cat, args, layers, gk, device, p, variants)
        if r is not None:
            res[cat] = r
    if not res:
        p('không cat nào chạy được'); return

    p('\n' + '=' * 92 + '\n===== TỔNG SegF1 (pool pixel toàn cục, threshold ksig) =====')
    for head in ['grid', 'nrs']:
        p(f'\n  --- head={head} ---')
        vnames = [v for v, _ in variants]
        means = {vn: float(np.mean([res[c][head][vn] for c in res])) for vn in vnames}
        base = means['none']
        for vn in vnames:
            mark = '  <== best' if vn == max(means, key=means.get) and vn != 'none' else ''
            p(f'    {vn:14s} MEAN SegF1={means[vn]:.4f}  Δ={means[vn] - base:+.4f}'
              f'  | thắng baseline {sum(1 for c in res if res[c][head][vn] > res[c][head]["none"])}/{len(res)} cat{mark}')

    p('\nĐỌC (pre-registered):')
    p('  - best morph >= baseline +0.01 SegF1 & thắng >=2/3 cat (head grid) -> SỐNG, fold vào submit.')
    p('  - < +0.01 hoặc thất thường                                        -> BỎ, đừng bịa novelty.')


if __name__ == '__main__':
    main()
