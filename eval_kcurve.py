# -*- coding: utf-8 -*-
# =============================================================================
# eval_kcurve.py — CON DAO CẮT hỏng ở đâu, và một hằng số k per-cat có đủ không?
#
# BỐI CẢNH (eval_native, 7 cat public, ViT-L, native GT):
#   rice        đạt 0.550 / trần 0.747  -> HỤT 0.197  (map thấy, dao cắt vứt đi)
#   walnuts     đạt 0.754 / trần 0.754  -> hụt 0.000  (đã kịch trần)
#   sheet_metal đạt 0.253 / trần 0.258  -> hụt 0.004  (map MÙ, dao cắt vô can)
#   MEAN(7)     đạt 0.635 / trần 0.683  -> hụt 0.048, và rice chiếm gần hết
#
# Pipeline đang ép MỘT hằng số k=4.5 cho CẢ 8 cat (mean + k*sigma trên pool test).
# Script này hỏi đúng 3 câu, cho từng cat:
#   (1) F1 biến thiên ra sao theo k? -> k* = argmax, F1(k*) so với F1(4.5)
#   (2) NGƯỠNG ORACLE ứng với k bằng bao nhiêu (k_orc)? Nếu F1(k*) ~ trần thì
#       dạng quy tắc "mu + k*sigma" là ĐÚNG, chỉ có hằng số sai -> chỉnh k per-cat là XONG.
#       Nếu F1(k*) << trần thì mu+k*sigma là SAI DẠNG (đuôi phân bố không gaussian)
#       -> phải đổi sang quy tắc theo phân vị/rate, không phải chỉnh hằng số.
#   (3) rate = tỉ lệ pixel bị gán defect. So rate@4.5 với rate@oracle và với gt_rate thật
#       -> biết ta đang OVER-detect hay UNDER-detect, và lệch bao nhiêu LẦN.
#
# Chạy ở NATIVE (GT gốc, canvas 256 = đúng pipeline submit).
# =============================================================================
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
while _D and _D not in sys.path:
    sys.path.insert(0, _D)
    if os.path.exists(os.path.join(_D, 'dataset.py')):
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                    # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, VALID, IMG_EXT,
)
from backbones_ext import load_backbone                                  # noqa: E402
from eval_native import Hist, make_map                                   # noqa: E402
from dataset import MVTecAD2Dataset                                      # noqa: E402
from utils import get_gaussian_kernel, get_logger                        # noqa: E402

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')


def stats(h):
    """mu, sigma từ moment streaming."""
    mu = h.s1 / max(h.n, 1)
    var = max(h.s2 / max(h.n, 1) - mu * mu, 0.0)
    return mu, float(np.sqrt(var))


def f1_curve(h):
    """F1 tại MỌI bin (vectorized) + rate (tỉ lệ pixel bị gán defect) tại mọi bin."""
    tp = np.cumsum(h.pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(h.neg[::-1])[::-1].astype(np.float64)
    fn = float(h.pos.sum()) - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    rate = (tp + fp) / max(h.n, 1)
    return f1, rate


def bin_to_thr(h, b):
    return h.lo + (b + 0.5) / len(h.pos) * (h.hi - h.lo)


def thr_to_bin(h, thr):
    return int(np.clip((thr - h.lo) / (h.hi - h.lo + 1e-12) * len(h.pos), 0, len(h.pos) - 1))


def run_cat(bb, cat, args, layers, gk, device):
    """Trả về Hist ở NATIVE + gt_rate thật. Tái dùng nguyên xi pipeline submit."""
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr[:args.max_train] if args.max_train else tr,
                      T, R, gt, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None

    idx = [i for i in bad if i not in set(bad[:args.shots])][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]

    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt, layers, args.enc_batch)
            d = np.asarray(nn_map(g, bank, device))
            pr = torch.sigmoid(head(g.reshape(-1, C))).reshape(g.shape[0], g.shape[0]).cpu().numpy()
            grids.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])
    s_grids = [((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32) for d, pr in grids]

    gmin = min(float(s.min()) for s in s_grids)
    gmax = max(float(s.max()) for s in s_grids)
    h = Hist(gmin - 0.05, gmax + 0.05)
    npos = 0
    for (s, (H, W), i) in zip(s_grids, sizes, idx):
        m = make_map(s, 256, gk, (H, W), device).cpu().numpy()
        if ds.labels[i] == 0:
            g = np.zeros((H, W), np.uint8)
        else:
            g = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)
        h.add(m.reshape(-1), g.reshape(-1))
        npos += int(g.sum())
    return h, npos / max(h.n, 1)


def main():
    ap = argparse.ArgumentParser('eval_kcurve: chẩn đoán con dao cắt per-category')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--ks', type=float, nargs='+',
                    default=[2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0])
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
    ap.add_argument('--max_eval', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./kcurve')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('kcurve', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4.0).to(device)
    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} '
      f'| NATIVE (GT gốc), canvas=256, k hiện tại={args.thr_sigma}')

    rows = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ')
            continue
        h, gt_rate = r
        f1, rate = f1_curve(h)
        mu, sd = stats(h)
        b_orc = int(np.argmax(f1))
        thr_orc = bin_to_thr(h, b_orc)
        k_orc = (thr_orc - mu) / (sd + 1e-12)

        cur = h.f1_at(h.ksig(args.thr_sigma))
        rate_cur = float(rate[thr_to_bin(h, h.ksig(args.thr_sigma))])
        f1_ks = [(k, h.f1_at(h.ksig(k))) for k in args.ks]
        k_star, f1_star = max(f1_ks, key=lambda t: t[1])
        rows[cat] = dict(cur=cur, k_star=k_star, f1_star=f1_star, ceil=float(f1[b_orc]),
                         k_orc=k_orc, rate_cur=rate_cur, rate_orc=float(rate[b_orc]), gt_rate=gt_rate)

        p(f'\n  [{cat}]  gt_rate={gt_rate*100:.3f}%   mu={mu:.4f} sigma={sd:.4f}')
        p('    ' + '  '.join(f'k={k:<4g}:{v:.3f}' for k, v in f1_ks))
        p(f'    k=4.5 (đang dùng): F1={cur:.4f}  rate={rate_cur*100:.3f}%')
        p(f'    k* = {k_star:<5g}       : F1={f1_star:.4f}   (+{f1_star-cur:+.4f})')
        p(f'    ORACLE (k~{k_orc:.2f}) : F1={f1[b_orc]:.4f}  rate={rate[b_orc]*100:.3f}%')
        p(f'    -> mu+k*sigma bỏ lỡ {f1[b_orc]-f1_star:.4f} so với trần  '
          f'| over-detect x{rate_cur/max(rate[b_orc],1e-12):.2f} so với oracle')

    if not rows:
        return
    p('\n' + '=' * 100)
    p(f'  {"cat":12s} {"k=4.5":>7s} {"k*":>5s} {"F1(k*)":>7s} {"trần":>7s} {"k_orc":>6s} '
      f'{"rate@4.5":>9s} {"rate@orc":>9s} {"gt_rate":>8s}')
    for c, r in rows.items():
        p(f'  {c:12s} {r["cur"]:7.4f} {r["k_star"]:5g} {r["f1_star"]:7.4f} {r["ceil"]:7.4f} '
          f'{r["k_orc"]:6.2f} {r["rate_cur"]*100:8.3f}% {r["rate_orc"]*100:8.3f}% {r["gt_rate"]*100:7.3f}%')
    m_cur = float(np.mean([r['cur'] for r in rows.values()]))
    m_star = float(np.mean([r['f1_star'] for r in rows.values()]))
    m_ceil = float(np.mean([r['ceil'] for r in rows.values()]))
    p('-' * 100)
    p(f'  MEAN   k=4.5 chung : {m_cur:.4f}')
    p(f'  MEAN   k* PER-CAT  : {m_star:.4f}   (+{m_star-m_cur:+.4f})   <-- lấy được ngay, không cần ý tưởng mới')
    p(f'  MEAN   TRẦN oracle : {m_ceil:.4f}   (còn {m_ceil-m_star:.4f} mà mu+k*sigma KHÔNG với tới)')
    p('\nĐỌC:')
    p('  * F1(k*) ~ trần  => quy tắc mu+k*sigma ĐÚNG DẠNG, chỉ hằng số k sai => per-cat k là xong.')
    p('  * F1(k*) << trần => mu+k*sigma SAI DẠNG (đuôi không gaussian) => phải đổi sang quy tắc rate/phân vị.')
    p('  * rate@4.5 >> rate@orc => đang OVER-detect: ngưỡng quá thấp, precision bị giết.')


if __name__ == '__main__':
    main()
