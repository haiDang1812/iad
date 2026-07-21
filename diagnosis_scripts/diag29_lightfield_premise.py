# -*- coding: utf-8 -*-
# =============================================================================
# diag29_lightfield_premise.py — PREMISE TEST cho IFC (Illumination-Field Compensation)
# trên DỮ LIỆU PRIVATE THẬT, KHÔNG CẦN GT, KHÔNG TỐN LƯỢT SUBMIT.
#
# GIẢ THUYẾT (từ research + diag24):
#   d(f, bank) = "xa miền train" + "xa normality". Private light-shift (spotlight dịch
#   chỗ, thêm đèn, đổi nhiệt độ màu — theo paper MVTec AD 2) làm số hạng đầu phình ở
#   MỌI patch normal -> nhấn chìm defect -> cả AUPRO lẫn SegF1 tụt.
#   diag24 đã chứng minh (trên sim GLOBAL): độ trôi là HẠNG THẤP + NHẤT QUÁN.
#   IFC đặt cược thêm: trên private THẬT, độ trôi còn TRƠN THEO KHÔNG GIAN (per-image),
#   nên fit được một trường Ω(x,y) từ chính các patch confident-normal rồi khử đi.
#
# 3 CÂU HỎI (đo trên ảnh private thật, so với public làm mốc):
#   Q1 DRIFT   : phân vị ĐÁY distance (q25/q50 — vùng chắc chắn normal vì defect <2.2%
#                pixel) của private có PHÌNH so với public không? Phình = drift có thật.
#   Q2 DẠNG    : residual r_i = f_i - bank[NN] của patch confident-normal (đáy 50%):
#                (a) |mean r| (năng lượng offset) so với sàn nhiễu public
#                (b) EVR top-5 của phần lệch (hạng thấp?)
#                (c) R^2 fit đa thức bậc 2 theo (x,y) trên 3 PC đầu (TRƠN không gian?)
#   Q3 MIXED   : private_mixed (đèn lạ) tệ hơn private (đèn quen) đúng kiểu đèn không?
#
# QUYẾT ĐỊNH:
#   Q1 phình + Q2 (a) cao (b) cao (c) cao  => trường Ω tồn tại, IFC đáng dựng.
#   Q1 phình + Q2 thấp                     => drift KHÔNG cấu trúc -> IFC chết từ trứng,
#                                             và đó cũng là một phát hiện (đỡ tốn 1 tuần).
#   Q1 KHÔNG phình                         => private tụt vì defect lạ, không phải đèn
#                                             -> toàn bộ hướng domain-shift là ảo giác.
#
#   python diag29_lightfield_premise.py --data_path ../data --out_dir ./diag29
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
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                    # noqa: E402
    build_bank, img_featgrid, VALID, IMG_EXT,
)
from backbones_ext import load_backbone                                  # noqa: E402
from utils import get_logger                                             # noqa: E402

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')
SPLITS = ['test_public', 'test_private', 'test_private_mixed']


@torch.no_grad()
def nn_res(flat, bank, chunk=2048):
    """distance + residual vector tới NN trong bank. flat [N,C] trên device."""
    d = torch.empty(flat.shape[0], device=flat.device)
    idx = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
    for s in range(0, flat.shape[0], chunk):
        dd = torch.cdist(flat[s:s + chunk], bank)
        m = dd.min(1)
        d[s:s + chunk] = m.values
        idx[s:s + chunk] = m.indices
    return d, flat - bank[idx]


def smooth_r2(res, xy, n_pc=3):
    """Residual chiếu lên n_pc hướng chính -> fit đa thức bậc 2 theo (x,y) -> R^2 TB.
    R^2 cao = trường trôi TRƠN theo không gian = fit được Ω(x,y)."""
    rc = res - res.mean(0, keepdims=True)
    # PCA qua SVD trên phần lệch (offset trung bình đã tách riêng ở |mean r|)
    U, S, Vt = np.linalg.svd(rc, full_matrices=False)
    pcs = rc @ Vt[:n_pc].T                                # [N, n_pc]
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
    r2s = []
    for j in range(n_pc):
        coef, *_ = np.linalg.lstsq(A, pcs[:, j], rcond=None)
        pred = A @ coef
        ss_res = float(((pcs[:, j] - pred) ** 2).sum())
        ss_tot = float(((pcs[:, j] - pcs[:, j].mean()) ** 2).sum()) + 1e-12
        r2s.append(1.0 - ss_res / ss_tot)
    evr5 = float((S[:5] ** 2).sum() / (S ** 2).sum().clip(min=1e-12))
    return float(np.mean(r2s)), evr5


def split_paths(data_path, cat, split):
    if split == 'test_public':
        # gộp good + bad — private cũng trộn cả hai, so sánh phải cùng thành phần
        fs = []
        for sub in ('good', 'bad'):
            for e in IMG_EXT:
                fs += glob.glob(os.path.join(data_path, cat, 'test_public', sub, e))
        return sorted(fs)
    fs = []
    for e in IMG_EXT:
        fs += glob.glob(os.path.join(data_path, cat, split, e))
    return sorted(fs)


def run_cat(bb, cat, args, layers, device, p):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch
    G = T * gt
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] bỏ: train/good rỗng')
        return None
    bank = build_bank(bb, tr[:args.max_train] if args.max_train else tr,
                      T, R, gt, layers, args.enc_batch, args.bank_size, device)
    # toạ độ lưới chuẩn hoá [0,1] cho fit trường
    yy, xx = np.meshgrid(np.linspace(0, 1, G), np.linspace(0, 1, G), indexing='ij')
    xy_all = np.stack([xx.reshape(-1), yy.reshape(-1)], 1).astype(np.float32)

    out = {}
    for split in SPLITS:
        paths = split_paths(args.data_path, cat, split)
        if not paths:
            p(f'  [{cat}/{split}] bỏ: không có ảnh')
            continue
        rng.shuffle(paths)
        paths = paths[:args.max_imgs]
        q25s, q50s, q90s, offs, evrs, r2s = [], [], [], [], [], []
        with torch.no_grad():
            for pth in tqdm(paths, ncols=70, desc=f'    {split[:16]:16s}', leave=False):
                g = img_featgrid(bb, Image.open(pth).convert('RGB'), T, R, gt, layers, args.enc_batch)
                flat = g.reshape(-1, g.shape[-1])
                d, res = nn_res(flat, bank)
                dn = d.cpu().numpy()
                q25s.append(float(np.percentile(dn, 25)))
                q50s.append(float(np.percentile(dn, 50)))
                q90s.append(float(np.percentile(dn, 90)))
                # confident-normal = đáy args.conf_q theo distance (defect <2.2% pixel -> sạch)
                keep = dn <= np.percentile(dn, args.conf_q * 100)
                rk = res.cpu().numpy()[keep]
                offs.append(float(np.linalg.norm(rk.mean(0))))
                r2, evr = smooth_r2(rk, xy_all[keep])
                r2s.append(r2)
                evrs.append(evr)
        out[split] = dict(n=len(paths),
                          q25=float(np.median(q25s)), q50=float(np.median(q50s)),
                          q90=float(np.median(q90s)), off=float(np.mean(offs)),
                          evr5=float(np.mean(evrs)), r2=float(np.mean(r2s)))
    if 'test_public' not in out:
        return None

    b = out['test_public']
    p(f'\n  [{cat}]  (mốc = test_public; q* = median per-image; off/EVR/R2 = mean per-image)')
    p(f'    {"split":20s} {"n":>4s} {"q25":>8s} {"q50":>8s} {"q90":>8s} '
      f'{"|mean r|":>9s} {"EVR5":>6s} {"R2xy":>6s} {"q50/pub":>8s} {"off/pub":>8s}')
    for split in SPLITS:
        if split not in out:
            continue
        o = out[split]
        p(f'    {split:20s} {o["n"]:4d} {o["q25"]:8.4f} {o["q50"]:8.4f} {o["q90"]:8.4f} '
          f'{o["off"]:9.4f} {o["evr5"]:6.3f} {o["r2"]:6.3f} '
          f'{o["q50"]/max(b["q50"],1e-9):8.3f} {o["off"]/max(b["off"],1e-9):8.3f}')
    return out


def main():
    ap = argparse.ArgumentParser('diag29: premise IFC — drift trên private thật, không cần GT')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_imgs', type=int, default=25, help='số ảnh mỗi split (3 split/cat)')
    ap.add_argument('--conf_q', type=float, default=0.5, help='đáy phân vị lấy confident-normal')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag29')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag29', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} conf_q={args.conf_q} '
      f'| bank từ TRAIN SẠCH; private KHÔNG cần GT')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        return

    p('\n' + '=' * 100)
    p('  TỔNG (mean trên các cat, tỉ số so với test_public của chính cat đó):')
    p(f'  {"split":20s} {"q50/pub":>8s} {"q90/pub":>8s} {"off/pub":>8s} {"EVR5":>6s} {"R2xy":>6s}')
    for split in SPLITS:
        rows = [res[c][split] for c in res if split in res[c]]
        base = [res[c]['test_public'] for c in res if split in res[c]]
        if not rows:
            continue
        q50r = float(np.mean([r['q50'] / max(b['q50'], 1e-9) for r, b in zip(rows, base)]))
        q90r = float(np.mean([r['q90'] / max(b['q90'], 1e-9) for r, b in zip(rows, base)]))
        offr = float(np.mean([r['off'] / max(b['off'], 1e-9) for r, b in zip(rows, base)]))
        evr = float(np.mean([r['evr5'] for r in rows]))
        r2 = float(np.mean([r['r2'] for r in rows]))
        p(f'  {split:20s} {q50r:8.3f} {q90r:8.3f} {offr:8.3f} {evr:6.3f} {r2:6.3f}')

    p('\nĐỌC:')
    p('  Q1: q50/pub của private_mixed > ~1.1  => normal-manifold drift CÓ THẬT trên dữ liệu thật.')
    p('  Q2: off/pub cao + EVR5 cao (>~0.6) + R2xy cao (>~0.4) => trôi CÓ CẤU TRÚC (hạng thấp,')
    p('      trơn không gian) => trường Ω(x,y) fit được => IFC đáng dựng.')
    p('  Q3: mixed >> private thường => đúng chữ ký của ĐÈN LẠ (không phải defect lạ).')
    p('  Nếu q50/pub ~ 1.0 ở cả 2 private => drift không tồn tại => hướng domain-shift là ảo giác,')
    p('      private tụt vì lý do khác (defect khó hơn / tỉ lệ split) => DỪNG IFC tại đây.')


if __name__ == '__main__':
    main()
