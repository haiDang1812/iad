# -*- coding: utf-8 -*-
# =============================================================================
# eval_raterule.py — NGƯỠNG THEO RATE (phân vị) vs NGƯỠNG THEO k*sigma, DƯỚI DOMAIN SHIFT
#
# PHÁT HIỆN TỪ eval_kcurve (7 cat, ViT-L, native):
#   k_orc nhảy 3.75 -> 8.77 giữa các cat  => KHÔNG tồn tại hằng số k chung.
#   NHƯNG rate@orc / gt_rate ~ 1.0 ở CẢ 7 cat => ngưỡng tối ưu chỉ đang làm 1 việc:
#   cắt sao cho SỐ PIXEL ĐƯỢC GÁN defect ~ SỐ PIXEL DEFECT THẬT.
#
# GIẢ THUYẾT (cái script này kiểm):
#   k    = thuộc tính của PHÂN BỐ SCORE  -> đuôi phân bố DỊCH khi ánh sáng đổi -> k CHẾT.
#   rate = thuộc tính của DEFECT         -> defect không to ra khi đèn đổi     -> rate SỐNG.
#   Ngưỡng phân vị BẤT BIẾN với mọi biến đổi đơn điệu của score; mu+k*sigma thì không.
#   => giải thích cú tụt private (public ~0.635 ngang SuperADD 0.626, nhưng private 0.458 vs 0.574).
#
# THIẾT KẾ: bank dựng từ TRAIN SẠCH (đúng như lúc submit), ảnh TEST bị bóp méo quang học.
#   4 quy tắc, ĐỀU fit trên public SẠCH, rồi áp lên test ĐÃ BÓP MÉO:
#     R1  k=4.5 chung          (pipeline hiện tại)
#     R2  k* per-cat           (fit sạch — "chữa hằng số")
#     R3  RATE per-cat         (fit sạch — "chữa đại lượng")   <-- thứ đang thử
#     R4  ORACLE trên ảnh méo  (trần — không dùng được, chỉ để biết mất mát đến từ dao hay từ map)
#
# ĐỌC:
#   R3 tụt ÍT hơn R1/R2  => rate là đại lượng chuyển miền được => đây là cần gạt cho private.
#   R3 tụt NGANG R1/R2   => shift không đơn điệu, giả thuyết sai, quay về k* per-cat (vẫn +0.0436).
#   R4 cũng tụt mạnh     => domain shift phá MAP chứ không phải phá dao => bài toán khác hẳn.
#
# CHẠY 2 LƯỢT:
#   (A) lấy hằng số đem submit — toàn bộ test_public, không bóp méo:
#       --max_eval 0 --perts none
#   (B) stress test — subsample + bóp méo:
#       --max_eval 20 --perts none b0.8 b1.2 g0.7 g1.5 c0.8
# =============================================================================
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


# ---------------------------------------------------------------- bóp méo quang học
def perturb(pil, tag):
    """Mô phỏng domain shift ánh sáng của private set. Áp lên TEST, KHÔNG áp lên train."""
    if tag == 'none':
        return pil
    kind, val = tag[0], float(tag[1:])
    if kind == 'b':                                   # brightness
        return ImageEnhance.Brightness(pil).enhance(val)
    if kind == 'c':                                   # contrast
        return ImageEnhance.Contrast(pil).enhance(val)
    if kind == 'g':                                   # gamma
        lut = [min(255, int(255.0 * ((i / 255.0) ** val) + 0.5)) for i in range(256)]
        return pil.point(lut * len(pil.getbands()))
    raise ValueError(tag)


# ---------------------------------------------------------------- đọc ngưỡng từ histogram
def curves(h):
    tp = np.cumsum(h.pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(h.neg[::-1])[::-1].astype(np.float64)
    fn = float(h.pos.sum()) - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    rate = (tp + fp) / max(h.n, 1)                    # tỉ lệ pixel bị gán defect nếu cắt tại bin i
    return f1, rate


def thr_at_rate(h, r):
    """Ngưỡng PHÂN VỊ: cắt sao cho đúng r phần pixel của POOL ĐÍCH nằm trên ngưỡng.
    Bất biến với mọi biến đổi đơn điệu của score — đó là toàn bộ lý do dùng nó."""
    _, rate = curves(h)
    b = int(np.argmin(np.abs(rate - r)))
    return h.lo + (b + 0.5) / len(h.pos) * (h.hi - h.lo)


def gt_rate_full(ds):
    """r_cat THẬT trên TOÀN split (kể cả ảnh good) — đây là hằng số đem đi submit."""
    npos, ntot, nb = 0, 0, 0
    for i in range(len(ds.img_paths)):
        if ds.labels[i] == 1:
            g = np.asarray(Image.open(ds.gt_paths[i]).convert('L'))
            npos += int((g > 127).sum())
            ntot += g.size
            nb += 1
        else:
            w, hh = Image.open(ds.img_paths[i]).size
            ntot += w * hh
    return npos / max(ntot, 1), nb, len(ds.img_paths) - nb


# ---------------------------------------------------------------- 1 cat
def hist_for(bb, ds, idx, bank, head, args, layers, gk, device, pert):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch
    hw = args.head_w
    C = bank.shape[-1]
    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {pert:5s}', leave=False):
            pil = perturb(Image.open(ds.img_paths[i]).convert('RGB'), pert)
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt, layers, args.enc_batch)
            d = np.asarray(nn_map(g, bank, device))
            pr = torch.sigmoid(head(g.reshape(-1, C))).reshape(g.shape[0], g.shape[0]).cpu().numpy()
            grids.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])
    s = [((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32) for d, pr in grids]

    h = Hist(min(float(x.min()) for x in s) - 0.05, max(float(x.max()) for x in s) + 0.05)
    for (sg, (H, W), i) in zip(s, sizes, idx):
        m = make_map(sg, 256, gk, (H, W), device).cpu().numpy()
        if ds.labels[i] == 0:
            g = np.zeros((H, W), np.uint8)
        else:
            g = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)
        h.add(m.reshape(-1), g.reshape(-1))
    return h


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] bỏ: train/good rỗng')
        return None
    bank = build_bank(bb, tr[:args.max_train] if args.max_train else tr,      # bank từ TRAIN SẠCH
                      T, R, gt, layers, args.enc_batch, args.bank_size, device)
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    if not bad:
        p(f'  [{cat}] bỏ: test_public không có ảnh bad')
        return None
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        p(f'  [{cat}] bỏ: head None')
        return None

    r_full, nb, ng = gt_rate_full(ds)
    rest = [i for i in bad if i not in set(bad[:args.shots])]
    if args.max_eval:
        idx = rest[:args.max_eval] + good[:args.max_eval]
    else:
        idx = rest + good
    p(f'\n  [{cat}]  split: {nb} bad / {ng} good   r_cat(TOÀN split)={r_full*100:.4f}%   '
      f'|  đánh giá trên {len(idx)} ảnh')

    # ---- fit 3 quy tắc trên ảnh SẠCH
    h0 = hist_for(bb, ds, idx, bank, head, args, layers, gk, device, 'none')
    f1_0, rate_0 = curves(h0)
    b0 = int(np.argmax(f1_0))
    r_star = float(rate_0[b0])                        # RATE tối ưu, fit sạch  -> R3
    ks = np.arange(2.0, 12.01, 0.25)
    k_star = float(max(ks, key=lambda k: h0.f1_at(h0.ksig(k))))   # k* per-cat, fit sạch -> R2
    p(f'    fit(sạch): k*={k_star:.2f}   r*={r_star*100:.4f}%   (trần sạch={f1_0[b0]:.4f})')

    out = {}
    for pt in args.perts:
        h = h0 if pt == 'none' else hist_for(bb, ds, idx, bank, head, args, layers, gk, device, pt)
        f1, _ = curves(h)
        mu = h.s1 / max(h.n, 1)
        sd = float(np.sqrt(max(h.s2 / max(h.n, 1) - mu * mu, 0.0)))
        out[pt] = dict(
            R1=h.f1_at(h.ksig(args.thr_sigma)),       # k=4.5 chung
            R2=h.f1_at(h.ksig(k_star)),               # k* per-cat (fit sạch)
            R3=h.f1_at(thr_at_rate(h, r_star)),       # RATE per-cat (fit sạch)
            R4=float(np.max(f1)),                     # oracle trên chính ảnh méo
            mu=mu, sd=sd,
        )
        d = out[pt]
        p(f'    {pt:5s}  R1(k=4.5)={d["R1"]:.4f}  R2(k*)={d["R2"]:.4f}  R3(RATE)={d["R3"]:.4f}  '
          f'| trần={d["R4"]:.4f}  mu={mu:.4f} sd={sd:.4f}')
    out['_r'] = r_star
    out['_k'] = k_star
    out['_rfull'] = r_full
    return out


def main():
    ap = argparse.ArgumentParser('eval_raterule: ngưỡng RATE vs k*sigma dưới domain shift')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--perts', type=str, nargs='+', default=['none'],
                    help="none | b<val> brightness | c<val> contrast | g<val> gamma. VD: b0.8 g1.5")
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
    ap.add_argument('--max_eval', type=int, default=20, help='0 = TOÀN BỘ test_public')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./raterule')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('raterule', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4.0).to(device)
    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} perts={args.perts} '
      f'| bank từ TRAIN SẠCH, bóp méo chỉ áp lên TEST')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        return

    # ---- hằng số đem submit
    p('\n' + '=' * 100)
    p('  HẰNG SỐ CHO SUBMIT (fit trên test_public sạch)')
    p(f'  {"cat":12s} {"k*":>6s} {"r* (pool eval)":>15s} {"r_cat (toàn split)":>19s}')
    for c, r in res.items():
        p(f'  {c:12s} {r["_k"]:6.2f} {r["_r"]*100:14.4f}% {r["_rfull"]*100:18.4f}%')

    # ---- bảng chính: quy tắc nào sống sót
    p('\n' + '=' * 100)
    p('  SEGF1 TRUNG BÌNH THEO QUY TẮC  (mọi quy tắc FIT TRÊN SẠCH, ÁP LÊN MÉO)')
    p(f'  {"pert":6s} {"R1 k=4.5":>9s} {"R2 k* cat":>10s} {"R3 RATE":>9s} {"R4 trần":>9s}')
    base = {}
    for pt in args.perts:
        row = {r_: float(np.mean([res[c][pt][r_] for c in res])) for r_ in ('R1', 'R2', 'R3', 'R4')}
        if pt == 'none':
            base = row
        p(f'  {pt:6s} {row["R1"]:9.4f} {row["R2"]:10.4f} {row["R3"]:9.4f} {row["R4"]:9.4f}')
    if base and len(args.perts) > 1:
        p('\n  TỤT so với ảnh sạch (âm = mất điểm):')
        p(f'  {"pert":6s} {"R1 k=4.5":>9s} {"R2 k* cat":>10s} {"R3 RATE":>9s} {"R4 trần":>9s}')
        drop = {r_: [] for r_ in ('R1', 'R2', 'R3', 'R4')}
        for pt in args.perts:
            if pt == 'none':
                continue
            row = {r_: float(np.mean([res[c][pt][r_] for c in res])) for r_ in ('R1', 'R2', 'R3', 'R4')}
            for r_ in drop:
                drop[r_].append(row[r_] - base[r_])
            p(f'  {pt:6s} ' + ' '.join(f'{row[r_]-base[r_]:+9.4f}' if r_ != 'R2' else
                                       f'{row[r_]-base[r_]:+10.4f}' for r_ in ('R1', 'R2', 'R3', 'R4')))
        p('  ' + '-' * 50)
        p(f'  {"TB tụt":6s} ' + ' '.join(f'{np.mean(drop[r_]):+9.4f}' if r_ != 'R2' else
                                         f'{np.mean(drop[r_]):+10.4f}' for r_ in ('R1', 'R2', 'R3', 'R4')))
    p('\nĐỌC:')
    p('  R3 tụt ÍT hơn R1/R2  => rate CHUYỂN MIỀN ĐƯỢC, k thì không => đây là cần gạt cho private.')
    p('  R3 tụt NGANG R1/R2   => shift không đơn điệu => giả thuyết SAI, chốt R2 (vẫn +0.0436 public).')
    p('  R4 (trần) cũng tụt mạnh => shift phá MAP chứ không phá dao => bài toán hoàn toàn khác.')


if __name__ == '__main__':
    main()
