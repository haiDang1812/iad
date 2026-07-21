# eval_hysteresis.py
# -----------------------------------------------------------------------------
# PREMISE TEST — hysteresis thresholding cho png (SegF1).
#
# Bối cảnh (server 2026-07-20, sub v3): AUPRO 81.27 (transfer OK), SegF1 51.75
#   (≈ flat vs v2 51.20). Đòn "thêm supervision" cứu AUPRO KHÔNG cứu SegF1 -> ĐÓNG.
#   Gate png CHẾT (can byte-identical, wallplugs nhiễu). Lever còn sống = khoảng
#   TRẦN−ksig trên public: wallplugs trần 0.678 vs ksig 0.495 (kẹt 0.18), vial
#   0.548 vs 0.396 (kẹt 0.15). can KHÔNG kẹt ngưỡng (trần 0.316 ≈ ksig 0.293 = map
#   yếu, không cứu bằng ngưỡng).
#
# Ý tưởng (CV cổ điển, own design): 2 ngưỡng + connected-component.
#   seed = map > τ_hi  (điểm chắc chắn defect, ít FP)
#   grow = map > τ_lo  (nới rộng, τ_lo < τ_hi)
#   mask = các thành phần liên thông của `grow` CÓ CHỨA ít nhất 1 pixel seed.
#   -> FP lốm đốm (dưới τ_hi, không dính seed) BỊ XOÁ => precision lên;
#      vùng defect thật nới xuống τ_lo => recall lên.
#   Đây là cơ chế duy nhất có thể VƯỢT trần single-threshold (trần = giới hạn của
#   MỘT ngưỡng; hysteresis = 2 ngưỡng + lọc không gian).
#
# τ_hi = mu + k_hi·σ (pooled, y hệt ksig), τ_lo = mu + k_lo·σ, k_lo < k_hi.
#   Self-calibrating theo mu/σ đích -> tự trượt theo shift như ksig, KHÔNG cần GT
#   đích, KHÔNG cần estimator. Sweep k_hi ∈ K_HI, k_lo ∈ K_LO -> best pooled F1.
#
# Đối chứng cùng map/head/pipeline (chỉ khác luật ngưỡng):
#   ksig   = single-threshold mu+4.5σ (đúng submission hiện tại)
#   trần   = single-threshold ORACLE (best bin pooled) = giới hạn 1 ngưỡng
#   hyst   = best hysteresis qua lưới (k_hi,k_lo)
# Robust prevalence: chấm lại trên bad×{1,0.5,0.25} -> worst-case (kiểu vỡ private).
#
# ĐỌC (pre-register): hysteresis SỐNG nếu trên {can,vial,wallplugs}:
#   (1) hyst_best ≥ ksig + 0.03  VÀ
#   (2) hyst_best ≥ trần (single-oracle) trên ≥2/3 cat
#       -> hysteresis làm được điều KHÔNG ngưỡng đơn nào làm được => lever mới thật,
#          nấu (τ_hi=4.5σ, τ_lo=best-k_lo, CC) vào infer png cho sub v4.
#   Nếu hyst ≤ trần khắp nơi -> chỉ là 1 điểm trên frontier ngưỡng-đơn, GIẾT.
#
#   python eval_hysteresis.py --data_path ../data --out_dir ./hyst \
#       --categories can vial wallplugs fabric
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
from scipy.ndimage import label as cc_label, generate_binary_structure

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, IMG_EXT,
)
from infer_submit_nrs import PER_CAT                                    # noqa: E402
from eval_nrs_head import build_nrs_head                                # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
FRACS = [1.0, 0.5, 0.25]                     # subsample ảnh bad (mô phỏng prevalence private)
K_HI = [3.5, 4.5]                            # seed sigma
K_LO = [1.0, 1.5, 2.0, 2.5, 3.0]            # grow sigma (< k_hi)
STRUCT = generate_binary_structure(2, 2)     # 8-connectivity cho grow


def f1_from_counts(tp, fp, fn):
    return 2.0 * tp / (2.0 * tp + fp + fn + 1e-9)


def single_f1(maps, gts, sel, tau):
    tp = fp = fn = 0.0
    for j in sel:
        m, gm = maps[j], gts[j]
        pred = m > tau
        tp += float(np.count_nonzero(pred & gm))
        fp += float(np.count_nonzero(pred & ~gm))
        fn += float(np.count_nonzero(~pred & gm))
    return f1_from_counts(tp, fp, fn)


def single_oracle(maps, gts, sel, cand):
    best = 0.0
    for tau in cand:
        f = single_f1(maps, gts, sel, tau)
        if f > best:
            best = f
    return best


def hyst_mask(m, thi, tlo):
    grow = m > tlo
    if not grow.any():
        return np.zeros_like(grow)
    seed = m > thi
    if not seed.any():
        return np.zeros_like(grow)
    lab, n = cc_label(grow, structure=STRUCT)
    if n == 0:
        return np.zeros_like(grow)
    keep = np.unique(lab[seed])
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros_like(grow)
    return np.isin(lab, keep)


def hyst_f1(maps, gts, sel, thi, tlo):
    tp = fp = fn = 0.0
    for j in sel:
        m, gm = maps[j], gts[j]
        pred = hyst_mask(m, thi, tlo)
        tp += float(np.count_nonzero(pred & gm))
        fp += float(np.count_nonzero(pred & ~gm))
        fn += float(np.count_nonzero(~pred & gm))
    return f1_from_counts(tp, fp, fn)


def run_cat(bb, cat, args, layers, gk, device, p):
    cfg = PER_CAT[cat]
    args.tiles, args.grid_tile = cfg['tiles'], cfg['grid_tile']
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    p(f'  [{cat}] cfg={T}:{gt_} f1_head={cfg["f1_head"]} | bank từ {len(tr)} ảnh...')
    bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shots = bad[:args.shots]
    torch.manual_seed(args.seed)                       # ĐÚNG thứ tự seed của infer_submit_nrs
    head_grid = build_head(bb, ds, shots, bank, args, layers, device)
    head_nrs = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)
    if head_nrs is None:
        return None
    head_f1 = head_nrs if (cfg['f1_head'] == 'nrs' or head_grid is None) else head_grid

    idx_bad = [i for i in bad if i not in set(shots)]
    idx_good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    if args.max_eval:
        idx_bad = idx_bad[:args.max_eval]
        idx_good = idx_good[:args.max_eval]
    idx = idx_bad + idx_good

    # PASS 1: fused grid + normalization (pooled lo/hi giống submission)
    grids, sizes, is_bad_lst = [], [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat} enc', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            is_bad_lst.append(ds.labels[i] == 1)
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]
            d = np.asarray(nn_map(g, bank, device))
            pr = torch.sigmoid(head_f1(g.reshape(-1, C))).reshape(G, G).cpu().numpy()
            grids.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])
    s_grids = [((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32) for d, pr in grids]

    # native map (float16) + GT native (bool) + pooled moments cho mu/σ
    maps, gts = [], []
    s1 = s2 = 0.0
    npix = 0
    for s, (H, W), i in zip(s_grids, sizes, idx):
        with torch.no_grad():
            t = torch.tensor(s, device=device)[None, None].float()
            t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
            t = gk(t)
            m = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        if ds.labels[i] == 0:
            gm = np.zeros((H, W), bool)
        else:
            gm = np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127
        maps.append(m.astype(np.float16))
        gts.append(gm)
        s1 += float(m.sum(dtype=np.float64))
        s2 += float(np.square(m, dtype=np.float64).sum())
        npix += m.size
    del grids, s_grids

    mu = s1 / npix
    sd = float(np.sqrt(max(s2 / npix - mu * mu, 0.0)))
    # ứng viên trần: percentile lưới trên phân bố pooled (nhẹ RAM)
    samp = np.concatenate([m.reshape(-1)[::7].astype(np.float32) for m in maps])
    cand = np.percentile(samp, np.linspace(80, 99.98, 60))
    del samp
    p(f'    [{cat}] mu={mu:.4f} sd={sd:.4f} ksig(4.5σ)={mu + 4.5 * sd:.4f} '
      f'(bad {sum(is_bad_lst)} / good {len(is_bad_lst) - sum(is_bad_lst)} ảnh)')

    bad_pos = [j for j, b in enumerate(is_bad_lst) if b]
    good_pos = [j for j, b in enumerate(is_bad_lst) if not b]

    out = {}
    for f in FRACS:
        sel = good_pos + bad_pos[:max(1, round(f * len(bad_pos)))]
        ksig = single_f1(maps, gts, sel, mu + 4.5 * sd)
        tran = single_oracle(maps, gts, sel, cand)
        best_h, best_cfg = 0.0, None
        for khi in K_HI:
            thi = mu + khi * sd
            for klo in K_LO:
                if klo >= khi:
                    continue
                tlo = mu + klo * sd
                fh = hyst_f1(maps, gts, sel, thi, tlo)
                if fh > best_h:
                    best_h, best_cfg = fh, (khi, klo)
        out[f] = dict(ksig=ksig, tran=tran, hyst=best_h, cfg=best_cfg)
        p(f'    [{cat}] bad x{f:<4}: ksig={ksig:.4f}  trần={tran:.4f}  '
          f'hyst={best_h:.4f} @k_hi/k_lo={best_cfg}  '
          f'(hyst−ksig={best_h - ksig:+.4f} hyst−trần={best_h - tran:+.4f})')
    del maps, gts
    return out


def main():
    ap = argparse.ArgumentParser('premise test hysteresis thresholding cho png')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
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
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=['can', 'vial', 'wallplugs', 'fabric'])
    ap.add_argument('--out_dir', type=str, default='./hyst')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('hyst', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(v / 12 * bb.n_layers))) for v in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} model={args.model} K_HI={K_HI} K_LO={K_LO} fracs={FRACS} '
      f'| per-cat config + f1_head THEO SUBMISSION (PER_CAT)')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        return

    p('\n' + '=' * 96 + '\n===== TỔNG (mean qua cat) =====')
    for f in FRACS:
        ks = float(np.mean([res[c][f]['ksig'] for c in res]))
        tr = float(np.mean([res[c][f]['tran'] for c in res]))
        hy = float(np.mean([res[c][f]['hyst'] for c in res]))
        p(f'  bad x{f:<4}: ksig={ks:.4f}  trần={tr:.4f}  hyst={hy:.4f}')
    p('\n  WORST-CASE qua prevalence (mean qua cat):')
    for key in ['ksig', 'tran', 'hyst']:
        wc = float(np.mean([min(res[c][f][key] for f in FRACS) for c in res]))
        p(f'    {key:5s}: worst-F1={wc:.4f}')

    p('\nĐỌC (pre-registered): hysteresis SỐNG nếu trên can/vial/wallplugs:')
    p('  (1) hyst ≥ ksig + 0.03  VÀ  (2) hyst ≥ trần trên ≥2/3 cat.')
    p('  -> nấu (τ_hi=4.5σ, τ_lo=best-k_lo, CC 8-conn) vào png cho sub v4.')
    p('  Nếu hyst ≤ trần khắp nơi -> chỉ là 1 điểm trên frontier ngưỡng-đơn -> GIẾT.')


if __name__ == '__main__':
    main()
