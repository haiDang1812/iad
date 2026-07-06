# eval_segf1_resolution.py
# -----------------------------------------------------------------------------
# ABLATION ĐỘ PHÂN GIẢI cho SegF1 (đo trên PUBLIC có nhãn, ĐÚNG giao thức server).
#
# ĐỘNG CƠ (chốt 2026-07-05): submit thật cho SegF1 private = 0.30 (public 0.644).
#   class-F1 0.78 + AUPRO0.05 0.67 -> feature RANKING không hỏng, mask THÔ/BIÊN LỆCH là
#   nút thắt. eval_multiscale_hires đã ghi: bottleneck = ĐỘ PHÂN GIẢI (defect nhỏ bị lưới
#   thô nuốt). Tăng tiles -> lưới mịn -> recall defect nhỏ + biên mask sắc -> nâng CẢ
#   SegF1 LẪN AUPRO0.05. Đây là nâng SegF1 BẢN CHẤT (transfer sang private), KHÔNG hack ngưỡng.
#
# KHÁC eval_fewshot_segf1 (cũ): (a) DÙNG ĐÚNG pipeline production (DINOv3 + SoftPRO head +
#   global-norm distance + fuse hw=0.6, import thẳng từ infer_submit_mvtec_ad2 -> parity),
#   (b) đo SegF1 tại NGƯỠNG CỐ ĐỊNH mean+3σ trên validation/good = kịch bản server,
#   KHÔNG P-F1_max oracle (oracle khen nhầm phương án không transfer). Oracle vẫn in để
#   tham chiếu headroom.
#
# Split public: shot_pool = 10 ảnh defect train head (giống submit) -> LOẠI khỏi eval;
#   eval trên phần defect còn lại + good. Không leak.
#
# CẬP NHẬT 2026-07-05: ablation resolution BÁC BỎ (tiles=3 giết AUPRO −0.11, SegF1 +0.015 nhiễu;
#   morph vô dụng). Cột oracle lộ nút thắt SegF1 = OPERATING POINT (gap 0.354->0.575 = 0.22 thuần
#   ngưỡng; precision chết vì mean+3σ đặt quá thấp + ảnh good bị bôi đen).
#   GATE (+0.042 mean, an toàn) đã xác nhận: ảnh good rỉ FP là 1 phần cú sụp. GATE_SC (percentile
#   CỐ ĐỊNH) bimodal — cứu rice/wallplugs (+0.2~0.4) nhưng GIẾT fabric/walnuts (defect lớn > 1%
#   -> ép top-1% cắt mất defect). => operating-point phải THÍCH NGHI diện tích/ảnh, không cố định.
#   Script quét (đều +gate ảnh normal) ở tiles=2:
#     BASE      = mean+3σ (hiện tại)   GATE      = +bỏ mask ảnh normal
#     GATE_SC   = +percentile cố định  GATE_OTSU = +Otsu/ảnh (tự tách 2 mode, không prior diện tích)
#     GATE_ACAP = +area-cap/ảnh (chỉ NÂNG ngưỡng khi BASE phủ > cap%)
#   Tất cả KHÔNG nhìn nhãn -> áp y hệt private. oracle = cận trên tham chiếu.
#
# LƯU Ý THƯỚC ĐO: harness đọc BASE~0.35 @256 vs server public 0.644 @native. Muốn số KHỚP server
#   (mục tiêu 0.6x) PHẢI chạy --eval_res 512 (gần native, F1 tuyệt đối cao hơn). 256 chỉ để quét nhanh.
#
# NOVELTY TEST (--sim_shift): so |Δrel| (độ tụt SegF1 dưới light-shift mô phỏng) giữa NGƯỠNG CỐ ĐỊNH
#   (BASE mean+3σ, P95G p95×gain kiểu SuperADD — calib clean, KHÔNG đổi khi test shift) vs TỰ-HIỆU-CHỈNH
#   (GATE_OTSU/GATE_ACAP — tính lại per-ảnh). Luận đề đứng nếu |Δrel| của ta NHỎ HƠN RÕ = robust hơn.
#   Đây là bài test sống-còn TRƯỚC khi đầu tư chất nền hi-res.
#
# Chạy (server, public-only, CHƯA đụng submit):
#   # test novelty robustness (rẻ):
#   HF_HUB_OFFLINE=1 python eval_segf1_resolution.py --data_path ../data --sim_shift \
#       --gate_k 3 --area_cap 5 --p95_gain 1.4 --eval_res 256 --max_eval 60 --out_dir ./op_robust
#   # xác nhận @512 nếu luận đề đứng:  --sim_shift --eval_res 512
# -----------------------------------------------------------------------------

import os
import glob
import argparse
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageEnhance

from dataset import MVTecAD2Dataset
from utils import get_gaussian_kernel, get_logger, ader_evaluator
from backbones_ext import load_backbone

# import NGUYÊN các hàm production -> đảm bảo khớp submit từng dòng
from infer_submit_mvtec_ad2 import (
    build_bank, build_head, score_grid, up_to, gt_grid,
    VALID, IMG_EXT,
)

warnings.filterwarnings("ignore")

METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
# đo mọi config ở CÙNG khung -> so được. 256 = nhanh (quét chọn config);
# 512 = gần native server hơn -> con số SegF1 thật hơn (chậm hơn ở AUPRO CPU).
# up_to vẫn smooth ở SMOOTH_RES=256 rồi upsample lên eval_res (khớp production 256->native).


def morph_close(arr2d, k):
    if k <= 0:
        return arr2d
    from scipy import ndimage
    return ndimage.grey_closing(arr2d, size=(k, k))


def seg_f1_at_thr(preds, gts, thr):
    """SegF1 pooled-pixel tại 1 ngưỡng cố định (semantics server)."""
    tp = fp = fn = 0
    for pr, g in zip(preds, gts):
        pm = pr > thr
        gm = g > 0
        tp += int(np.logical_and(pm, gm).sum())
        fp += int(np.logical_and(pm, ~gm).sum())
        fn += int(np.logical_and(~pm, gm).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def otsu_threshold(x, nbins=256):
    """Otsu per-image: tách histogram score thành 2 mode (normal/defect), KHÔNG cần prior diện tích."""
    x = x.reshape(-1)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return hi
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = hist.astype(float) / max(1.0, hist.sum())
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom < 1e-12] = 1e-12
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.nanargmax(sigma_b2))])


def robust_thr(vals, k):
    """Ngưỡng robust median+k*MAD (bất-biến dạng shift: tự căn theo quần thể-normal của CHÍNH tập test)."""
    vals = np.asarray(vals).reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return med + k * mad


def per_image_thr(pr, mode, pixel_thr, p, cap):
    """Ngưỡng LABEL-FREE cho 1 ảnh (đã qua gate). mode: pixel|pct|otsu|acap."""
    if mode == 'pct':
        return np.percentile(pr, p)                              # top-(100-p)% ảnh này
    if mode == 'otsu':
        return otsu_threshold(pr)                                # tự tách 2 mode
    if mode == 'acap':
        return max(pixel_thr, np.percentile(pr, 100.0 - cap))    # chỉ NÂNG khi BASE phủ > cap%
    return pixel_thr                                             # 'pixel' = ngưỡng chung mean+kσ


def seg_f1_gated(preds, gts, img_scores, img_thr, pixel_thr, mode, p=99.0, cap=5.0):
    """SegF1 pooled-pixel, LABEL-FREE operating point.
      - img_scores/img_thr != None: GATE -> ảnh có score < img_thr xuất mask RỖNG (dự đoán normal).
      - mode chọn ngưỡng pixel per-image: pixel(chung mean+kσ) | pct | otsu | acap.
    BASE=(no gate, pixel). GATE=(gate, pixel). GATE_SC=(gate, pct).
    GATE_OTSU=(gate, otsu). GATE_ACAP=(gate, acap)."""
    tp = fp = fn = 0
    for i, (pr, g) in enumerate(zip(preds, gts)):
        if img_scores is not None and img_scores[i] < img_thr:
            pm = np.zeros(pr.shape, dtype=bool)                     # gate: ảnh normal -> rỗng
        else:
            pm = pr > per_image_thr(pr, mode, pixel_thr, p, cap)
        gm = g > 0
        tp += int(np.logical_and(pm, gm).sum())
        fp += int(np.logical_and(pm, ~gm).sum())
        fn += int(np.logical_and(~pm, gm).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def seg_f1_oracle(preds, gts, n=200):
    """SegF1 tối đa khi quét ngưỡng (chỉ để tham chiếu headroom, KHÔNG dùng chọn model)."""
    allv = np.concatenate([p.reshape(-1) for p in preds])
    lo, hi = np.percentile(allv, 50), np.percentile(allv, 99.9)
    best = 0.0
    for t in np.linspace(lo, hi, n):
        f1, _, _ = seg_f1_at_thr(preds, gts, t)
        best = max(best, f1)
    return best


def photometric_shift(pil, s):
    """Mô phỏng light-shift private (khớp diag22): brightness/contrast/color/gamma theo mức s>=0."""
    if s <= 0:
        return pil
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


METHODS = ('BASE', 'P95G', 'GATE', 'GATE_OTSU', 'GATE_ACAP', 'SELFNORM')  # SELFNORM = gate+floor tự-căn/tập test


def _cond_maps(bb, ds, eval_idx, bank, head, a, layers, gk, morph, er, device, srng, shift_lo, shift_hi, do_shift):
    """Score eval set (clean hoặc shift), trả về preds/gts/sp + lo/hi global-norm của chính điều kiện đó."""
    ev = []
    for i in tqdm(eval_idx, ncols=70, desc=f"    {ds_cat(ds)}/{'shift' if do_shift else 'clean'}", leave=False):
        pil = Image.open(ds.img_paths[i])
        if do_shift:
            pil = photometric_shift(pil, float(srng.uniform(shift_lo, shift_hi)))   # per-image, hetero
        d, pr = score_grid(bb, pil, bank, head, a, layers, device)
        g = gt_grid(ds.gt_paths[i], ds.labels[i], d.shape[0])
        ev.append((d, pr, g))
    all_dist = np.concatenate([e[0].reshape(-1) for e in ev])
    lo, hi = np.percentile(all_dist, 1), np.percentile(all_dist, 99)

    def fuse(d, pr):
        dr = (d - lo) / (hi - lo + 1e-8)
        return dr if (head is None or pr is None) else (1 - a.head_w) * dr + a.head_w * pr

    preds, gts, sp = [], [], []
    for d, pr, g in ev:
        m = morph_close(up_to(fuse(d, pr), (er, er), gk, device), morph)
        gg = np.asarray(Image.fromarray(g).resize((er, er), Image.NEAREST)).astype(np.uint8)
        preds.append(m); gts.append(gg)
        k = max(1, int(m.size * 0.01))
        sp.append(np.sort(m.reshape(-1))[::-1][:k].mean())
    return preds, gts, np.array(sp), fuse


def ds_cat(ds):
    return os.path.basename(os.path.normpath(ds.root))


def _methods_for(preds, gts, sp, thr, p95g, img_thr, cap, pix_k=3.0, gate_k=3.0):
    """Operating-point. BASE/P95G/GATE* = ngưỡng CỐ ĐỊNH calib clean. SELFNORM = tự-căn theo CHÍNH tập test."""
    # SELFNORM: gate + pixel-floor tính từ median+MAD của chính điều kiện đang xét (bất-biến dạng shift).
    sn_pthr = robust_thr(np.concatenate([p.reshape(-1) for p in preds]), pix_k)
    sn_ithr = robust_thr(sp, gate_k)
    return {
        'BASE': seg_f1_gated(preds, gts, None, None, thr, 'pixel'),        # mean+3σ (cố định)
        'P95G': seg_f1_gated(preds, gts, None, None, p95g, 'pixel'),       # p95×gain kiểu SuperADD (cố định)
        'GATE': seg_f1_gated(preds, gts, sp, img_thr, thr, 'pixel'),       # gate + mean+3σ
        'GATE_OTSU': seg_f1_gated(preds, gts, sp, img_thr, thr, 'otsu'),   # gate + Otsu/ảnh (tự-hiệu-chỉnh)
        'GATE_ACAP': seg_f1_gated(preds, gts, sp, img_thr, thr, 'acap', cap=cap),
        'SELFNORM': seg_f1_gated(preds, gts, sp, sn_ithr, sn_pthr, 'acap', cap=cap),  # gate+floor tự-căn/tập test
    }


def run_config(bb, layers, gk, device, args, tiles, morph, p):
    """Đo 5 operating-point trên clean (+shift nếu --sim_shift) -> SegF1 + |Δrel| robustness."""
    a = SimpleNamespace(**vars(args))
    a.tiles = tiles
    rng = np.random.default_rng(args.seed)
    srng = np.random.default_rng(args.seed + 777)
    er = args.eval_res
    per_cat = []

    for cat in args.categories:
        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'    [{cat}] không có train/good -> bỏ'); continue
        R = args.grid_tile * bb.patch
        bank = build_bank(bb, tr, tiles, R, args.grid_tile, layers, args.enc_batch, args.bank_size, device)

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        shot_pool = bad[:args.shots]           # train head (giống submit) -> loại khỏi eval
        eval_bad = bad[args.shots:]
        if args.max_eval:
            eval_bad = eval_bad[:args.max_eval]
            good = good[:args.max_eval]
        eval_idx = eval_bad + good
        head = build_head(bb, ds, shot_pool, bank, a, layers, device)

        # ---- CLEAN condition ----
        preds, gts, sp, fuse = _cond_maps(bb, ds, eval_idx, bank, head, a, layers, gk, morph, er,
                                          device, srng, args.shift_lo, args.shift_hi, do_shift=False)

        # ---- ngưỡng CỐ ĐỊNH calib trên clean validation/good (mean+3σ, p95×gain, gate) ----
        val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e))
                               for e in IMG_EXT], []))
        if args.max_val and len(val_imgs) > args.max_val:
            val_imgs = val_imgs[:args.max_val]
        val_scored = [score_grid(bb, Image.open(v), bank, head, a, layers, device)
                      for v in tqdm(val_imgs, ncols=70, desc=f'    {cat}/val', leave=False)]
        if not val_scored:
            p(f'    [{cat}] không có validation/good -> bỏ'); continue
        val_maps = [morph_close(up_to(fuse(d, pr), (er, er), gk, device), morph) for d, pr in val_scored]
        vpix = np.concatenate([m.reshape(-1) for m in val_maps])
        thr = float(vpix.mean() + args.thr_sigma * vpix.std())
        p95g = float(np.percentile(vpix, 95) * args.p95_gain)                 # SuperADD-style
        vimg = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * 0.01))].mean() for m in val_maps])
        img_thr = float(vimg.mean() + args.gate_k * vimg.std())

        # AUPRO (clean, threshold-free -> báo tham chiếu)
        gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
        spj = sp + (np.random.default_rng(0).normal(0, 1e-6, sp.shape) if float(sp.max() - sp.min()) < 1e-9 else 0.0)
        aupro05 = ader_evaluator(np.stack(preds), spj, np.stack(gts), gt_sp, use_metrics=METRIC_NAMES)[
            METRIC_NAMES.index('AUPRO0.05')]

        clean_m = _methods_for(preds, gts, sp, thr, p95g, img_thr, args.area_cap,
                               pix_k=args.thr_sigma, gate_k=args.gate_k)
        shift_m = None
        if args.sim_shift:
            spd, gtd, sps, _ = _cond_maps(bb, ds, eval_idx, bank, head, a, layers, gk, morph, er,
                                          device, srng, args.shift_lo, args.shift_hi, do_shift=True)
            # GATE*/BASE dùng ngưỡng CỐ ĐỊNH clean; SELFNORM tự tính lại từ tập shift -> so trực tiếp
            shift_m = _methods_for(spd, gtd, sps, thr, p95g, img_thr, args.area_cap,
                                   pix_k=args.thr_sigma, gate_k=args.gate_k)

        per_cat.append({'cat': cat, 'aupro05': aupro05, 'oracle': seg_f1_oracle(preds, gts),
                        'clean': clean_m, 'shift': shift_m})
        line = f'    [{cat}] AUPRO05={aupro05:.4f} | ' + ' '.join(f'{k}={clean_m[k][0]:.3f}' for k in METHODS)
        if shift_m:
            line += ' || shift: ' + ' '.join(f'{k}={shift_m[k][0]:.3f}' for k in METHODS)
        p(line)

    if not per_cat:
        return None
    out = {'aupro05': float(np.mean([c['aupro05'] for c in per_cat])),
           'oracle': float(np.mean([c['oracle'] for c in per_cat])), 'sim_shift': args.sim_shift}
    for mn in METHODS:
        cvals = np.array([c['clean'][mn] for c in per_cat]).mean(0)
        entry = {'clean': float(cvals[0]), 'clean_P': float(cvals[1]), 'clean_R': float(cvals[2])}
        if args.sim_shift:
            svals = np.array([c['shift'][mn] for c in per_cat]).mean(0)
            entry['shift'] = float(svals[0])
            entry['drel'] = (entry['clean'] - entry['shift']) / entry['clean'] if entry['clean'] > 1e-9 else 0.0
        out[mn] = entry
    return out


def main():
    ap = argparse.ArgumentParser('Ablation resolution -> SegF1@mean+3σ + AUPRO0.05 (public)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles_list', type=int, nargs='+', default=[2])   # resolution đã đóng -> giữ 2
    ap.add_argument('--morph', type=int, nargs='+', default=[0])
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
    ap.add_argument('--thr_sigma', type=float, default=3.0)
    ap.add_argument('--gate_k', type=float, default=3.0, help='image-gate: ảnh có score < mean+k*std(val) -> mask rỗng')
    ap.add_argument('--selfcal_p', type=float, default=99.0, help='GATE_SC: ngưỡng = percentile-p cố định/ảnh')
    ap.add_argument('--area_cap', type=float, default=5.0, help='GATE_ACAP: chỉ nâng ngưỡng khi BASE phủ > cap%% diện tích')
    ap.add_argument('--p95_gain', type=float, default=1.4, help='P95G (SuperADD-style): ngưỡng = p95(train)×gain')
    ap.add_argument('--sim_shift', action='store_true', help='BẬT để đo |Δrel| robustness (eval thêm bản shift mô phỏng)')
    ap.add_argument('--shift_lo', type=float, default=0.3, help='mức shift per-image ~ U[lo,hi]')
    ap.add_argument('--shift_hi', type=float, default=1.2)
    ap.add_argument('--eval_res', type=int, default=256, help='256=nhanh (quét config); 512=gần native, thật hơn')
    ap.add_argument('--max_val', type=int, default=0)
    ap.add_argument('--max_eval', type=int, default=0, help='0=hết; >0 giới hạn #defect & #good/category để chạy nhanh')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./ablation_segf1')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('segf1res', args.out_dir).info
    torch.manual_seed(args.seed)

    bb = load_backbone(args.model, device)
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    p('=' * 104)
    p(f'ROBUSTNESS TEST operating-point (public{"+shift mô phỏng" if args.sim_shift else ""}) | model={args.model} '
      f'tiles={args.tiles_list} eval_res={args.eval_res} p95_gain={args.p95_gain} '
      f'gate_k={args.gate_k} area_cap={args.area_cap} max_eval={args.max_eval or "all"}')
    p('CỐ ĐỊNH (calib clean): BASE=mean+3σ, P95G=p95×gain (kiểu SuperADD). '
      'TỰ-HIỆU-CHỈNH/ảnh: GATE_OTSU, GATE_ACAP. Novelty test = |Δrel| của ta < của ngưỡng cố định?')
    p('=' * 104)

    rows = []
    for tiles in args.tiles_list:
        for morph in args.morph:
            p(f'\n>>> tiles={tiles} (eff_grid={tiles*args.grid_tile}) morph={morph}')
            r = run_config(bb, layers, gk, device, args, tiles, morph, p)
            if r is not None:
                rows.append((tiles, r))

    p('\n' + '=' * 104)
    if args.sim_shift:
        p('{:<12}{:>10}{:>10}{:>10}'.format('method', 'SegF1clean', 'SegF1shift', '|Δrel|%'))
        for t, r in rows:
            p(f'  -- tiles={t} (AUPRO05 clean={r["aupro05"]:.4f}, oracle={r["oracle"]:.4f}) --')
            for mn in METHODS:
                e = r[mn]
                p('{:<12}{:>10.4f}{:>10.4f}{:>10.1f}'.format(mn, e['clean'], e['shift'], 100 * e['drel']))
    else:
        p('{:<12}{:>11}{:>8}{:>8}'.format('method', 'SegF1clean', 'Prec', 'Rec'))
        for t, r in rows:
            p(f'  -- tiles={t} (AUPRO05={r["aupro05"]:.4f}, oracle={r["oracle"]:.4f}) --')
            for mn in METHODS:
                e = r[mn]
                p('{:<12}{:>11.4f}{:>8.3f}{:>8.3f}'.format(mn, e['clean'], e['clean_P'], e['clean_R']))
    csv = os.path.join(args.out_dir, 'operating_point_segf1.csv')
    with open(csv, 'w') as f:
        f.write('tiles,method,SegF1_clean,SegF1_shift,drel,Prec,Rec\n')
        for t, r in rows:
            for mn in METHODS:
                e = r[mn]
                sh = e.get('shift', ''); dr = e.get('drel', '')
                f.write(f'{t},{mn},{e["clean"]:.4f},{sh if sh == "" else f"{sh:.4f}"},'
                        f'{dr if dr == "" else f"{dr:.4f}"},{e["clean_P"]:.4f},{e["clean_R"]:.4f}\n')
    p(f'\nĐã lưu: {csv}')
    if args.sim_shift:
        p('ĐỌC NOVELTY: nếu |Δrel| của GATE_OTSU/GATE_ACAP NHỎ HƠN RÕ so BASE/P95G (ngưỡng cố định) '
          '-> operating-point tự-hiệu-chỉnh robust hơn dưới shift = luận đề đứng, đáng đầu tư hi-res.')
    else:
        p('ĐỌC: thêm --sim_shift để đo |Δrel| (bài test sống-còn cho novelty). Chạy eval_res 512 khớp server.')


if __name__ == '__main__':
    main()
