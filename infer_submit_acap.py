# infer_submit_acap.py
# -----------------------------------------------------------------------------
# BIẾN THỂ SUBMIT: y hệt infer_submit_mvtec_ad2.py (import NGUYÊN core -> parity),
# CHỈ KHÁC bước tạo mask nhị phân (.png / SegF1):
#   thay ngưỡng CỐ ĐỊNH mean+3σ  ->  GATE + AREA-CAP (self-calibrating per-image):
#     (1) GATE: ảnh có image-score < img_thr(calib clean val) -> mask RỖNG (dự đoán normal).
#     (2) AREA-CAP: ảnh còn lại -> ngưỡng = max(mean+3σ, percentile_{100-cap}(map))
#         => chỉ NÂNG ngưỡng khi map phủ > cap% diện tích -> cắt over-segment do light-shift.
#
# .tiff (anomaly map) + fuse GIỮ NGUYÊN -> AUPRO private KHÔNG ĐỔI (=0.674). Chỉ SegF1 đổi.
# Động cơ: robustness test |Δrel| (2026-07-05) cho GATE_ACAP tụt dưới shift chỉ ~nửa ngưỡng
#   cố định (15.1% vs 25-27%, kể cả kiểu SuperADD p95×gain) + shift-SegF1 cao nhất. Sim PASS
#   -> validate PRIVATE THẬT (sim từng lừa ở shift-aug).
#
# Chạy (khớp lệnh submit chuẩn, chỉ đổi script + thêm --gate_k --area_cap):
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_acap.py \
#     --data_path ../mvtec_ad_2/data --model v3_large --tiles 2 --grid_tile 28 \
#     --shots 10 --head_w 0.6 --loss softpro --gate_k 3 --area_cap 5 --out_dir ./submit_acap
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../submit_acap
# -----------------------------------------------------------------------------

import os
import sys
import glob
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from dataset import MVTecAD2Dataset
from utils import get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

# import NGUYÊN core -> khớp submit chuẩn từng dòng (chỉ khác logic PNG bên dưới)
from infer_submit_mvtec_ad2 import (
    build_bank, build_head, score_grid, up_to, save_tiff_f16, save_png_binary,
    list_split_files, VALID, SPLITS, OBJECT_FILE_COUNTER, IMG_EXT, SMOOTH_RES,
)

warnings.filterwarnings("ignore")


def top1_mean(a):
    """image-level score = trung bình 1% pixel cao nhất (khớp eval_segf1_resolution)."""
    f = a.reshape(-1)
    k = max(1, int(f.size * 0.01))
    return float(np.sort(f)[::-1][:k].mean())


def acap_threshold(amap, base_thr, area_cap):
    """ngưỡng self-cal per-image: chỉ NÂNG khi base_thr phủ > cap% diện tích."""
    return max(base_thr, float(np.percentile(amap, 100.0 - area_cap)))


def main():
    ap = argparse.ArgumentParser('MVTec AD 2 submit — GATE + AREA-CAP threshold (SegF1 shift-robust)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64, help='PHẢI trùng train_softpro (64)')
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
    ap.add_argument('--thr_sigma', type=float, default=3.0, help='ngưỡng nền = mean + k*std trên validation/good')
    ap.add_argument('--gate_k', type=float, default=3.0, help='GATE: ảnh có img-score < mean+k*std(val) -> mask rỗng')
    ap.add_argument('--area_cap', type=float, default=5.0, help='AREA-CAP: chỉ nâng ngưỡng khi map phủ > cap%% diện tích')
    ap.add_argument('--max_val', type=int, default=0)
    ap.add_argument('--no_thresholded', action='store_true')
    ap.add_argument('--tiff_compression', type=str, default='zlib')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anomaly_dirname', type=str, default='anomaly_images')
    ap.add_argument('--thresh_dirname', type=str, default='anomaly_images_thresholded')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./submit_acap')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_acap', args.out_dir).info
    torch.manual_seed(args.seed)                       # KHỚP train_softpro (Head init)

    bb = load_backbone(args.model, device)
    R = args.grid_tile * bb.patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile; hw = args.head_w
    rng = np.random.default_rng(args.seed)
    p('=' * 92)
    p(f'SUBMIT-ACAP | model={args.model} eff_grid={T*gt} layers={layers} | loss={args.loss} shots={args.shots} '
      f'head_w={hw} | PNG=GATE(k={args.gate_k})+AREA-CAP(cap={args.area_cap}%) | tiff/AUPRO KHÔNG đổi')
    p('=' * 92)

    for cat in args.categories:
        if cat not in OBJECT_FILE_COUNTER:
            p(f'  [skip] {cat}: không phải object AD2'); continue
        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ'); continue
        bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng.shuffle(bad)
        shot_pool = bad[:args.shots]
        head = build_head(bb, ds, shot_pool, bank, args, layers, device)
        if head is None:
            p(f'  [{cat}] thiếu defect region -> fallback DISTANCE-ONLY')
        else:
            p(f'  [{cat}] head={args.loss} từ {len(shot_pool)} shot test_public')

        # validation/good -> ngưỡng nền (mean+3σ) + gate threshold (img-score)
        val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e))
                               for e in IMG_EXT], []))
        if args.max_val and len(val_imgs) > args.max_val:
            val_imgs = val_imgs[:args.max_val]
        val_pairs = [score_grid(bb, Image.open(v), bank, head, args, layers, device)
                     for v in tqdm(val_imgs, ncols=80, desc=f'  {cat}/val', leave=False)] if val_imgs else []

        # PASS 1: score cả 2 private split, gộp distance -> lo/hi (giống submit chuẩn)
        split_recs = {}; pooled_dist = []
        for split in SPLITS:
            files, root = list_split_files(args.data_path, cat, split)
            if files is None:
                p(f'  [{cat}/{split}] không tồn tại -> bỏ'); continue
            if len(files) != OBJECT_FILE_COUNTER[cat]:
                p(f'  [{cat}/{split}] CẢNH BÁO: {len(files)} ảnh (checker cần {OBJECT_FILE_COUNTER[cat]})')
            recs = []
            for fp in tqdm(files, ncols=80, desc=f'  {cat}/{split}', leave=False):
                pil = Image.open(fp); W, Himg = pil.size
                d, pr = score_grid(bb, pil, bank, head, args, layers, device)
                recs.append((fp, Himg, W, d, pr)); pooled_dist.append(d.reshape(-1))
            split_recs[split] = (recs, root)
        if not pooled_dist:
            p(f'  [{cat}] không có ảnh private -> bỏ'); continue
        all_dist = np.concatenate(pooled_dist)
        lo, hi = np.percentile(all_dist, 1), np.percentile(all_dist, 99)

        def fuse(d, pr):
            dr = (d - lo) / (hi - lo + 1e-8)
            return dr if (head is None or pr is None) else (1 - hw) * dr + hw * pr

        # ngưỡng nền + gate threshold từ validation/good (fused, up_to 256)
        base_thr = img_thr = None
        if not args.no_thresholded and val_pairs:
            vmaps = [up_to(fuse(d, pr), (SMOOTH_RES, SMOOTH_RES), gk, device) for d, pr in val_pairs]
            vpix = np.concatenate([m.reshape(-1) for m in vmaps])
            base_thr = float(vpix.mean() + args.thr_sigma * vpix.std())
            vimg = np.array([top1_mean(m) for m in vmaps])
            img_thr = float(vimg.mean() + args.gate_k * vimg.std())
        p(f'  [{cat}] lo={lo:.3f} hi={hi:.3f}' +
          ('' if base_thr is None else f' base_thr={base_thr:.4f} img_thr={img_thr:.4f}'))

        # PASS 2: fuse -> native -> tiff (GIỮ NGUYÊN) + png GATE+AREA-CAP
        n_gated = 0
        for split, (recs, root) in split_recs.items():
            for fp, Himg, W, d, pr in recs:
                amap = up_to(fuse(d, pr), (Himg, W), gk, device)
                stem = os.path.splitext(os.path.basename(fp))[0]
                save_tiff_f16(amap, os.path.join(args.out_dir, args.anomaly_dirname, cat, split, stem + '.tiff'),
                              args.tiff_compression)
                if base_thr is not None:
                    if top1_mean(amap) < img_thr:
                        mask = np.zeros(amap.shape, dtype=bool)      # GATE: ảnh normal -> rỗng
                        n_gated += 1
                    else:
                        mask = amap > acap_threshold(amap, base_thr, args.area_cap)   # AREA-CAP
                    save_png_binary(mask, os.path.join(args.out_dir, args.thresh_dirname, cat, split, stem + '.png'))
            p(f'  [{cat}/{split}] lưu {len(recs)} tiff' + ('' if base_thr is None else ' + png'))
        if base_thr is not None:
            p(f'  [{cat}] gated {n_gated} ảnh (dự đoán normal -> mask rỗng)')

    p('\n' + '=' * 92)
    p(f'XONG: {args.out_dir}')
    p('KIỂM TRA: cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py '
      f'{os.path.abspath(args.out_dir)}')
    p('LƯU Ý: tiff giữ nguyên -> AUPRO private KHÔNG đổi (~0.674). Chỉ SegF1 (từ png) đổi.')


if __name__ == '__main__':
    main()
