# infer_submit_uniform.py
# -----------------------------------------------------------------------------
# BẢN NỘP SẠCH (method paper) — 1 config ĐỒNG NHẤT cho cả 8 cat, KHÔNG per-cat,
#   KHÔNG threshold trick (bỏ hết fbrate/rate/gate/all-bads/a1b0 của v1/v2/v3).
#
#   - Backbone/bank/head: y hệt (v3_large, enc_batch 16, max_train 60, seed 0,
#     head_w 0.6, canvas 256 + gaussian(5,4), softpro).
#   - LUẬT per-METRIC (contribution 2, từ ablation uniform 2026-07-22, KHÔNG per-cat):
#       tiff (AUPRO) = NRS head  (thắng AUPRO 8/8, +0.069 mean)
#       png  (SegF1) = GRID head (thắng F1 7/8, mean 0.599 vs NRS 0.504)
#     = "metric đích quyết định độ phân giải supervision". Đổi bằng --png_head.
#   - Grid: 1 giá trị cố định (--grid_tile, mặc định 48) cho MỌI cat.
#   - Head 10-shot build đúng thứ tự seed eval_nrs_head (manual_seed->grid->nrs)
#     -> head TRÙNG bảng ablation.
#   - tiff (AUPRO): map fused NRS thô.
#   - png (SegF1): ngưỡng mu+kσ pooled trên private.
#       * mặc định k=4.5 (ksig) — hằng số.
#       * --support_k: HỌC k TỪ support GT (10 shot) — sạch, prevalence-immune,
#         scale-invariant (xem eval_fsthresh.py). rule f1 (max support-F1) | med
#         (median z-score defect). Đây là contribution "operating-point từ few-shot".
#
#   => Đây là con số method THẬT để báo cáo + nộp lấy điểm leaderboard sạch.
#      Không có lựa chọn nào peek vào test/private. Cùng bộ tham số này chạy được
#      trên VisA / MVTec-AD để chứng minh tổng quát hoá.
#
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_uniform.py \
#       --data_path ../data --out_dir ./submit_uniform            # ksig thuần
#   ... --support_k                                               # + operating-point học
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../submit_uniform
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

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, save_tiff_f16, save_png_binary,
    list_split_files, SPLITS, OBJECT_FILE_COUNTER, IMG_EXT,
)
from eval_nrs_head import build_nrs_head                                # noqa: E402
from infer_submit_nrs import native_map                                 # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
CATS = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
K_GRID = np.arange(1.0, 6.01, 0.25)


def ksig_threshold(maps_iter_factory, k):
    """mu + k·σ trên pooled native maps (streaming float64)."""
    n, s1, s2 = 0, 0.0, 0.0
    for m in maps_iter_factory():
        md = m.double()
        n += m.numel()
        s1 += float(md.sum())
        s2 += float((md * md).sum())
    mu = s1 / max(n, 1)
    var = max(s2 / max(n, 1) - mu * mu, 0.0)
    return mu + k * var ** 0.5


def calibrate_k(bb, ds, shots, head, bank, args, T, R, gt_, layers, gk, device, hw, rule, p):
    """Học k = số σ trên trung-bình-normal mà ngưỡng nên đặt, TỪ support GT.
    rule='f1': k tối đa pooled-F1 trên support; 'med': median z-score pixel defect."""
    C = bank.shape[-1]
    grids, sizes = [], []
    with torch.no_grad():
        for i in shots:
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]
            d = np.asarray(nn_map(g, bank, device))
            pr = torch.sigmoid(head(g.reshape(-1, C))).reshape(G, G).cpu().numpy()
            grids.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])
    ms, gs = [], []
    s1 = s2 = 0.0
    npix = 0
    for (d, pr), (H, W), i in zip(grids, sizes, shots):
        s = ((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32)
        with torch.no_grad():
            m = native_map(s, gk, H, W, device).cpu().numpy()
        gm = np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127
        ms.append(m)
        gs.append(gm)
        s1 += float(m.sum(dtype=np.float64))
        s2 += float(np.square(m, dtype=np.float64).sum())
        npix += m.size
    mu = s1 / max(npix, 1)
    sd = float(np.sqrt(max(s2 / max(npix, 1) - mu * mu, 0.0)))
    if rule == 'med':
        zd = np.concatenate([((ms[j][gs[j]]) - mu) / (sd + 1e-9) for j in range(len(ms)) if gs[j].any()]) \
            if any(g.any() for g in gs) else np.array([4.5])
        k = float(np.median(zd))
    else:
        k, best = 4.5, -1.0
        for kk in K_GRID:
            tau = mu + kk * sd
            tp = fp = fn = 0.0
            for m, gm in zip(ms, gs):
                pred = m > tau
                tp += float(np.count_nonzero(pred & gm))
                fp += float(np.count_nonzero(pred & ~gm))
                fn += float(np.count_nonzero(~pred & gm))
            f = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if f > best:
                best, k = f, float(kk)
    p(f'    [support-k rule={rule}] k={k:.2f} (baseline 4.5)')
    return k


def main():
    ap = argparse.ArgumentParser('submission SẠCH uniform: NRS head 1 config, ksig hoặc support-k')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--grid_tile', type=int, default=48, help='1 giá trị cho MỌI cat (uniform)')
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
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
    ap.add_argument('--png_head', type=str, default='grid', choices=['grid', 'nrs'],
                    help='head cho png/SegF1 (mặc định grid = thắng F1 7/8); tiff luôn NRS')
    ap.add_argument('--k', type=float, default=4.5, help='k ksig khi KHÔNG dùng --support_k')
    ap.add_argument('--support_k', action='store_true', help='học k ngưỡng từ support GT (thay k cố định)')
    ap.add_argument('--support_k_rule', type=str, default='f1', choices=['f1', 'med'])
    ap.add_argument('--no_thresholded', action='store_true')
    ap.add_argument('--tiff_compression', type=str, default='zlib')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anomaly_dirname', type=str, default='anomaly_images')
    ap.add_argument('--thresh_dirname', type=str, default='anomaly_images_thresholded')
    ap.add_argument('--categories', type=str, nargs='+', default=CATS)
    ap.add_argument('--out_dir', type=str, default='./submit_uniform')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_uniform', args.out_dir).info
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    thr_mode = f'support_k({args.support_k_rule})' if args.support_k else f'ksig(k={args.k})'
    p('=' * 92)
    p(f'SUBMIT UNIFORM (sạch) | model={args.model} grid={T}:{gt_} shots={args.shots} '
      f'head_w={hw} | head=NRS mọi cat | tiff=png-head | ngưỡng png={thr_mode}')
    p('=' * 92)

    for cat in args.categories:
        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ'); continue
        if args.max_train:
            tr = tr[:args.max_train]
        p(f'  [{cat}] grid={T}:{gt_} | bank từ {len(tr)} ảnh...')
        bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
        C = bank.shape[-1]

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(bad)
        shots10 = bad[:args.shots]
        # thứ tự seed Y HỆT eval_nrs_head (manual_seed -> grid -> nrs) -> head trùng ablation
        torch.manual_seed(args.seed)
        head_grid = build_head(bb, ds, shots10, bank, args, layers, device)
        head_nrs = build_nrs_head(bb, ds, shots10, bank, args, layers, device, p)
        if head_nrs is None:
            p(f'  [{cat}] NRS head None -> bỏ (không nộp thiếu cat!)'); continue
        head_tiff = head_nrs                                              # AUPRO = NRS
        head_png = head_grid if (args.png_head == 'grid' and head_grid is not None) else head_nrs
        if args.png_head == 'grid' and head_grid is None:
            p(f'  [{cat}] grid head None (defect dưới-ô) -> png rơi về NRS')

        # k ngưỡng: cố định hoặc học từ support (chỉ dùng khi có threshold)
        k_use = args.k
        if args.support_k and not args.no_thresholded:
            k_use = calibrate_k(bb, ds, shots10, head_png, bank, args, T, R, gt_, layers, gk, device,
                                hw, args.support_k_rule, p)

        # ---- encode PRIVATE 1 lần ----
        split_recs = {}
        pooled_dist = []
        with torch.no_grad():
            for split in SPLITS:
                files, _ = list_split_files(args.data_path, cat, split)
                if files is None:
                    p(f'  [{cat}/{split}] không tồn tại -> bỏ'); continue
                exp = OBJECT_FILE_COUNTER[cat]
                if len(files) != exp:
                    p(f'  [{cat}/{split}] CẢNH BÁO: {len(files)} ảnh (checker cần {exp})')
                recs = []
                for fp in tqdm(files, ncols=80, desc=f'  {cat}/{split}'):
                    pil = Image.open(fp)
                    W, H = pil.size
                    g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
                    G = g.shape[0]
                    d = np.asarray(nn_map(g, bank, device))
                    flat = g.reshape(-1, C)
                    prF = torch.sigmoid(head_png(flat)).reshape(G, G).cpu().numpy()     # png/SegF1
                    prA = torch.sigmoid(head_tiff(flat)).reshape(G, G).cpu().numpy()    # tiff/AUPRO
                    recs.append((fp, H, W, d, prF, prA))
                    pooled_dist.append(d.reshape(-1))
                split_recs[split] = recs
        if not pooled_dist:
            p(f'  [{cat}] không có ảnh private -> bỏ'); continue
        all_d = np.concatenate(pooled_dist)
        lo, hi = np.percentile(all_d, 1), np.percentile(all_d, 99)

        def fuse(d, pr):
            return ((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32)

        thr = None
        if not args.no_thresholded:
            def maps_iter():
                for recs in split_recs.values():
                    for _, H, W, d, prF, _ in recs:
                        with torch.no_grad():
                            yield native_map(fuse(d, prF), gk, H, W, device)
            thr = ksig_threshold(maps_iter, k_use)
            p(f'  [{cat}] dist_lo={lo:.3f} dist_hi={hi:.3f} png_head={args.png_head} '
              f'k={k_use:.2f} thr={thr:.5f}')

        # ---- PASS 2: tiff = NRS fused map, png = grid fused > thr ----
        for split, recs in split_recs.items():
            for fp, H, W, d, prF, prA in recs:
                stem = os.path.splitext(os.path.basename(fp))[0]
                with torch.no_grad():
                    mA = native_map(fuse(d, prA), gk, H, W, device).cpu().numpy()
                save_tiff_f16(mA, os.path.join(args.out_dir, args.anomaly_dirname, cat, split, stem + '.tiff'),
                              args.tiff_compression)
                if thr is not None:
                    with torch.no_grad():
                        mF = native_map(fuse(d, prF), gk, H, W, device).cpu().numpy()
                    save_png_binary(mF > thr,
                                    os.path.join(args.out_dir, args.thresh_dirname, cat, split, stem + '.png'))
            p(f'  [{cat}/{split}] lưu {len(recs)} tiff' + ('' if thr is None else ' + png'))

    p('\n' + '=' * 92)
    p(f'XONG: {args.out_dir}')
    p('KIỂM TRA: cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py '
      f'{os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
