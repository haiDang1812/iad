# infer_submit_nrs.py
# -----------------------------------------------------------------------------
# Submission "operating-point alignment" đầy đủ — kết tinh của native348/nrs24/nrs48:
#
#   1) PER-CAT CONFIG (res là đòn bẩy per-cat, không universal):
#        3:48 cho can/fabric/sheet_metal/wallplugs/walnuts, 3:24 cho phần còn lại.
#   2) HAI HEAD, HAI METRIC (NRS premise test, 8/8 cat):
#        .tiff (chấm AUPRO0.05)  <- fused head NRS (supervision native, cân region
#                                   = hình dáng AUPRO; thắng 8/8 cat, mean 0.861@512)
#        .png  (chấm SegF1)      <- fused head thắng-F1 per-cat (can/wallplugs: NRS
#                                   defect dưới-ô; 6 cat còn lại: grid head)
#      Server chấm 2 artifact độc lập — mỗi metric nhận map sinh từ đúng cây thước
#      supervision của nó.
#   3) RATE RULE cho ngưỡng png (R3 ~ oracle 7/7 public): r_cat = tỉ lệ pixel defect
#      trên test_public (GT native) -> ngưỡng = quantile sao cho tỉ lệ pixel vượt
#      ngưỡng trên PHÂN BỐ PRIVATE (pooled 2 split, native, streaming 2-stage hist)
#      đúng bằng r_cat. Tự hiệu chỉnh theo shift (diag27/29), không cần GT đích.
#
# KHỚP EVAL (mọi con số public đều đo với setup này — đổi là số mất giá trị):
#   enc_batch=16, max_train=60 (bank), shots=10 seed=0 (rng shuffle bad test_public),
#   torch.manual_seed trước build head (init trùng), head_w=0.6, canvas 256 + gaussian
#   (5,4) rồi bung native, loss softpro.
#
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_nrs.py \
#       --data_path ../data --out_dir ./submit_nrs
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../submit_nrs
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
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
NBINS = 8192

# per-cat: config res + head cho nhánh F1 (.png). Head cho nhánh AUPRO (.tiff) LUÔN là NRS.
# Nguồn số: nrs24/nrs48/nrs48b/nrs24rv + native324/native348 (F1@ksig thước native, full split).
PER_CAT = {
    'can':         dict(tiles=3, grid_tile=48, f1_head='nrs'),    # F1 0.3055 / AUPRO 0.7440
    'fabric':      dict(tiles=3, grid_tile=48, f1_head='grid'),   # F1 0.8629 / AUPRO 0.9576
    'fruit_jelly': dict(tiles=3, grid_tile=24, f1_head='grid'),   # F1 0.7832 / AUPRO 0.7636
    'rice':        dict(tiles=3, grid_tile=24, f1_head='grid'),   # F1 0.5928 / AUPRO 0.9217
    'sheet_metal': dict(tiles=3, grid_tile=48, f1_head='grid'),   # F1 0.5500 / AUPRO 0.7960
    'vial':        dict(tiles=3, grid_tile=24, f1_head='grid'),   # F1 0.4372 / AUPRO 0.8632
    'wallplugs':   dict(tiles=3, grid_tile=48, f1_head='nrs'),    # F1 0.5531 / AUPRO 0.8271
    'walnuts':     dict(tiles=3, grid_tile=48, f1_head='grid'),   # F1 0.8146 / AUPRO 0.9158
}


def native_map(s_grid, gk, H, W, device):
    """grid -> 256 -> gaussian(5,4) -> native. Y HỆT đường eval (eval_native/eval_nrs_head)."""
    t = torch.tensor(s_grid, device=device)[None, None].float()
    t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
    t = gk(t)
    return F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0]


def public_gt_rate(ds):
    """r_cat = pixel defect / tổng pixel trên TOÀN test_public (GT native, cả ảnh good).
    Đây là exceedance-rate mục tiêu của rate rule (R3 ~ oracle 7/7 trên public)."""
    pos, tot = 0, 0
    for i in range(len(ds.img_paths)):
        w, h = Image.open(ds.img_paths[i]).size          # đọc header, không decode
        tot += w * h
        if ds.labels[i] == 1 and isinstance(ds.gt_paths[i], str) and os.path.exists(ds.gt_paths[i]):
            pos += int((np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).sum())
    return pos / max(tot, 1)


def rate_threshold(maps_iter_factory, r, gmin, gmax, device):
    """Ngưỡng t sao cho P(score > t) = r trên pooled native maps — streaming 2-stage
    histogram (không giữ nổi ~3e9 pixel trong RAM). Stage 1 khoanh bin, stage 2 zoom
    vào bin đó -> sai số ~ (range/8192^2), thừa cho ngưỡng."""
    lo, hi = float(gmin), float(gmax)
    for _ in range(2):
        cnt = torch.zeros(NBINS, dtype=torch.float64, device=device)
        n_tot, n_above_hi = 0, 0
        for m in maps_iter_factory():
            n_tot += m.numel()
            n_above_hi += int((m > hi).sum())
            cnt += torch.histc(m.clamp(lo, hi), bins=NBINS, min=lo, max=hi).double()
        k = r * n_tot - n_above_hi                        # số pixel cần vượt ngưỡng TRONG [lo,hi]
        if k <= 0:
            return hi
        cum = torch.flip(torch.cumsum(torch.flip(cnt, [0]), 0), [0])  # cum[i] = #pixel ở bin >= i
        idx = int(torch.searchsorted(-cum, -torch.tensor(float(k), dtype=torch.float64, device=device)))
        idx = max(0, min(NBINS - 1, idx))
        wd = (hi - lo) / NBINS
        lo, hi = lo + idx * wd, lo + (idx + 1) * wd       # zoom vào bin chứa ngưỡng
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser('submission: per-cat config + dual-head (tiff=NRS, png=best-F1) + rate rule')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16, help='PHẢI = 16 (mọi eval đo với 16; đổi là đổi bank)')
    ap.add_argument('--max_train', type=int, default=60, help='PHẢI = 60 (bank của mọi eval)')
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
    ap.add_argument('--rate_scale', type=float, default=1.0,
                    help='nhân r_cat (an toàn: <1 = ngưỡng cao/precision hơn nếu sợ private ít defect hơn)')
    ap.add_argument('--no_thresholded', action='store_true')
    ap.add_argument('--tiff_compression', type=str, default='zlib')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anomaly_dirname', type=str, default='anomaly_images')
    ap.add_argument('--thresh_dirname', type=str, default='anomaly_images_thresholded')
    ap.add_argument('--categories', type=str, nargs='+', default=list(PER_CAT.keys()))
    ap.add_argument('--out_dir', type=str, default='./submit_nrs')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_nrs', args.out_dir).info
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p('=' * 92)
    p(f'SUBMIT NRS | model={args.model} layers={layers} loss={args.loss} shots={args.shots} '
      f'head_w={args.head_w} enc_batch={args.enc_batch} max_train={args.max_train} '
      f'rate_scale={args.rate_scale} | tiff=NRS-head  png=best-F1-head @ rate rule')
    p('=' * 92)

    for cat in args.categories:
        cfg = PER_CAT[cat]
        args.tiles, args.grid_tile = cfg['tiles'], cfg['grid_tile']    # build_head/build_nrs_head đọc từ args
        T, gt_ = args.tiles, args.grid_tile
        R = gt_ * bb.patch
        hw = args.head_w

        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ'); continue
        if args.max_train:
            tr = tr[:args.max_train]
        p(f'  [{cat}] cfg={T}:{gt_} f1_head={cfg["f1_head"]} | bank từ {len(tr)} ảnh...')
        bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
        C = bank.shape[-1]

        # ---- shots + 2 head (thứ tự seed Y HỆT eval_nrs_head.run_cat -> head trùng eval) ----
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(bad)
        shots = bad[:args.shots]
        torch.manual_seed(args.seed)
        head_grid = build_head(bb, ds, shots, bank, args, layers, device)
        head_nrs = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)
        if head_nrs is None:
            p(f'  [{cat}] NRS không build được -> bỏ (không nộp thiếu cat!)'); continue
        head_f1 = head_nrs if (cfg['f1_head'] == 'nrs' or head_grid is None) else head_grid
        r_cat = args.rate_scale * public_gt_rate(ds)
        p(f'  [{cat}] head_grid={"None" if head_grid is None else "OK"} head_nrs=OK '
          f'| r_cat={r_cat:.3e} (gt-rate public native x {args.rate_scale})')

        # ---- PASS 1: encode private 1 lần, giữ (d, pr_f1, pr_nrs) grid-level ----
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
                    prF = torch.sigmoid(head_f1(flat)).reshape(G, G).cpu().numpy()
                    prA = torch.sigmoid(head_nrs(flat)).reshape(G, G).cpu().numpy()
                    recs.append((fp, H, W, d, prF, prA))
                    pooled_dist.append(d.reshape(-1))
                split_recs[split] = recs
        if not pooled_dist:
            p(f'  [{cat}] không có ảnh private -> bỏ'); continue
        all_d = np.concatenate(pooled_dist)
        lo, hi = np.percentile(all_d, 1), np.percentile(all_d, 99)

        def fuse(d, pr):
            dr = (d - lo) / (hi - lo + 1e-8)
            return ((1 - hw) * dr + hw * pr).astype(np.float32)

        # ---- ngưỡng png: rate rule trên pooled NATIVE maps của nhánh F1 (2 split gộp) ----
        thr = None
        if not args.no_thresholded:
            sF = [fuse(d, prF) for recs in split_recs.values() for _, _, _, d, prF, _ in recs]
            gmin = min(float(s.min()) for s in sF) - 0.05
            gmax = max(float(s.max()) for s in sF) + 0.05

            def maps_iter():
                for recs in split_recs.values():
                    for _, H, W, d, prF, _ in recs:
                        with torch.no_grad():
                            yield native_map(fuse(d, prF), gk, H, W, device)
            thr = rate_threshold(maps_iter, r_cat, gmin, gmax, device)
            p(f'  [{cat}] dist_lo={lo:.3f} dist_hi={hi:.3f} thr(rate)={thr:.5f}')

        # ---- PASS 2: tiff = NRS fused, png = F1 fused > thr ----
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
                    save_png_binary(mF > thr, os.path.join(args.out_dir, args.thresh_dirname, cat, split, stem + '.png'))
            p(f'  [{cat}/{split}] lưu {len(recs)} tiff' + ('' if thr is None else ' + png'))

    p('\n' + '=' * 92)
    p(f'XONG: {args.out_dir}')
    p('KIỂM TRA: cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py '
      f'{os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
