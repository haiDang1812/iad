# infer_submit_photproj.py
# -----------------------------------------------------------------------------
# SUBMISSION với method novel "Photometric-invariant feature distance".
# = infer_submit_mvtec_ad2.py NGUYÊN VẸN, chỉ THÊM: bọc backbone để extract() trả
#   feature ĐÃ CHIẾU BỎ nuisance subspace (hướng feature trôi dưới đổi sáng, hạng-thấp).
#   => build_bank / build_head / score_grid / fuse / threshold / tiff GIỮ NGUYÊN từng
#      dòng, chỉ khác KHÔNG GIAN feature => quy mọi thay đổi private cho đúng projection.
#
# Bằng chứng (không submit): diag24 (patch-AUROC vial 0.30->0.89) + eval_photometric_proj
#   (AUPRO0.05 sim-shift MEAN raw 0.143 -> k2 0.280, clean gần như nguyên). k_sub=2 tốt nhất.
#
# Nuisance basis / category: SVD hướng dịch chuyển feature train/good dưới photometric-aug.
#
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_photproj.py \
#     --data_path ../data --model v3_large --tiles 2 --grid_tile 28 --shots 10 \
#     --head_w 0.6 --loss softpro --k_sub 2 --out_dir ./submit_photproj
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py <out_dir>
# -----------------------------------------------------------------------------
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
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
# import NGUYÊN pipeline production -> parity từng dòng
from infer_submit_mvtec_ad2 import (                      # noqa: E402
    build_bank, build_head, score_grid, up_to, gt_grid, img_featgrid,
    save_tiff_f16, save_png_binary, list_split_files,
    SPLITS, OBJECT_FILE_COUNTER, SMOOTH_RES, VALID, IMG_EXT,
)
from dataset import MVTecAD2Dataset                        # noqa: E402
from utils import get_gaussian_kernel, get_logger         # noqa: E402
from backbones_ext import load_backbone                    # noqa: E402

warnings.filterwarnings('ignore')


def photometric_shift(pil, s):
    """Khớp diag22/eval: brightness/contrast/color/gamma mức s>=0."""
    if s <= 0:
        return pil.convert('RGB')
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


class ProjBackbone:
    """Bọc backbone: extract() trả feature đã CHIẾU BỎ nuisance subspace U (C x k trực chuẩn).
    Mọi thứ khác (patch, n_layers, ...) uỷ quyền cho backbone gốc -> pipeline không đổi."""
    def __init__(self, bb, U):
        self._bb = bb
        self.U = U                      # C x k (trên device, trực chuẩn)

    def __getattr__(self, name):
        return getattr(self._bb, name)  # patch, n_layers, device, model, ...

    @torch.no_grad()
    def extract(self, imgs, layers):
        f = self._bb.extract(imgs, layers)          # [B, N, C]
        return f - (f @ self.U) @ self.U.t()        # chiếu bỏ U trên chiều C


@torch.no_grad()
def nuisance_basis(bb, tr_disp, T, R, gt, layers, eb, shift_levels, patch_per_img, k, device, rng):
    """SVD (KHÔNG center) hướng dịch chuyển feature train/good dưới photometric-aug -> U (C x k)."""
    disp = []
    for pth in tqdm(tr_disp, ncols=80, desc='    basis'):
        pil0 = Image.open(pth).convert('RGB')
        F0 = img_featgrid(bb, pil0, T, R, gt, layers, eb)
        C = F0.shape[-1]; F0 = F0.reshape(-1, C)
        sel = torch.from_numpy(rng.choice(F0.shape[0], size=min(patch_per_img, F0.shape[0]),
                                          replace=False)).to(device)
        F0s = F0[sel]
        for s in shift_levels:
            Fs = img_featgrid(bb, photometric_shift(pil0, s), T, R, gt, layers, eb).reshape(-1, C)[sel]
            disp.append((Fs - F0s).cpu())
    D = torch.cat(disp, 0).to(device)
    A = (D.t() @ D) / D.shape[0]
    evals, evecs = torch.linalg.eigh(A)
    ev = (torch.clamp(evals.flip(0), min=0.0)).cpu().numpy()
    U = evecs.flip(1)[:, :k].contiguous()
    return U, ev


def main():
    ap = argparse.ArgumentParser('MVTec AD 2 submission + photometric-invariant projection')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
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
    ap.add_argument('--max_val', type=int, default=0)
    ap.add_argument('--no_thresholded', action='store_true')
    ap.add_argument('--tiff_compression', type=str, default='zlib')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anomaly_dirname', type=str, default='anomaly_images')
    ap.add_argument('--thresh_dirname', type=str, default='anomaly_images_thresholded')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./submit_photproj')
    # ---- projection (novel) ----
    ap.add_argument('--k_sub', type=int, default=2, help='số chiều nuisance bỏ (0 = tắt = production)')
    ap.add_argument('--n_disp', type=int, default=16, help='ảnh train dựng nuisance basis')
    ap.add_argument('--patch_per_img', type=int, default=400)
    ap.add_argument('--shift_levels', type=float, nargs='+', default=[0.3, 0.6, 0.9, 1.2])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_photproj', args.out_dir).info
    torch.manual_seed(args.seed)                       # KHỚP production (Head init)

    bb_raw = load_backbone(args.model, device)
    R = args.grid_tile * bb_raw.patch
    if args.layers_fixed or not bb_raw.n_layers:
        layers = [ll for ll in args.layers if ll < (bb_raw.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb_raw.n_layers - 1, round(ll / 12 * bb_raw.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile; hw = args.head_w
    rng = np.random.default_rng(args.seed)             # KHỚP production (chọn shots)
    brng = np.random.default_rng(args.seed + 13)       # subsample patch dựng basis
    p('=' * 88)
    p(f'SUBMIT+PHOTPROJ | model={args.model} eff_grid={T*gt} layers={layers} | loss={args.loss} '
      f'shots={args.shots} head_w={hw} | k_sub={args.k_sub} | seg={"off" if args.no_thresholded else f"mean+{args.thr_sigma}std"}')
    p('=' * 88)

    for cat in args.categories:
        if cat not in OBJECT_FILE_COUNTER:
            p(f'  [skip] {cat}: không phải object AD2'); continue
        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ'); continue

        # ---- dựng nuisance basis (feature RAW) rồi bọc backbone ----
        if args.k_sub > 0:
            U, ev = nuisance_basis(bb_raw, tr[:args.n_disp], T, R, gt, layers, args.enc_batch,
                                   args.shift_levels, args.patch_per_img, args.k_sub, device, brng)
            bb = ProjBackbone(bb_raw, U)
            p(f'  [{cat}] nuisance basis k={args.k_sub} explvar={np.cumsum(ev)[args.k_sub-1]:.3f} -> PROJECTED')
        else:
            bb = bb_raw
            p(f'  [{cat}] k_sub=0 -> DISTANCE production (no projection)')

        # ===== từ đây: Y HỆT infer_submit_mvtec_ad2, chỉ bb đã projected =====
        bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng.shuffle(bad)
        shot_pool = bad[:args.shots]
        head = build_head(bb, ds, shot_pool, bank, args, layers, device)
        if head is None:
            p(f'  [{cat}] thiếu defect region ở shots -> fallback DISTANCE-ONLY')
        else:
            p(f'  [{cat}] head={args.loss} từ {len(shot_pool)} shot test_public')

        val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e)) for e in IMG_EXT], []))
        if args.max_val and len(val_imgs) > args.max_val:
            val_imgs = val_imgs[:args.max_val]
        val_pairs = [score_grid(bb, Image.open(v), bank, head, args, layers, device)
                     for v in tqdm(val_imgs, ncols=80, desc=f'  {cat}/val')] if val_imgs else []

        split_recs = {}; pooled_dist = []
        for split in SPLITS:
            files, root = list_split_files(args.data_path, cat, split)
            if files is None:
                p(f'  [{cat}/{split}] không tồn tại -> bỏ'); continue
            exp = OBJECT_FILE_COUNTER[cat]
            if len(files) != exp:
                p(f'  [{cat}/{split}] CẢNH BÁO: {len(files)} ảnh (checker cần {exp})')
            recs = []
            for fp in tqdm(files, ncols=80, desc=f'  {cat}/{split}'):
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

        thr = None
        if not args.no_thresholded and val_pairs:
            vs = np.concatenate([up_to(fuse(d, pr), (SMOOTH_RES, SMOOTH_RES), gk, device).reshape(-1)
                                 for d, pr in val_pairs])
            thr = float(vs.mean() + args.thr_sigma * vs.std())
        p(f'  [{cat}] dist_lo={lo:.3f} dist_hi={hi:.3f}' + (f' thr={thr:.4f}' if thr is not None else ' (no thr)'))

        for split, (recs, root) in split_recs.items():
            for fp, Himg, W, d, pr in recs:
                amap = up_to(fuse(d, pr), (Himg, W), gk, device)
                stem = os.path.splitext(os.path.basename(fp))[0]
                save_tiff_f16(amap, os.path.join(args.out_dir, args.anomaly_dirname, cat, split, stem + '.tiff'),
                              args.tiff_compression)
                if thr is not None:
                    save_png_binary(amap > thr, os.path.join(args.out_dir, args.thresh_dirname, cat, split, stem + '.png'))
            p(f'  [{cat}/{split}] lưu {len(recs)} tiff' + ('' if thr is None else ' + png'))

    p('\n' + '=' * 88)
    p(f'XONG: {args.out_dir}')
    p('KIỂM TRA: cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py '
      f'{os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
