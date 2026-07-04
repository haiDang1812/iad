# infer_submit_shiftaug.py
# -----------------------------------------------------------------------------
# BẢN SHIFT-AUG của infer_submit_mvtec_ad2.py — sinh anomaly map NỘP server MVTec AD 2.
#
# NOVELTY (đã được diag22 chứng minh PASS CẢ HAI metric trên shift mô phỏng):
#   Head few-shot train trên shot SẠCH bị tụt khi gặp light-shift của test_private
#   (AUPRO0.05 -0.077, SegF1 -0.05). Augment 10 shot bằng photometric_shift khi train head
#   -> head học boundary defect BẤT BIẾN với sáng -> transfer sang private:
#     diag22 MEAN: SHIFT-NOAUG 0.7105/0.283  ->  SHIFT-AUG 0.7716/0.309  (gần trần clean 0.787/0.308).
#
# GIỐNG HỆT infer_submit_mvtec_ad2.py (copy nguyên văn, để so A/B sạch):
#   bank, chọn shots, nn_map, fuse (1-hw)*dr+hw*prob, global-norm 1/99 gộp 2 split,
#   smoothing 56->256 gaussian ->native, threshold mean+kσ trên validation/good, xuất tiff f16 + png.
#   CHỈ KHÁC build_head: thêm augment (Xpos ở nhiều mức shift + Xneg có normal-pixel đã-shift).
#
# A/B trong CÙNG file (loại trừ mọi khác biệt ngoài augment):
#   --n_aug 0  -> BASELINE (giống hệt infer_submit_mvtec_ad2.py từng byte: shifts=[clean], Xneg=bank).
#   --n_aug 3  -> METHOD shift-aug.
#   * shift_rng RIÊNG (seed+12345); n_aug=0 KHÔNG bốc shift_rng -> shots & mọi thứ y hệt baseline.
#
# Chạy (method):
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_shiftaug.py \
#     --data_path ../mvtec_ad_2/data --model v3_large --tiles 2 --grid_tile 28 \
#     --shots 10 --n_aug 3 --shift_lo 0.3 --shift_hi 1.2 --head_w 0.6 --loss softpro \
#     --out_dir ./submit_out_shiftaug
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../submit_out_shiftaug
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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageEnhance
from scipy.ndimage import label as cc_label

from dataset import MVTecAD2Dataset
from utils import get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
SPLITS = ['test_private', 'test_private_mixed']
OBJECT_FILE_COUNTER = {
    'can': 321, 'fabric': 314, 'fruit_jelly': 255, 'rice': 277,
    'sheet_metal': 142, 'vial': 276, 'wallplugs': 232, 'walnuts': 228,
}
IMG_EXT = ('*.png', '*.jpg', '*.JPG', '*.jpeg', '*.bmp')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SMOOTH_RES = 256   # độ phân giải áp gaussian, KHỚP train_softpro.region_metrics/upmap


# ===== augment: copy y nguyên từ diag22 (đã validate) =======================
def photometric_shift(pil, s):
    """Giả lập dark-field/back-light của test_private: giảm sáng + tăng contrast + gamma + bạc màu.
    s=0 -> no-op. Deterministic (tái lập được)."""
    if s <= 0:
        return pil
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


# ===== core: COPY nguyên văn từ infer_submit_mvtec_ad2.py ====================
def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    return [pil.crop((round(j * w / T), round(i * h / T), round((j + 1) * w / T), round((i + 1) * h / T)))
            for i in range(T) for j in range(T)]


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def img_featgrid(bb, pil, T, R, gt, layers, eb):
    tiles = tile_pils(pil, T)
    parts = []
    for s in range(0, len(tiles), eb):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + eb]])
        parts.append(bb.extract(b, layers))
    f = torch.cat(parts, 0)
    C = f.shape[-1]
    grid = torch.zeros(T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * gt:(i + 1) * gt, j * gt:(j + 1) * gt] = f[k, :gt * gt].reshape(gt, gt, C)
    return grid


@torch.no_grad()
def nn_map(grid, bank, device, chunk=4096):
    G = grid.shape[0]
    C = grid.shape[-1]
    q = grid.reshape(-1, C)
    out = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(1)[0]
    return out.reshape(G, G).cpu().numpy()


def gt_grid(gpath, label, G):
    if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((G, G), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


@torch.no_grad()
def build_bank(bb, tr, T, R, gt, layers, eb, bank_size, device):
    keep = max(64, bank_size * 4 // max(1, len(tr) * T * T))
    acc = []
    buf = []
    for pth in tr:
        buf.extend(tile_pils(Image.open(pth), T))
        while len(buf) >= eb:
            b = torch.stack([to_tensor(t, R) for t in buf[:eb]])
            buf = buf[eb:]
            f = bb.extract(b, layers)
            acc.append(subsample(f.reshape(-1, f.shape[-1]), eb * keep).cpu())
    if buf:
        f = bb.extract(torch.stack([to_tensor(t, R) for t in buf]), layers)
        acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
    return subsample(torch.cat(acc, 0), bank_size).to(device)


class Head(nn.Module):
    def __init__(self, C, mu, sd):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)
        self.lin = nn.Linear(C, 1)

    def forward(self, x):
        return self.lin((x - self.mu) / self.sd).squeeze(-1)


def train_bce(head, Xpos, Xneg, steps, lr, device):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    X = torch.cat([Xpos, Xneg], 0)
    y = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    w = torch.cat([torch.full((len(Xpos),), len(Xneg) / max(1, len(Xpos))), torch.ones(len(Xneg))]).to(device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(X), y, weight=w)
        loss.backward()
        opt.step()
    return head


def train_softpro(head, Xpos, region_ids, Xneg, steps, lr, q, temp, w_bce, w_fp, device):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    rids = torch.tensor(region_ids, device=device)
    uniq = torch.unique(rids)
    y = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    wbce = torch.cat([torch.full((len(Xpos),), len(Xneg) / max(1, len(Xpos))), torch.ones(len(Xneg))]).to(device)
    Xall = torch.cat([Xpos, Xneg], 0)
    for _ in range(steps):
        opt.zero_grad()
        sp = head(Xpos)
        sn = head(Xneg)
        tau = torch.quantile(sn.detach(), q)
        scale = temp * (sn.detach().std() + 1e-6)
        rec = torch.stack([torch.sigmoid((sp[rids == r] - tau) / scale).mean() for r in uniq]).mean()
        fp_excess = torch.sigmoid((sn - tau) / scale).mean()
        bce = F.binary_cross_entropy_with_logits(head(Xall), y, weight=wbce)
        loss = (1 - rec) + w_fp * fp_excess + w_bce * bce
        loss.backward()
        opt.step()
    return head


# ===== head few-shot + SHIFT-AUG (ĐIỂM KHÁC DUY NHẤT so với infer_submit gốc) =====
def build_head(bb, ds, shot_pool, bank, args, layers, device, shift_rng):
    """Trích defect/normal từ 10 shot. n_aug>0: thêm bản photometric_shift.
       n_aug=0: shifts=[clean], negsh rỗng -> Y HỆT build_head gốc (baseline)."""
    T = args.tiles
    gt = args.grid_tile
    R = args.grid_tile * bb.patch
    Cdim = bank.shape[-1]
    pos_list, rid_list, negsh_list = [], [], []
    rbase = 0
    for i in shot_pool:
        pil0 = Image.open(ds.img_paths[i])
        shifts = [0.0] + [float(shift_rng.uniform(args.shift_lo, args.shift_hi)) for _ in range(args.n_aug)]
        for s in shifts:
            pil = photometric_shift(pil0, s) if s > 0 else pil0
            g = img_featgrid(bb, pil, T, R, gt, layers, args.enc_batch).cpu().numpy()
            G = g.shape[0]
            m = gt_grid(ds.gt_paths[i], 1, G)
            lab, n = cc_label(m)
            flat = g.reshape(-1, Cdim)
            for rid in range(1, n + 1):
                idxs = (lab.reshape(-1) == rid)
                if idxs.sum() > 0:
                    pos_list.append(flat[idxs])
                    rid_list.append(np.full(int(idxs.sum()), rbase))
                    rbase += 1
            if s > 0:                                    # normal-pixel ĐÃ SHIFT -> negative aug
                negsh_list.append(flat[m.reshape(-1) == 0])
    # điều kiện GIỐNG train_softpro: cần >=1 region và >=3 pixel defect
    if rbase < 1 or sum(len(x) for x in pos_list) < 3:
        return None
    Xpos = torch.tensor(np.concatenate(pos_list), device=device)
    region_ids = np.concatenate(rid_list)
    Xneg = subsample(bank, args.n_neg).detach()
    if negsh_list:
        neg_sh = subsample(torch.tensor(np.concatenate(negsh_list), device=device), args.n_neg // 2)
        Xneg = torch.cat([Xneg, neg_sh], 0)
    mu = bank.mean(0, keepdim=True)
    sd = bank.std(0, keepdim=True) + 1e-6
    if args.loss == 'softpro':
        head = train_softpro(Head(Cdim, mu, sd).to(device), Xpos, region_ids, Xneg,
                             args.steps, args.lr, args.q, args.temp, args.w_bce, args.w_fp, device)
    else:
        head = train_bce(Head(Cdim, mu, sd).to(device), Xpos, Xneg, args.steps, args.lr, device)
    head.eval()
    return head


@torch.no_grad()
def score_grid(bb, pil, bank, head, args, layers, device):
    T = args.tiles
    gt = args.grid_tile
    R = args.grid_tile * bb.patch
    Cdim = bank.shape[-1]
    grid = img_featgrid(bb, pil, T, R, gt, layers, args.enc_batch)
    G = grid.shape[0]
    d = nn_map(grid, bank, device)
    pr = None
    if head is not None:
        pr = torch.sigmoid(head(grid.reshape(-1, Cdim))).reshape(G, G).cpu().numpy()
    return d, pr


# ===== map 56 -> 256 (gaussian, khớp train_softpro) -> (H,W) gốc ============
def up_to(arr2d, size_hw, gk, device):
    H, W = size_hw
    t = torch.tensor(arr2d, device=device)[None, None].float()
    t = F.interpolate(t, size=SMOOTH_RES, mode='bilinear', align_corners=False)
    t = gk(t)
    t = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)
    return t[0, 0].cpu().numpy()


def save_tiff_f16(arr, path, compression=None):
    import tifffile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = {} if compression in (None, 'none', '') else {'compression': compression}
    tifffile.imwrite(path, arr.astype(np.float16), **kw)


def save_png_binary(mask_bool, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((mask_bool.astype(np.uint8) * 255), mode='L').save(path)


def list_split_files(data_path, cat, split):
    root = os.path.join(data_path, cat, split)
    if not os.path.isdir(root):
        return None, None
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(root, ext))
    return sorted(files), root


def main():
    ap = argparse.ArgumentParser('MVTec AD 2 submission — SHIFT-AUG few-shot head (mirror infer_submit)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64, help='PHẢI trùng train_softpro (64) để bank giống hệt')
    ap.add_argument('--shots', type=int, default=10)
    # ---- SHIFT-AUG ----
    ap.add_argument('--n_aug', type=int, default=3, help='số bản shift-augment mỗi shot (0 = baseline y hệt infer gốc)')
    ap.add_argument('--shift_lo', type=float, default=0.3)
    ap.add_argument('--shift_hi', type=float, default=1.2)
    # -------------------
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=3.0, help='ngưỡng seg = mean + k*std trên validation/good')
    ap.add_argument('--max_val', type=int, default=0, help='0 = dùng hết validation/good cho ngưỡng')
    ap.add_argument('--no_thresholded', action='store_true', help='bỏ nhánh .png (chỉ chấm AUPRO)')
    ap.add_argument('--tiff_compression', type=str, default='zlib',
                    help="nén tiff (map mượt -> ~8-10x). 'zlib'/'lzw'/'none'.")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anomaly_dirname', type=str, default='anomaly_images')
    ap.add_argument('--thresh_dirname', type=str, default='anomaly_images_thresholded')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./submit_out_shiftaug')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_shiftaug', args.out_dir).info
    torch.manual_seed(args.seed)                       # KHỚP train_softpro (Head init)

    bb = load_backbone(args.model, device)
    R = args.grid_tile * bb.patch
    if args.layers_fixed or not bb.n_layers:
        layers = [ly for ly in args.layers if ly < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(ly / 12 * bb.n_layers))) for ly in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles
    gt = args.grid_tile
    hw = args.head_w
    rng = np.random.default_rng(args.seed)             # KHỚP train_softpro (chọn shots)
    shift_rng = np.random.default_rng(args.seed + 12345)   # RIÊNG cho augment (n_aug=0 -> không bốc)
    p('=' * 92)
    p(f'SUBMIT SHIFT-AUG | model={args.model} eff_grid={T*gt} layers={layers} | loss={args.loss} '
      f'shots={args.shots} n_aug={args.n_aug} shift~U[{args.shift_lo},{args.shift_hi}] head_w={hw} '
      f'enc_batch={args.enc_batch}')
    p(f'  smooth@{SMOOTH_RES}->native | seg={"off" if args.no_thresholded else f"mean+{args.thr_sigma}std"} | '
      f'{"BASELINE (n_aug=0)" if args.n_aug == 0 else "METHOD shift-aug"}')
    p('=' * 92)

    for cat in args.categories:
        if cat not in OBJECT_FILE_COUNTER:
            p(f'  [skip] {cat}: không phải object AD2')
            continue
        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ')
            continue
        bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)

        # ---- shots=10 từ test_public (giống train_softpro; rng KHÔNG bị shift_rng đụng) ----
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng.shuffle(bad)
        shot_pool = bad[:args.shots]
        head = build_head(bb, ds, shot_pool, bank, args, layers, device, shift_rng)
        if head is None:
            p(f'  [{cat}] thiếu defect region ở shots -> fallback DISTANCE-ONLY')
        else:
            n_views = 1 + args.n_aug
            p(f'  [{cat}] head={args.loss} từ {len(shot_pool)} shot x {n_views} view (n_aug={args.n_aug})')

        # ---- validation/good (chỉ để lấy ngưỡng seg) ----
        val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e)) for e in IMG_EXT], []))
        if args.max_val and len(val_imgs) > args.max_val:
            val_imgs = val_imgs[:args.max_val]
        val_pairs = [score_grid(bb, Image.open(v), bank, head, args, layers, device)
                     for v in tqdm(val_imgs, ncols=80, desc=f'  {cat}/val')] if val_imgs else []

        # ---- PASS 1: score cả 2 private split, GỘP distance -> 1 lo/hi ----
        split_recs = {}
        pooled_dist = []
        for split in SPLITS:
            files, root = list_split_files(args.data_path, cat, split)
            if files is None:
                p(f'  [{cat}/{split}] không tồn tại -> bỏ')
                continue
            exp = OBJECT_FILE_COUNTER[cat]
            if len(files) != exp:
                p(f'  [{cat}/{split}] CẢNH BÁO: {len(files)} ảnh (checker cần {exp})')
            recs = []
            for fp in tqdm(files, ncols=80, desc=f'  {cat}/{split}'):
                pil = Image.open(fp)
                W, Himg = pil.size
                d, pr = score_grid(bb, pil, bank, head, args, layers, device)
                recs.append((fp, Himg, W, d, pr))
                pooled_dist.append(d.reshape(-1))
            split_recs[split] = (recs, root)
        if not pooled_dist:
            p(f'  [{cat}] không có ảnh private -> bỏ')
            continue

        all_dist = np.concatenate(pooled_dist)
        lo, hi = np.percentile(all_dist, 1), np.percentile(all_dist, 99)

        def fuse(d, pr):
            dr = (d - lo) / (hi - lo + 1e-8)
            return dr if (head is None or pr is None) else (1 - hw) * dr + hw * pr

        # ---- ngưỡng seg từ validation/good, cùng lo/hi (private không có GT) ----
        thr = None
        if not args.no_thresholded and val_pairs:
            vs = np.concatenate([up_to(fuse(d, pr), (SMOOTH_RES, SMOOTH_RES), gk, device).reshape(-1)
                                 for d, pr in val_pairs])
            thr = float(vs.mean() + args.thr_sigma * vs.std())
        p(f'  [{cat}] dist_lo={lo:.3f} dist_hi={hi:.3f}' + (f' thr={thr:.4f}' if thr is not None else ' (no thr)'))

        # ---- PASS 2: fuse -> upsample (H,W) -> lưu float16 tiff (+ png) ----
        for split, (recs, root) in split_recs.items():
            for fp, Himg, W, d, pr in recs:
                amap = up_to(fuse(d, pr), (Himg, W), gk, device)
                stem = os.path.splitext(os.path.basename(fp))[0]
                save_tiff_f16(amap, os.path.join(args.out_dir, args.anomaly_dirname, cat, split, stem + '.tiff'),
                              args.tiff_compression)
                if thr is not None:
                    save_png_binary(amap > thr, os.path.join(args.out_dir, args.thresh_dirname, cat, split, stem + '.png'))
            p(f'  [{cat}/{split}] lưu {len(recs)} tiff' + ('' if thr is None else ' + png'))

    p('\n' + '=' * 92)
    p(f'XONG: {args.out_dir}')
    p('KIỂM TRA: cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py '
      f'{os.path.abspath(args.out_dir)}')
    if args.no_thresholded:
        p('LƯU Ý: bỏ .png -> chỉ chấm AUPRO0.05 (threshold-independent), KHÔNG có SegF1.')


if __name__ == '__main__':
    main()
