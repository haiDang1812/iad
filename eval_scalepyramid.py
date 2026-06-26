# eval_scalepyramid.py
# -----------------------------------------------------------------------------
# NOVELTY BET: Scale-Pyramid frozen-DINOv2 distance + cross-scale PEAK-PRESERVING
# aggregation cho chế độ LOW-FPR (AUPRO@0.05).
#
# Động lực (từ diagnosis + các thí nghiệm trước):
#   - Granularity là đòn bẩy (full_t4=0.600) nhưng 1 scale cố định là dưới-tối-ưu vì
#     defect đủ kích cỡ: tile nhỏ (T=4) bắt defect bé, tile to (T=1/2) nền sạch hơn.
#   - low-FPR bị chi phối bởi ĐỈNH SẮC -> gộp 'mean' làm tù đỉnh (đã thấy hại);
#     gộp 'max' (per-pixel) GIỮ đỉnh -> chọn granularity tốt nhất cho từng pixel.
#
# So trong 1 lần chạy: từng scale riêng + agg 'mean' + agg 'max' (= method).
# Mốc cần vượt: full_t4 (single-scale T=4) = AUPRO0.05 0.600.
#
# Chuẩn hoá mỗi scale per-image (percentile 1/99) trước khi gộp -> các scale so được.
#
# Chạy:
#   uv run python eval_scalepyramid.py --data_path ../data --scales 1 2 4 \
#       --enc_batch 64 --out_dir ./diagnosis_pyr/s124
# -----------------------------------------------------------------------------

import os
import math
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from models import vit_encoder
from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger

warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    out = []
    for i in range(T):
        for j in range(T):
            out.append(pil.crop((round(j * w / T), round(i * h / T),
                                 round((j + 1) * w / T), round((i + 1) * h / T))))
    return out


@torch.no_grad()
def extract(encoder, imgs, layers, n_reg, device):
    x = encoder.prepare_tokens(imgs.to(device))
    feats, last = [], max(layers)
    for i, blk in enumerate(encoder.blocks):
        if i <= last:
            x = blk(x)
        if i in layers:
            feats.append(x[:, 1 + n_reg:, :])
    return torch.stack(feats, dim=1).mean(dim=1)


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def nn_dist(feats, bank, device, chunk=4096):
    q = feats.reshape(-1, feats.shape[-1]).to(device)
    out = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(dim=1)[0]
    return out


def build_bank(paths, encoder, T, args, n_reg, device):
    # bank cho 1 scale (chia ảnh T×T tile @tile_res), enc_batch streaming
    keep = max(64, (args.bank_size * 4) // max(1, len(paths) * T * T))
    acc, buf = [], []

    def flush():
        if not buf:
            return
        f = extract(encoder, torch.stack(buf), args.layers, n_reg, device)
        acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        buf.clear()

    for p in tqdm(paths, ncols=80, desc=f'  bank-T{T}'):
        for t in tile_pils(Image.open(p), T):
            buf.append(to_tensor(t, args.tile_res))
            if len(buf) >= args.enc_batch:
                flush()
    flush()
    return subsample(torch.cat(acc, 0), args.bank_size).to(device)


@torch.no_grad()
def scale_map(encoder, pil, T, bank, args, n_reg, device, gk):
    # map bất thường ở 1 scale -> [resize_mask, resize_mask] (đã smooth)
    ts = args.tile_res // 14
    tiles = tile_pils(pil, T)
    feats = []
    for s in range(0, len(tiles), args.enc_batch):
        b = torch.stack([to_tensor(t, args.tile_res) for t in tiles[s:s + args.enc_batch]])
        feats.append(extract(encoder, b, args.layers, n_reg, device))
    feats = torch.cat(feats, 0)                                   # [T*T, N, C]
    d = nn_dist(feats, bank, device).reshape(T * T, ts, ts)
    grid = torch.zeros(T * ts, T * ts, device=device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = d[k]
    m = F.interpolate(grid[None, None], size=args.resize_mask, mode='bilinear', align_corners=False)
    return gk(m)[0, 0].cpu().numpy()


def gpnorm(stack, lo=1.0, hi=99.0):
    # chuẩn hoá GLOBAL theo toàn bộ test set của 1 scale (giữ khác biệt magnitude giữa ảnh
    # -> điểm ảnh-level biến thiên, tránh score_min==score_max làm adeval assert chết)
    a, b = np.percentile(stack, lo), np.percentile(stack, hi)
    return np.clip((stack - a) / (b - a + 1e-12), 0.0, 1.0)


def sp_score(m, r=0.01):
    f = m.reshape(-1)
    return np.sort(f)[::-1][:max(1, int(f.size * r))].mean()


def evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn):
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                         transform=None, gt_transform=None, phase='test')
    train_paths = (glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                   glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')) +
                   glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.JPG')))

    print_fn(f'  [{cat}] build banks scales={args.scales} (train={len(train_paths)})...')
    banks = {T: build_bank(train_paths, encoder, T, args, n_reg, device) for T in args.scales}

    # branch = từng scale + mean + max
    branches = [f'T{T}' for T in args.scales] + ['MEAN', 'MAX']
    raw = {T: [] for T in args.scales}            # map THÔ mỗi scale (chưa norm)
    gt_maps = []

    for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc='  test'):
        ipath, gpath, label = ds.img_paths[idx], ds.gt_paths[idx], ds.labels[idx]
        pil = Image.open(ipath)
        for T in args.scales:
            raw[T].append(scale_map(encoder, pil, T, banks[T], args, n_reg, device, gk))

        if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
            gt = np.zeros((args.resize_mask, args.resize_mask), dtype=np.uint8)
        else:
            g = Image.open(gpath).convert('L').resize((args.resize_mask, args.resize_mask), Image.NEAREST)
            gt = (np.asarray(g) > 127).astype(np.uint8)
        gt_maps.append(gt)

    # chuẩn hoá GLOBAL per-scale rồi gộp (giữ khác biệt giữa ảnh)
    normed = {T: gpnorm(np.stack(raw[T], 0)) for T in args.scales}
    arrs = {f'T{T}': normed[T] for T in args.scales}
    stk = np.stack([normed[T] for T in args.scales], 0)           # [S, Ntest, H, W]
    arrs['MEAN'] = stk.mean(0)
    arrs['MAX'] = stk.max(0)                                      # peak-preserving = method

    gt = np.stack(gt_maps, 0)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    res = {}
    for b in branches:
        arr = arrs[b]
        sp = np.array([sp_score(m, args.max_ratio) for m in arr])
        r = ader_evaluator(arr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
        res[b] = r
        print_fn(f'  [{cat}/{b}] AUPRO0.05={r[7]:.4f} AUPRO={r[6]:.4f} P-AUROC={r[3]:.4f}')
    return res


def main():
    ap = argparse.ArgumentParser('Scale-pyramid frozen-DINOv2 distance (peak-preserving)')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tile_res', type=int, default=392, help='res mỗi tile (chia hết 14)')
    ap.add_argument('--scales', type=int, nargs='+', default=[1, 2, 4], help='các mức tiling T')
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=128)
    ap.add_argument('--resize_mask', type=int, default=256)
    ap.add_argument('--max_ratio', type=float, default=0.01)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    ap.add_argument('--out_dir', type=str, default='./diagnosis_pyr')
    args = ap.parse_args()

    if args.tile_res % 14 != 0:
        raise SystemExit('tile_res phải chia hết 14.')
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA không khả dụng (adeval cần GPU).")
    device = 'cuda:0'

    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger('pyramid', args.out_dir)
    print_fn = logger.info
    print_fn('=' * 70)
    print_fn(f'SCALE-PYRAMID | scales={args.scales} (eff {[T*args.tile_res for T in args.scales]}) '
             f'| layers={args.layers} | bank={args.bank_size}')
    print_fn('Branch MAX = cross-scale peak-preserving (method). Mốc: full_t4 single-scale = 0.600')
    print_fn('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_res = {}
    for cat in args.categories:
        all_res[cat] = evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn)

    branches = [f'T{T}' for T in args.scales] + ['MEAN', 'MAX']
    print_fn('\n' + '=' * 70)
    print_fn('{:<8} '.format('Branch') + ' '.join(f'{m:>10}' for m in METRIC_NAMES))
    mean_by_branch = {}
    for b in branches:
        mr = np.array([all_res[c][b] for c in all_res]).mean(0)
        mean_by_branch[b] = mr
        print_fn('{:<8} '.format(b) + ' '.join(f'{v:>10.4f}' for v in mr))

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('branch,category,' + ','.join(METRIC_NAMES) + '\n')
        for b in branches:
            for c in all_res:
                f.write(f'{b},{c},' + ','.join(f'{v:.4f}' for v in all_res[c][b]) + '\n')
            f.write(f'{b},MEAN,' + ','.join(f'{v:.4f}' for v in mean_by_branch[b]) + '\n')
    print_fn(f'\nĐã lưu: {csv}')


if __name__ == '__main__':
    main()
