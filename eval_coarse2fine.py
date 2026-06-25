# eval_coarse2fine.py
# -----------------------------------------------------------------------------
# PROTOTYPE NOVELTY: Coarse-to-Fine granularity (đạt chất lượng hi-res với chi phí thấp).
# Chạy được trên 16GB (tile ở 392, không cần forward ảnh 672 khổng lồ).
#
# 3 mode (so trong cùng script để trả lời "novelty có thật không"):
#   coarse    : chỉ chấm ở res thấp (full ảnh @coarse_res). Rẻ, yếu (~baseline 392).
#   full_tile : chia ảnh T×T tile, MỌI tile chạy hi-res. = brute hi-res (mạnh, đắt).
#   c2f       : coarse screen -> CHỈ refine hi-res ở tile nghi ngờ. = METHOD.
#
# Câu hỏi quyết định: c2f có ≈ full_tile (AUPRO0.05) mà xử lý ÍT tile hơn nhiều không?
# -> nếu có: granularity hiệu quả = novelty. In kèm "avg fine-tiles/image" làm proxy chi phí.
#
# Bank: coarse_bank từ train full@coarse_res; fine_bank từ train TILE@tile_res (khác phân bố).
#
# Chạy:
#   python eval_coarse2fine.py --data_path ../data --mode c2f      --out_dir ./diagnosis_c2f/c2f
#   python eval_coarse2fine.py --data_path ../data --mode full_tile --out_dir ./diagnosis_c2f/full
#   python eval_coarse2fine.py --data_path ../data --mode coarse    --out_dir ./diagnosis_c2f/coarse
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
    # chia ảnh gốc thành T×T crop (theo toạ độ gốc, giữ chi tiết), trả list theo (row,col)
    w, h = pil.size
    out = []
    for i in range(T):
        for j in range(T):
            box = (round(j * w / T), round(i * h / T), round((j + 1) * w / T), round((i + 1) * h / T))
            out.append(pil.crop(box))
    return out


@torch.no_grad()
def extract(encoder, imgs, layers, n_reg, device):
    # imgs [B,3,R,R] -> [B, N, C] (fuse mean các layer)
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


def build_bank(paths, encoder, args, n_reg, device, kind):
    # kind='coarse' -> full@coarse_res ; kind='fine' -> T×T tile@tile_res
    keep = max(256, (args.bank_size * 4) // max(1, len(paths)))
    acc = []
    for p in tqdm(paths, ncols=80, desc=f'  bank-{kind}'):
        pil = Image.open(p)
        if kind == 'coarse':
            batch = [to_tensor(pil, args.coarse_res)]
        else:
            batch = [to_tensor(t, args.tile_res) for t in tile_pils(pil, args.tiles)]
        f = extract(encoder, torch.stack(batch), args.layers, n_reg, device)
        flat = f.reshape(-1, f.shape[-1])
        acc.append(subsample(flat, len(batch) * keep).cpu())
    return subsample(torch.cat(acc, 0), args.bank_size).to(device)


def evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn):
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                         transform=None, gt_transform=None, phase='test')
    train_paths = (glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                   glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')) +
                   glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.JPG')))

    print_fn(f'  [{cat}] build banks (train={len(train_paths)})...')
    coarse_bank = build_bank(train_paths, encoder, args, n_reg, device, 'coarse')
    fine_bank = (build_bank(train_paths, encoder, args, n_reg, device, 'fine')
                 if args.mode != 'coarse' else None)

    scs = args.coarse_res // 14            # coarse grid side
    ts = args.tile_res // 14               # tile grid side
    fg = args.tiles * ts                   # full canvas grid side
    T = args.tiles

    pr_maps, gt_maps, fine_tiles_used = [], [], []
    for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  test-{args.mode}'):
        ipath, gpath, label = ds.img_paths[idx], ds.gt_paths[idx], ds.labels[idx]
        pil = Image.open(ipath)

        # --- coarse pass (full ảnh) ---
        cfeat = extract(encoder, to_tensor(pil, args.coarse_res).unsqueeze(0), args.layers, n_reg, device)
        cmap = nn_dist(cfeat, coarse_bank, device).reshape(scs, scs)          # [scs,scs]
        canvas = F.interpolate(cmap[None, None], size=(fg, fg), mode='bilinear',
                               align_corners=False)[0, 0]                      # bắt đầu từ coarse

        n_fine = 0
        if args.mode != 'coarse':
            # chọn tile ứng viên
            tile_score = F.adaptive_max_pool2d(cmap[None, None], (T, T))[0, 0]  # [T,T] max coarse score/tile
            if args.mode == 'full_tile':
                cand = torch.ones(T, T, dtype=torch.bool)
            else:  # c2f: tile có đỉnh coarse vượt mean+k*std (per-image, không leakage)
                thr = cmap.mean() + args.k_std * cmap.std()
                cand = tile_score > thr
                if not cand.any():
                    ci = torch.argmax(tile_score)
                    cand.view(-1)[ci] = True                                   # luôn giữ tile đỉnh nhất

            tiles = tile_pils(pil, T)
            sel = [(i, j) for i in range(T) for j in range(T) if cand[i, j]]
            n_fine = len(sel)
            if sel:
                batch = torch.stack([to_tensor(tiles[i * T + j], args.tile_res) for (i, j) in sel])
                ffeat = extract(encoder, batch, args.layers, n_reg, device)    # [k,N,C]
                fd = nn_dist(ffeat, fine_bank, device).reshape(len(sel), ts, ts)
                for m, (i, j) in enumerate(sel):
                    canvas[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = fd[m]
        fine_tiles_used.append(n_fine)

        amap = F.interpolate(canvas[None, None], size=args.resize_mask, mode='bilinear', align_corners=False)
        amap = gk(amap)[0, 0].cpu().numpy()

        # gt
        if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
            gt = np.zeros((args.resize_mask, args.resize_mask), dtype=np.uint8)
        else:
            g = Image.open(gpath).convert('L').resize((args.resize_mask, args.resize_mask), Image.NEAREST)
            gt = (np.asarray(g) > 127).astype(np.uint8)
        pr_maps.append(amap); gt_maps.append(gt)

    pr = np.stack(pr_maps, 0); gt = np.stack(gt_maps, 0)
    pr_sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * args.max_ratio))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    r = ader_evaluator(pr, pr_sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    avg_tiles = float(np.mean(fine_tiles_used))
    print_fn(f'  === {cat.upper()} ({args.mode}) === ' +
             ' '.join(f'{n}:{v:.4f}' for n, v in zip(METRIC_NAMES, r)) +
             f'  | avg_fine_tiles/img={avg_tiles:.2f}/{T*T}')
    return r, avg_tiles


def main():
    ap = argparse.ArgumentParser('Coarse-to-Fine granularity prototype')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--coarse_res', type=int, default=392, help='res screen (chia hết 14)')
    ap.add_argument('--tile_res', type=int, default=392, help='res mỗi tile (chia hết 14)')
    ap.add_argument('--tiles', type=int, default=2, help='T: chia ảnh T×T tile (eff res = T*tile_res)')
    ap.add_argument('--k_std', type=float, default=1.0, help='ngưỡng chọn tile c2f: mean+k*std')
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--resize_mask', type=int, default=256)
    ap.add_argument('--max_ratio', type=float, default=0.01)
    ap.add_argument('--mode', type=str, default='c2f', choices=['coarse', 'full_tile', 'c2f'])
    ap.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    ap.add_argument('--out_dir', type=str, default='./diagnosis_c2f')
    args = ap.parse_args()

    for v in (args.coarse_res, args.tile_res):
        if v % 14 != 0:
            raise SystemExit(f'res {v} phải chia hết 14 (vd 392=14*28).')
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA không khả dụng (adeval cần GPU). "
                         "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    device = 'cuda:0'

    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger(f'c2f_{args.mode}', args.out_dir)
    print_fn = logger.info
    print_fn('=' * 70)
    print_fn(f'COARSE2FINE | mode={args.mode} | coarse_res={args.coarse_res} tile_res={args.tile_res} '
             f'tiles={args.tiles} (eff {args.tiles*args.tile_res}) | k_std={args.k_std}')
    print_fn('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_res, all_tiles = {}, {}
    for cat in args.categories:
        r, t = evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn)
        all_res[cat] = r; all_tiles[cat] = t

    print_fn('\n' + '=' * 70)
    mean_r = np.array(list(all_res.values())).mean(0)
    print_fn('{:<10} '.format('') + ' '.join(f'{m:>10}' for m in METRIC_NAMES))
    print_fn('{:<10} '.format(f'MEAN[{args.mode}]') + ' '.join(f'{v:>10.4f}' for v in mean_r))
    print_fn(f'avg fine-tiles/img (mean over cats): {np.mean(list(all_tiles.values())):.2f} / {args.tiles**2}')

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('category,' + ','.join(METRIC_NAMES) + ',avg_fine_tiles\n')
        for cat in all_res:
            f.write(f'{cat},' + ','.join(f'{v:.4f}' for v in all_res[cat]) + f',{all_tiles[cat]:.2f}\n')
        f.write('MEAN,' + ','.join(f'{v:.4f}' for v in mean_r) +
                f',{np.mean(list(all_tiles.values())):.2f}\n')
    print_fn(f'\nĐã lưu: {csv}')


if __name__ == '__main__':
    main()
