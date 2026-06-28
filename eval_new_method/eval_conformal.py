# eval_conformal.py
# -----------------------------------------------------------------------------
# NOVELTY: Locally-Calibrated (conditional-conformal) anomaly score trên frozen DINOv2.
#
# Thay NN distance thô d  ->  d / (local normal scale)
#   local normal scale = khoảng cách NN NỘI-BỘ của patch bank gần nhất (độ rộng normal cục bộ).
# Ý nghĩa (gốc từ D9 + conformal): bất thường = độ lệch TƯƠNG ĐỐI so với normal cục bộ,
# không phải khoảng cách tuyệt đối.
#   - Re-rank pixel  -> TÁC ĐỘNG AUPRO@0.05 (khác global-conformal vốn bất biến rank).
#   - Điểm đã hiệu chỉnh -> ngưỡng FPR-controlled -> SegF1.
#
# So raw-MAX vs calib-MAX trên scale-pyramid {1,2,4}. Mốc raw-MAX = 0.611 (đã đo).
#
# Chạy:
#   python eval_conformal.py --data_path ../data --scales 1 2 4 --enc_batch 64 --out_dir ./diagnosis_conf/c124
# -----------------------------------------------------------------------------

import os
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
EXT = ('*.png', '*.jpg', '*.JPG')


def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    return [pil.crop((round(j * w / T), round(i * h / T), round((j + 1) * w / T), round((i + 1) * h / T)))
            for i in range(T) for j in range(T)]


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


def build_bank(paths, encoder, T, args, n_reg, device):
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
def within_bank_scale(bank, chunk=2048, eps=1e-6):
    # local normal scale = khoảng cách NN nội-bộ của mỗi điểm bank (bỏ chính nó)
    M = bank.shape[0]
    out = torch.empty(M, device=bank.device)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        d = torch.cdist(bank[s:e], bank)
        idx = torch.arange(e - s, device=bank.device)
        d[idx, s + idx] = float('inf')
        out[s:e] = d.min(1)[0]
    return out.clamp_min(eps)


@torch.no_grad()
def nn_dist_arg(feats, bank, chunk=4096):
    q = feats.reshape(-1, feats.shape[-1])
    dmin = torch.empty(q.shape[0], device=q.device)
    amin = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s:s + chunk], bank)
        v, a = d.min(1)
        dmin[s:s + chunk] = v
        amin[s:s + chunk] = a
    return dmin, amin


@torch.no_grad()
def scale_maps(encoder, pil, T, bank, local_scale, args, n_reg, device, gk, eps=1e-6):
    # trả (raw_map, calib_map) ở proc_size cho 1 scale
    ts = args.tile_res // 14
    tiles = tile_pils(pil, T)
    feats = []
    for s in range(0, len(tiles), args.enc_batch):
        b = torch.stack([to_tensor(t, args.tile_res) for t in tiles[s:s + args.enc_batch]])
        feats.append(extract(encoder, b, args.layers, n_reg, device))
    feats = torch.cat(feats, 0)
    dmin, amin = nn_dist_arg(feats, bank)
    calib = dmin / (local_scale[amin] + eps)                 # <-- novelty: chia local scale

    def to_map(vals):
        g = torch.zeros(T * ts, T * ts, device=device)
        v = vals.reshape(T * T, ts, ts)
        for k in range(T * T):
            i, j = k // T, k % T
            g[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = v[k]
        m = F.interpolate(g[None, None], size=args.proc_size, mode='bilinear', align_corners=False)
        return gk(m)[0, 0].cpu().numpy()

    return to_map(dmin), to_map(calib)


def gpnorm(stack, lo=1.0, hi=99.0):
    a, b = np.percentile(stack, lo), np.percentile(stack, hi)
    return (stack - a) / (b - a + 1e-12)


def sp_score(m, r=0.01):
    f = m.reshape(-1)
    return np.sort(f)[::-1][:max(1, int(f.size * r))].mean()


def evaluate(arr, gt, gt_sp, r=0.01):
    sp = np.array([sp_score(m, r) for m in arr])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    return ader_evaluator(arr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)


def find_imgs(root):
    fs = []
    for e in EXT:
        fs += glob.glob(os.path.join(root, '**', e), recursive=True)
    return sorted(set(fs))


def evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn):
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                         transform=None, gt_transform=None, phase='test')
    train_paths = find_imgs(os.path.join(args.data_path, cat, 'train', 'good'))

    print_fn(f'  [{cat}] build banks + local-scale scales={args.scales} (train={len(train_paths)})')
    banks, lscale = {}, {}
    for T in args.scales:
        banks[T] = build_bank(train_paths, encoder, T, args, n_reg, device)
        lscale[T] = within_bank_scale(banks[T])

    raw = {T: [] for T in args.scales}
    cal = {T: [] for T in args.scales}
    gts = []
    for idx in tqdm(range(len(ds.img_paths)), ncols=80, desc='  test'):
        ipath, gpath, label = ds.img_paths[idx], ds.gt_paths[idx], ds.labels[idx]
        pil = Image.open(ipath)
        for T in args.scales:
            rm, cm = scale_maps(encoder, pil, T, banks[T], lscale[T], args, n_reg, device, gk)
            raw[T].append(rm)
            cal[T].append(cm)
        if label == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
            g = np.zeros((args.proc_size, args.proc_size), dtype=np.uint8)
        else:
            gi = Image.open(gpath).convert('L').resize((args.proc_size, args.proc_size), Image.NEAREST)
            g = (np.asarray(gi) > 127).astype(np.uint8)
        gts.append(g)

    gt = np.stack(gts, 0)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    res = {}
    for name, store in [('RAW-MAX', raw), ('CALIB-MAX', cal)]:
        normed = [gpnorm(np.stack(store[T], 0)) for T in args.scales]
        fused = np.maximum.reduce(normed)
        res[name] = evaluate(fused, gt, gt_sp, args.max_ratio)
        r = res[name]
        print_fn(f'  [{cat}/{name}] AUPRO0.05={r[7]:.4f} AUPRO={r[6]:.4f} P-F1max={r[5]:.4f} P-AUROC={r[3]:.4f}')
    return res


def main():
    ap = argparse.ArgumentParser('Locally-calibrated (conditional-conformal) anomaly scoring')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--scales', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--tile_res', type=int, default=392)
    ap.add_argument('--proc_size', type=int, default=256)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--max_ratio', type=float, default=0.01)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    ap.add_argument('--out_dir', type=str, default='./diagnosis_conf')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger('conformal', args.out_dir)
    print_fn = logger.info
    print_fn('=' * 70)
    print_fn(f'LOCAL-CALIB | scales={args.scales} | layers={args.layers} | bank={args.bank_size}')
    print_fn('So RAW-MAX (mốc 0.611) vs CALIB-MAX (d / local-normal-scale)')
    print_fn('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_res = {}
    for cat in args.categories:
        all_res[cat] = evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn)

    print_fn('\n' + '=' * 70)
    print_fn('{:<10} '.format('Branch') + ' '.join(f'{m:>10}' for m in METRIC_NAMES))
    means = {}
    for name in ['RAW-MAX', 'CALIB-MAX']:
        mr = np.array([all_res[c][name] for c in all_res]).mean(0)
        means[name] = mr
        print_fn('{:<10} '.format(name) + ' '.join(f'{v:>10.4f}' for v in mr))

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('branch,category,' + ','.join(METRIC_NAMES) + '\n')
        for name in ['RAW-MAX', 'CALIB-MAX']:
            for c in all_res:
                f.write(f'{name},{c},' + ','.join(f'{v:.4f}' for v in all_res[c][name]) + '\n')
            f.write(f'{name},MEAN,' + ','.join(f'{v:.4f}' for v in means[name]) + '\n')
    print_fn(f'\nĐã lưu: {csv}')


if __name__ == '__main__':
    main()
