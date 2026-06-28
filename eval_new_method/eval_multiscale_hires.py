# eval_multiscale_hires.py
# -----------------------------------------------------------------------------
# HƯỚNG MỚI (sau khi research lật lại): bottleneck của ta là ĐỘ PHÂN GIẢI, không
# phải supervision. Ta downsample 2448x2048 -> 392 (lưới patch 28x28) nên defect
# nhỏ (<0.2% diện tích, D1/D2) bị xoá trước khi chấm điểm.
# Bằng chứng: SuperAD (DINOv2-L, shorter-side 672) = AUPRO0.05 0.605; RoBiS
# (DINOv2 ViT-B, 518 + crop 1024 overlap) = 0.672 — CÙNG họ frozen-DINOv2+distance
# nhưng res cao hơn nhiều -> vượt xa 0.436 của ta.
#
# Script này: frozen DINOv2 + NN distance ở ĐỘ PHÂN GIẢI CAO + MULTI-SCALE
#   - fine layers (nông, định vị defect nhỏ) và coarse layers (sâu, ngữ nghĩa)
#     dùng memory bank RIÊNG -> NN map riêng -> chuẩn hoá -> gộp.
#   - memory bank cap bằng random subsample (coreset 25% quá chậm ở res cao).
#   - postproc tuỳ chọn: morphological closing.
#
# Mốc cần vượt: DIST Euclid 392 = AUPRO0.05 0.436.
#
# Chạy:
#   # multi-scale, res 672 (lưới 48x48)
#   python eval_multiscale_hires.py --data_path ../data --input_size 672 --crop_size 672 \
#       --mode multi --out_dir ./diagnosis_hires/multi672
#   # single-scale để ablation
#   python eval_multiscale_hires.py --data_path ../data --input_size 672 --crop_size 672 \
#       --mode single --out_dir ./diagnosis_hires/single672
# -----------------------------------------------------------------------------

import os
import math
import argparse
import warnings

import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from utils import ader_evaluator, get_gaussian_kernel, get_logger


class MVTecAD2DatasetAR(MVTecAD2Dataset):
    """Bản giữ tỉ lệ (non-square): sửa gt ảnh 'good' dùng đúng [1,H,W] (gốc giả định vuông)."""
    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.shape[-2], img.shape[-1]])
        elif os.path.exists(gt):
            gt = self.gt_transform(Image.open(gt))
        else:
            gt = torch.zeros([1, img.shape[-2], img.shape[-1]])
        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size"
        return img, gt, label, img_path


def build_transforms(args, cat):
    """Trả (img_tf, gt_tf, hp, wp). hp,wp = số patch theo chiều cao/rộng."""
    if not args.aspect_preserve:
        dtf, gtf = get_data_transforms(args.input_size, args.crop_size)
        s = args.crop_size // 14
        return dtf, gtf, s, s
    # giữ tỉ lệ: resize cạnh ngắn -> short_side, ép cả 2 chiều chia hết 14 (đồng nhất/category)
    cand = (glob.glob(os.path.join(args.data_path, cat, 'test_public', 'good', '*')) or
            glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*')))
    w0, h0 = Image.open(cand[0]).size                    # PIL: (W, H)
    S = args.short_side
    if h0 <= w0:
        hs, ws = S, round(S * w0 / h0)
    else:
        ws, hs = S, round(S * h0 / w0)
    r14 = lambda v: max(14, int(round(v / 14)) * 14)
    hs, ws = r14(hs), r14(ws)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    dtf = transforms.Compose([transforms.Resize((hs, ws)), transforms.ToTensor(),
                              transforms.Normalize(mean, std)])
    gtf = transforms.Compose([transforms.Resize((hs, ws)), transforms.ToTensor()])
    return dtf, gtf, hs // 14, ws // 14

warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def get_arch(enc):
    for a in ('small', 'base', 'large'):
        if a in enc:
            return a
    raise ValueError(enc)


@torch.no_grad()
def extract_feature(encoder, img, layers, n_reg):
    x = encoder.prepare_tokens(img)
    feats, last = [], max(layers)
    for i, blk in enumerate(encoder.blocks):
        if i <= last:
            x = blk(x)
        if i in layers:
            feats.append(x[:, 1 + n_reg:, :])
    return torch.stack(feats, dim=1).mean(dim=1)        # [B, N, C]


def build_bank(feats, bank_size, device, seed=0):
    # feats [Ntr, N, C] -> bank [<=bank_size, C] (random subsample, nhanh ở res cao)
    flat = feats.reshape(-1, feats.shape[-1])
    M = flat.shape[0]
    if M > bank_size:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(M, generator=g)[:bank_size]
        flat = flat[idx]
    return flat.to(device)


@torch.no_grad()
def nn_map(feats, bank, device, chunk=2048):
    B, N, C = feats.shape
    q = feats.reshape(B * N, C).to(device)
    out = torch.empty(B * N, device=device)
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s:s + chunk], bank)
        out[s:s + chunk] = d.min(dim=1)[0]
    return out.reshape(B, N)


def global_minmax(maps, lo=1.0, hi=99.0):
    a, b = np.percentile(maps, lo), np.percentile(maps, hi)
    if b - a < 1e-12:
        return np.clip(maps, 0, 1)
    return np.clip((maps - a) / (b - a), 0, 1)


def sp_score(m, max_ratio=0.01):
    flat = m.reshape(-1)
    k = max(1, int(flat.shape[0] * max_ratio))
    return np.sort(flat)[::-1][:k].mean()


def morph_close(maps, k):
    if k <= 0:
        return maps
    from scipy import ndimage
    out = np.empty_like(maps)
    for i in range(maps.shape[0]):
        out[i] = ndimage.grey_closing(maps[i], size=(k, k))
    return out


def evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn):
    dtf, gtf, hp, wp = build_transforms(args, cat)
    if args.aspect_preserve:
        print_fn(f'  [{cat}] aspect-preserve grid = {hp}x{wp} patch')
    DS = MVTecAD2DatasetAR if args.aspect_preserve else MVTecAD2Dataset
    train_data = ImageFolder(root=os.path.join(args.data_path, cat, 'train'), transform=dtf)
    test_data = DS(root=os.path.join(args.data_path, cat),
                   transform=dtf, gt_transform=gtf, phase='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.mode == 'multi':
        layer_groups = {'fine': args.fine_layers, 'coarse': args.coarse_layers}
    else:
        layer_groups = {'all': sorted(set(args.fine_layers + args.coarse_layers))}

    # 1) Cache train features per scale + build banks
    # Subsample patch NGAY khi trích -> chặn RAM (ảnh rộng res cao có thể >12GB nếu giữ hết).
    print_fn(f'  [{cat}] cache train ({len(train_data)} ảnh)...')
    keep = max(256, (args.bank_size * 4) // max(1, len(train_data)))   # patch/ảnh giữ lại
    cache = {g: [] for g in layer_groups}
    with torch.no_grad():
        for img, _ in tqdm(train_loader, ncols=80, desc='  train-feat'):
            img = img.to(device)
            for g, lays in layer_groups.items():
                f = extract_feature(encoder, img, lays, n_reg)         # [B,N,C]
                flat = f.reshape(-1, f.shape[-1])
                cap = img.shape[0] * keep
                if flat.shape[0] > cap:
                    sel = torch.randperm(flat.shape[0], device=flat.device)[:cap]
                    flat = flat[sel]
                cache[g].append(flat.cpu())
    banks = {g: build_bank(torch.cat(cache[g], 0), args.bank_size, device) for g in layer_groups}
    for g in banks:
        print_fn(f'    bank[{g}] = {tuple(banks[g].shape)}')

    # 2) Test: NN map mỗi scale, gộp
    scale_maps = {g: [] for g in layer_groups}
    gt_maps = []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(test_loader, ncols=80, desc='  test'):
            img = img.to(device)
            for g, lays in layer_groups.items():
                feats = extract_feature(encoder, img, lays, n_reg)
                s = nn_map(feats, banks[g], device).reshape(feats.shape[0], 1, hp, wp)
                s = F.interpolate(s, size=args.resize_mask, mode='bilinear', align_corners=False)
                s = gk(s)
                scale_maps[g].append(s[:, 0].cpu().numpy())
            gt = F.interpolate(gt, size=args.resize_mask, mode='nearest')
            gt[gt > 0.5] = 1; gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            gt_maps.append(gt[:, 0].cpu().numpy())

    gt_arr = np.concatenate(gt_maps, 0).astype(np.uint8)
    # chuẩn hoá mỗi scale rồi trung bình
    fused = None
    for g in layer_groups:
        m = global_minmax(np.concatenate(scale_maps[g], 0))
        fused = m if fused is None else fused + m
    fused = fused / len(layer_groups)
    fused = morph_close(fused, args.morph_close)

    pr_sp = np.array([sp_score(m, args.max_ratio) for m in fused])
    gt_sp = np.array([1 if x.sum() > 0 else 0 for x in gt_arr])
    r = ader_evaluator(fused, pr_sp, gt_arr, gt_sp, use_metrics=METRIC_NAMES)
    print_fn(f'  === {cat.upper()} ({args.mode}) === ' +
             ' '.join(f'{n}:{v:.4f}' for n, v in zip(METRIC_NAMES, r)))
    return r


def main():
    ap = argparse.ArgumentParser('Multi-scale high-res frozen-DINOv2 distance')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--input_size', type=int, default=672, help='phải chia hết 14 (672=14*48)')
    ap.add_argument('--crop_size', type=int, default=672, help='phải chia hết 14')
    ap.add_argument('--resize_mask', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--max_ratio', type=float, default=0.01)
    ap.add_argument('--mode', type=str, default='multi', choices=['multi', 'single'])
    ap.add_argument('--fine_layers', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--coarse_layers', type=int, nargs='+', default=[6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000, help='cap memory bank (random subsample)')
    ap.add_argument('--morph_close', type=int, default=0, help='kernel morphological closing (0=off)')
    ap.add_argument('--aspect_preserve', action='store_true',
                    help='giữ tỉ lệ ảnh (resize cạnh ngắn), tốt cho ảnh rộng can/sheet_metal')
    ap.add_argument('--short_side', type=int, default=672, help='cạnh ngắn khi --aspect_preserve (chia hết 14)')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    ap.add_argument('--out_dir', type=str, default='./diagnosis_hires')
    args = ap.parse_args()

    if args.crop_size % 14 != 0:
        raise SystemExit(f'crop_size={args.crop_size} phải chia hết 14 (vd 672=14*48, 630=14*45).')
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA không khả dụng (adeval cần GPU). Cài torch khớp driver: "
                         "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    device = 'cuda:0'

    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger(f'hires_{args.mode}', args.out_dir)
    print_fn = logger.info
    print_fn('=' * 70)
    print_fn(f'MULTI-SCALE HI-RES | enc={args.encoder} | res={args.crop_size} (lưới {args.crop_size//14}x{args.crop_size//14}) '
             f'| mode={args.mode} | bank={args.bank_size} | morph={args.morph_close}')
    if args.mode == 'multi':
        print_fn(f'fine={args.fine_layers}  coarse={args.coarse_layers}')
    print_fn('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_res = {}
    for cat in args.categories:
        all_res[cat] = evaluate_category(args, device, cat, encoder, n_reg, gk, print_fn)

    print_fn('\n' + '=' * 70)
    mean_r = np.array(list(all_res.values())).mean(0)
    print_fn('{:<12} '.format('') + ' '.join(f'{m:>10}' for m in METRIC_NAMES))
    print_fn('{:<12} '.format(f'MEAN[{args.mode}]') + ' '.join(f'{v:>10.4f}' for v in mean_r))

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('category,' + ','.join(METRIC_NAMES) + '\n')
        for cat, r in all_res.items():
            f.write(f'{cat},' + ','.join(f'{v:.4f}' for v in r) + '\n')
        f.write('MEAN,' + ','.join(f'{v:.4f}' for v in mean_r) + '\n')
    print_fn(f'\nĐã lưu: {csv}')


if __name__ == '__main__':
    main()
