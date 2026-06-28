# eval_dist_only.py
# -----------------------------------------------------------------------------
# DIST-only: chấm điểm bất thường THUẦN bằng khoảng cách trên frozen DINOv2,
# KHÔNG dùng reconstruction, KHÔNG cần checkpoint INP-Former (encoder pretrained).
# Đây là hiện thực hoá hướng AnomalyDINO/SuperAD: memory bank PatchCore + NN distance.
#
# Lý do: kết quả eval_fusion_distance cho thấy nhánh DIST (PatchCore-DINOv2) một mình
# đã vượt baseline reconstruction ở AUPRO0.05. Script này đẩy tiếp bằng encoder MẠNH HƠN
# (ViT-L/14) và cho chọn layer linh hoạt (công thức SuperAD: layer 6/12/18/24).
#
# Chạy:
#   # ViT-L (mặc định) — KHÔNG cần ckpt_dir
#   python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_large_14 \
#       --coreset_ratio 0.25 --out_dir ./diagnosis_distonly_vitl
#   # ViT-B để so sánh ngang với eval_fusion_distance
#   python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
#       --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --out_dir ./diagnosis_distonly_vitb
# -----------------------------------------------------------------------------

import os
import math
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from utils import ader_evaluator, get_gaussian_kernel, get_logger

warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']

METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']

# Layer mặc định theo kiến trúc (0-indexed). ViT-L 24 block -> ~6/12/18/24 (SuperAD).
DEFAULT_LAYERS = {
    'base':  [2, 3, 4, 5, 6, 7, 8, 9],
    'large': [5, 11, 17, 23],
}


def get_arch(encoder):
    if 'small' in encoder: return 'small'
    if 'base'  in encoder: return 'base'
    if 'large' in encoder: return 'large'
    raise ValueError(encoder)


@torch.no_grad()
def extract_feature(encoder, img, layers, n_reg):
    # img -> fused patch feature [B, N, C] (mean các layer chỉ định)
    x = encoder.prepare_tokens(img)
    feats = []
    last = max(layers)
    for i, blk in enumerate(encoder.blocks):
        if i <= last:
            x = blk(x)
        if i in layers:
            feats.append(x[:, 1 + n_reg:, :])   # bỏ cls + register tokens
    return torch.stack(feats, dim=1).mean(dim=1)  # [B, N, C]


# ---- Feature whitening (khai thác D9: tín hiệu nằm ở chiều phương sai THẤP) ----
# Chiếu lên trục PCA rồi chia cho sqrt(variance) -> khuếch đại chiều phương sai thấp.
# Giữ top-`dim` trục theo variance (bỏ đuôi nhiễu cực nhỏ); eps để ổn định.
def fit_whiten(train_feats, dim, eps):
    C = train_feats.shape[-1]
    X = train_feats.reshape(-1, C).numpy().astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)            # [C, C]
    eigval, eigvec = np.linalg.eigh(cov)          # tăng dần
    eigval = eigval[::-1]; eigvec = eigvec[:, ::-1]  # giảm dần (theo variance)
    k = min(dim, C)
    V = eigvec[:, :k]                             # [C, k]
    scale = 1.0 / np.sqrt(eigval[:k] + eps)       # [k] -> chiều variance nhỏ được nhân lớn
    W = (V * scale).astype(np.float32)            # [C, k]
    return (torch.tensor(mu, dtype=torch.float32), torch.tensor(W))


def whiten_apply(feats, whiten):
    # feats [B,N,C] -> [B,N,k]
    mu, W = whiten
    return (feats - mu.to(feats.device)) @ W.to(feats.device)


# ---- Foreground masking (nhắm object-có-nền: can/wallplugs/vial) ----
# Background prototype = trung bình các patch viền ảnh; patch nào GIỐNG viền -> nền -> hạ điểm.
# Tự no-op trên texture full-frame (viền ~ ruột -> không patch nào bị coi là nền rõ rệt).
@torch.no_grad()
def foreground_weight(feats, side, border=2, percentile=30, bg_w=0.1):
    B, N, C = feats.shape
    f = feats.reshape(B, side, side, C)
    bm = torch.zeros(side, side, dtype=torch.bool, device=feats.device)
    bm[:border, :] = 1; bm[-border:, :] = 1; bm[:, :border] = 1; bm[:, -border:] = 1
    bg = f[:, bm, :].mean(dim=1, keepdim=True)            # [B,1,C] prototype nền
    d = torch.norm(f.reshape(B, N, C) - bg, dim=-1)        # [B,N] xa nền = foreground
    thr = torch.quantile(d, percentile / 100.0, dim=1, keepdim=True)  # [B,1]
    w = torch.where(d >= thr, torch.ones_like(d), torch.full_like(d, bg_w))
    return w.reshape(B, 1, side, side)                     # [B,1,side,side]


# ---- PatchCore: memory bank + greedy coreset + NN distance ----
def fit_patchcore(train_feats, coreset_ratio, device, proj_dim=128):
    Ntr, N, C = train_feats.shape
    bank = train_feats.reshape(Ntr * N, C)
    M = bank.shape[0]
    target = max(1, int(M * coreset_ratio))

    g = torch.Generator().manual_seed(0)
    proj = torch.randn(C, min(proj_dim, C), generator=g)
    bank_p = (bank @ proj).to(device)

    selected = torch.zeros(M, dtype=torch.bool)
    selected[0] = True
    min_dist = torch.cdist(bank_p, bank_p[0:1]).squeeze(1)
    for _ in tqdm(range(target - 1), ncols=80, desc='  coreset'):
        idx = torch.argmax(min_dist).item()
        selected[idx] = True
        d = torch.cdist(bank_p, bank_p[idx:idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, d)
    return bank[selected].to(device)


# ---- Region-coherence (nhắm TRỰC TIẾP low-FPR / AUPRO0.05) ----
# AUPRO là metric THEO VÙNG. Defect thật liền khối -> lân cận cùng cao;
# spike giả lẻ tẻ -> lân cận thấp. Làm nổi vùng liền khối, dập spike đơn lẻ.
#   mult/gmean : nhân điểm với trung bình lân cận (mềm)
#   median/open: lọc median / grey-opening (cứng, xoá spike nhỏ)
# Áp ở lưới patch [B,1,side,side]. LƯU Ý: có thể hại defect cực nhỏ (1 patch).
@torch.no_grad()
def region_coherence(score_map, mode, k):
    if mode == 'none':
        return score_map
    if mode in ('mult', 'gmean'):
        local = F.avg_pool2d(score_map, kernel_size=k, stride=1, padding=k // 2)
        if mode == 'mult':
            return score_map * local
        return torch.sqrt(torch.clamp(score_map * local, min=0.0))
    if mode in ('median', 'open'):
        from scipy import ndimage
        arr = score_map.cpu().numpy()
        out = np.empty_like(arr)
        for b in range(arr.shape[0]):
            if mode == 'median':
                out[b, 0] = ndimage.median_filter(arr[b, 0], size=k)
            else:
                out[b, 0] = ndimage.grey_opening(arr[b, 0], size=(k, k))
        return torch.tensor(out, device=score_map.device)
    raise ValueError(mode)


@torch.no_grad()
def patchcore_map(feats, bank, device, chunk=4096):
    B, N, C = feats.shape
    q = feats.reshape(B * N, C).to(device)
    out = torch.empty(B * N, device=device)
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s:s + chunk], bank)
        out[s:s + chunk] = d.min(dim=1)[0]
    return out.reshape(B, N)


def sp_score_from_map(map_2d, max_ratio=0.01):
    flat = map_2d.reshape(-1)
    k = max(1, int(flat.shape[0] * max_ratio))
    return np.sort(flat)[::-1][:k].mean()


def evaluate_category(args, device, cat, encoder, layers, n_reg, gk, print_fn):
    use_fg = args.fg_mask and (cat in args.fg_categories)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    train_data = ImageFolder(root=os.path.join(args.data_path, cat, 'train'),
                             transform=data_transform)
    test_data  = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                                 transform=data_transform, gt_transform=gt_transform,
                                 phase='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Memory bank từ train normal
    print_fn(f'  [{cat}] trích feature train ({len(train_data)} ảnh)...')
    tf = []
    with torch.no_grad():
        for img, _ in tqdm(train_loader, ncols=80, desc='  train-feat'):
            tf.append(extract_feature(encoder, img.to(device), layers, n_reg).cpu())
    tf = torch.cat(tf, dim=0)

    # Whitening (tuỳ chọn) — fit trên train normal, áp cho cả bank lẫn query
    whiten = None
    if args.whiten == 'pca':
        print_fn(f'  [{cat}] fit whitening (dim={args.whiten_dim}, eps={args.whiten_eps})...')
        whiten = fit_whiten(tf, args.whiten_dim, args.whiten_eps)
        tf = whiten_apply(tf, whiten)

    bank = fit_patchcore(tf, args.coreset_ratio, device)

    # Test
    pr_maps, gt_maps = [], []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(test_loader, ncols=80, desc='  test'):
            img = img.to(device)
            feats_raw = extract_feature(encoder, img, layers, n_reg)
            feats = whiten_apply(feats_raw, whiten) if whiten is not None else feats_raw
            side = int(math.sqrt(feats.shape[1]))
            scores = patchcore_map(feats, bank, device)
            m = scores.reshape(scores.shape[0], 1, side, side)
            m = region_coherence(m, args.region_mode, args.region_k)
            if use_fg:
                w = foreground_weight(feats_raw, side, args.fg_border,
                                      args.fg_percentile, args.fg_bg_w)
                m = m * w
            m = F.interpolate(m, size=args.resize_mask, mode='bilinear', align_corners=False)
            m = gk(m)
            gt = F.interpolate(gt, size=args.resize_mask, mode='nearest')
            gt[gt > 0.5] = 1; gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            pr_maps.append(m[:, 0].cpu().numpy())
            gt_maps.append(gt[:, 0].cpu().numpy())

    pr_maps = np.concatenate(pr_maps, 0)
    gt_maps = np.concatenate(gt_maps, 0).astype(np.uint8)
    pr_sp = np.array([sp_score_from_map(m, args.max_ratio) for m in pr_maps])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt_maps])
    r = ader_evaluator(pr_maps, pr_sp, gt_maps, gt_sp, use_metrics=METRIC_NAMES)

    print_fn(f'  === {cat.upper()} ===')
    print_fn('  ' + ' '.join(f'{m:>11}' for m in METRIC_NAMES))
    print_fn('  ' + ' '.join(f'{v:>11.4f}' for v in r))
    return r


def main():
    parser = argparse.ArgumentParser('DIST-only PatchCore trên frozen DINOv2')
    parser.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--encoder',   type=str, default='dinov2reg_vit_large_14')
    parser.add_argument('--layers',    type=int, nargs='+', default=None,
                        help='layer dùng để fuse (0-indexed). None -> mặc định theo arch')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_ratio',  type=float, default=0.01)
    parser.add_argument('--coreset_ratio', type=float, default=0.25)
    parser.add_argument('--whiten', type=str, default='none', choices=['none', 'pca'],
                        help="pca = whiten feature (khuếch đại chiều phương sai thấp, D9)")
    parser.add_argument('--whiten_dim', type=int, default=768,
                        help='số trục PCA giữ lại khi whiten (cao=giữ chiều phương sai thấp)')
    parser.add_argument('--whiten_eps', type=float, default=1e-2,
                        help='hệ số ổn định, chặn over-amplify đuôi nhiễu')
    parser.add_argument('--fg_mask', action='store_true',
                        help='bật foreground masking (chỉ áp cho --fg_categories)')
    parser.add_argument('--fg_categories', type=str, nargs='+',
                        default=['can', 'wallplugs', 'vial'],
                        help='category object-có-nền để áp fg-mask')
    parser.add_argument('--fg_percentile', type=float, default=30,
                        help='%% patch giống nền nhất bị hạ điểm')
    parser.add_argument('--fg_border', type=int, default=2,
                        help='độ rộng viền dùng làm prototype nền')
    parser.add_argument('--fg_bg_w', type=float, default=0.1,
                        help='trọng số giữ lại cho vùng nền (0=xoá hẳn)')
    parser.add_argument('--region_mode', type=str, default='none',
                        choices=['none', 'mult', 'gmean', 'median', 'open'],
                        help='region-coherence post-proc nhắm low-FPR (áp ở lưới patch)')
    parser.add_argument('--region_k', type=int, default=3,
                        help='kích thước cửa sổ lân cận cho region-coherence')
    parser.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    parser.add_argument('--out_dir', type=str, default='./diagnosis_distonly')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    arch = get_arch(args.encoder)
    layers = args.layers if args.layers is not None else DEFAULT_LAYERS[arch]

    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger(f'distonly_{arch}', args.out_dir)
    print_fn = logger.info

    print_fn('=' * 70)
    print_fn(f'DIST-ONLY | encoder={args.encoder} | layers={layers} '
             f'| coreset={args.coreset_ratio} | whiten={args.whiten}'
             + (f'(dim={args.whiten_dim},eps={args.whiten_eps})' if args.whiten == 'pca' else '')
             + f' | region={args.region_mode}(k={args.region_k})'
             + (f' | fg_mask={args.fg_categories}' if args.fg_mask else ''))
    print_fn(f'data_path={args.data_path}')
    print_fn('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_res = {}
    for cat in args.categories:
        all_res[cat] = evaluate_category(args, device, cat, encoder, layers, n_reg, gk, print_fn)

    print_fn('\n' + '=' * 70)
    print_fn('MEAN ACROSS CATEGORIES')
    print_fn('=' * 70)
    mean_r = np.array(list(all_res.values())).mean(0)
    print_fn('{:<12} '.format('') + ' '.join(f'{m:>11}' for m in METRIC_NAMES))
    print_fn('{:<12} '.format('DIST-only') + ' '.join(f'{v:>11.4f}' for v in mean_r))

    csv_path = os.path.join(args.out_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('category,' + ','.join(METRIC_NAMES) + '\n')
        for cat, r in all_res.items():
            f.write(f'{cat},' + ','.join(f'{v:.4f}' for v in r) + '\n')
        f.write('MEAN,' + ','.join(f'{v:.4f}' for v in mean_r) + '\n')
    print_fn(f'\nĐã lưu: {csv_path}')


if __name__ == '__main__':
    main()
