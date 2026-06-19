# eval_fusion_distance.py
# -----------------------------------------------------------------------------
# Mục tiêu: kiểm chứng giả thuyết từ diagnosis (D7: residual mù, D9: frozen
# feature tách được) + research (PatchCore/AnomalyDINO/SuperAD).
#
# Script này KHÔNG train. Nó load checkpoint INP-Former đã có và đánh giá 3 nhánh
# chấm điểm trên CÙNG một lần chạy để so sánh trực tiếp:
#   (1) RECON   : anomaly map reconstruction gốc của INP-Former (1 - cos(en, de))
#   (2) DIST    : anomaly map khoảng cách trên frozen encoder feature
#                 - scorer='maha'     : PaDiM-style per-position Gaussian (rẻ)
#                 - scorer='patchcore': memory bank + coreset + NN distance
#   (3) FUSION  : alpha*norm(RECON) + (1-alpha)*norm(DIST), quét nhiều alpha
#
# Mỗi nhánh in đủ 9 metric (gồm P-F1_max ~ SegF1, AUPRO, AUPRO0.05, AUPRO0.30).
#
# Chạy (trên GPU server):
#   python eval_fusion_distance.py --data_path /path/to/mvtecad2 \
#          --ckpt_dir ./reproduced_results --scorer maha
# -----------------------------------------------------------------------------

import os
import math
import argparse
import warnings
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import cal_anomaly_maps, ader_evaluator, get_gaussian_kernel

warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']

METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1', 'P-AUROC', 'P-AP',
                'P-F1(SegF1)', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


# ----------------------------------------------------------------------------
# Model loading (giống quick_test*, base ViT-B/14)
# ----------------------------------------------------------------------------
def load_model(args, device, cat):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("encoder phải là small/base/large")

    encoder = vit_encoder.load(args.encoder)
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])
    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder, target_layers=target_layers,
        remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP,
    )
    ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return model.to(device).eval()


# ----------------------------------------------------------------------------
# Frozen encoder feature extraction (fuse toàn bộ target_layers -> [B, N, C])
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_fused_feature(model, img):
    x = model.encoder.prepare_tokens(img)
    en_list = []
    for i, blk in enumerate(model.encoder.blocks):
        if i <= model.target_layers[-1]:
            x = blk(x)
        if i in model.target_layers:
            en_list.append(x[:, 1 + model.encoder.num_register_tokens:, :])
    fused = model.fuse_feature(en_list)          # [B, N, C]
    return fused


# ----------------------------------------------------------------------------
# DIST scorer 1: PaDiM-style per-position Gaussian (diagonal, trên không gian PCA)
# ----------------------------------------------------------------------------
def fit_padim(train_feats, n_pca=50):
    # train_feats: [Ntrain, N, C]
    Ntr, N, C = train_feats.shape
    flat = train_feats.reshape(Ntr * N, C).numpy()
    from sklearn.decomposition import PCA
    n_comp = min(n_pca, flat.shape[0] - 1, C)
    pca = PCA(n_components=n_comp, whiten=False)
    proj = pca.fit_transform(flat).reshape(Ntr, N, n_comp)
    return {
        'components': torch.tensor(pca.components_, dtype=torch.float32),  # [n_comp, C]
        'mean_feat':  torch.tensor(flat.mean(0), dtype=torch.float32),     # [C]
        'patch_mean': torch.tensor(proj.mean(0), dtype=torch.float32),     # [N, n_comp]
        'patch_std':  torch.tensor(proj.std(0),  dtype=torch.float32),     # [N, n_comp]
    }


@torch.no_grad()
def padim_map(feats, g, device):
    # feats: [B, N, C] -> [B, N] (squared Mahalanobis diagonal trung bình)
    comp = g['components'].to(device)
    mf   = g['mean_feat'].to(device)
    pm   = g['patch_mean'].to(device)
    ps   = g['patch_std'].to(device)
    x = feats.to(device) - mf
    proj = x @ comp.T
    norm = (proj - pm) / (ps + 1e-8)
    return (norm ** 2).mean(-1)                  # [B, N]


# ----------------------------------------------------------------------------
# DIST scorer 2: PatchCore (memory bank + greedy coreset + NN distance)
# ----------------------------------------------------------------------------
def fit_patchcore(train_feats, coreset_ratio=0.10, device='cuda', proj_dim=128):
    # train_feats: [Ntrain, N, C] -> memory bank [M, C]
    Ntr, N, C = train_feats.shape
    bank = train_feats.reshape(Ntr * N, C)
    M = bank.shape[0]
    target = max(1, int(M * coreset_ratio))

    # Greedy k-center coreset trên random projection (theo PatchCore)
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(C, min(proj_dim, C), generator=g)
    bank_p = (bank @ proj).to(device)            # [M, proj_dim]

    selected = torch.zeros(M, dtype=torch.bool)
    start = 0
    selected[start] = True
    min_dist = torch.cdist(bank_p, bank_p[start:start + 1]).squeeze(1)  # [M]
    for _ in tqdm(range(target - 1), ncols=80, desc='  coreset'):
        idx = torch.argmax(min_dist).item()
        selected[idx] = True
        d = torch.cdist(bank_p, bank_p[idx:idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, d)
    return {'bank': bank[selected].to(device)}   # [target, C]


@torch.no_grad()
def patchcore_map(feats, mb, device, chunk=4096):
    # feats: [B, N, C] -> [B, N] NN-distance tới memory bank
    B, N, C = feats.shape
    q = feats.reshape(B * N, C).to(device)
    bank = mb['bank']
    out = torch.empty(B * N, device=device)
    for s in range(0, q.shape[0], chunk):
        d = torch.cdist(q[s:s + chunk], bank)    # [chunk, M]
        out[s:s + chunk] = d.min(dim=1)[0]
    return out.reshape(B, N)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def map_from_patch_scores(scores, side, resize_mask, gaussian_kernel, device):
    # scores: [B, N] -> [B, 1, resize_mask, resize_mask] đã smooth
    B, Npx = scores.shape
    m = scores.reshape(B, 1, side, side)
    m = F.interpolate(m, size=resize_mask, mode='bilinear', align_corners=False)
    m = gaussian_kernel(m)
    return m


def global_minmax(maps, p_lo=1.0, p_hi=99.0):
    # maps: np array [Ntest, H, W] -> chuẩn hoá robust về [0,1] dùng percentile
    lo = np.percentile(maps, p_lo)
    hi = np.percentile(maps, p_hi)
    if hi - lo < 1e-12:
        return np.clip(maps, 0, 1)
    return np.clip((maps - lo) / (hi - lo), 0, 1)


def sp_score_from_map(map_2d, max_ratio=0.01):
    # map_2d: [H, W] -> điểm ảnh (mean top max_ratio pixel)
    flat = map_2d.reshape(-1)
    k = max(1, int(flat.shape[0] * max_ratio))
    return np.sort(flat)[::-1][:k].mean()


def evaluate_maps(px_maps, gt_px, max_ratio=0.01):
    # px_maps, gt_px: np [Ntest, H, W]
    pr_sp = np.array([sp_score_from_map(m, max_ratio) for m in px_maps])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt_px])
    return ader_evaluator(px_maps, pr_sp, gt_px, gt_sp, use_metrics=METRIC_NAMES)


# ----------------------------------------------------------------------------
# Đánh giá 1 category: trả về dict {branch_name: [9 metrics]}
# ----------------------------------------------------------------------------
def evaluate_category(args, device, cat, print_fn):
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    train_data = ImageFolder(root=os.path.join(args.data_path, cat, 'train'),
                             transform=data_transform)
    test_data  = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                                 transform=data_transform, gt_transform=gt_transform,
                                 phase='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(args, device, cat)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # --- Fit DIST scorer từ train normal features ---
    print_fn(f'  [{cat}] trích feature train ({len(train_data)} ảnh) để fit {args.scorer}...')
    train_feats = []
    with torch.no_grad():
        for img, _ in tqdm(train_loader, ncols=80, desc='  train-feat'):
            train_feats.append(extract_fused_feature(model, img.to(device)).cpu())
    train_feats = torch.cat(train_feats, dim=0)   # [Ntr, N, C]

    if args.scorer == 'maha':
        scorer_state = fit_padim(train_feats, n_pca=args.n_pca)
        dist_fn = lambda f: padim_map(f, scorer_state, device)
    elif args.scorer == 'patchcore':
        scorer_state = fit_patchcore(train_feats, coreset_ratio=args.coreset_ratio, device=device)
        dist_fn = lambda f: patchcore_map(f, scorer_state, device)
    else:
        raise ValueError(args.scorer)

    # --- Pass qua test: thu thập RECON map và DIST map ---
    recon_maps, dist_maps, gt_maps = [], [], []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(test_loader, ncols=80, desc='  test'):
            img = img.to(device)

            # RECON branch (đúng pipeline gốc)
            out = model(img)
            en, de = out[0], out[1]
            amap_recon, _ = cal_anomaly_maps(en, de, img.shape[-1])
            amap_recon = F.interpolate(amap_recon, size=args.resize_mask,
                                       mode='bilinear', align_corners=False)
            amap_recon = gaussian_kernel(amap_recon)

            # DIST branch (frozen feature)
            feats = extract_fused_feature(model, img)     # [B, N, C]
            side = int(math.sqrt(feats.shape[1]))
            dist_scores = dist_fn(feats)                  # [B, N]
            amap_dist = map_from_patch_scores(dist_scores, side, args.resize_mask,
                                              gaussian_kernel, device)

            gt = F.interpolate(gt, size=args.resize_mask, mode='nearest')
            gt[gt > 0.5] = 1; gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            recon_maps.append(amap_recon[:, 0].cpu().numpy())
            dist_maps.append(amap_dist[:, 0].cpu().numpy())
            gt_maps.append(gt[:, 0].cpu().numpy())

    recon_maps = np.concatenate(recon_maps, 0)
    dist_maps  = np.concatenate(dist_maps, 0)
    gt_maps    = np.concatenate(gt_maps, 0).astype(np.uint8)

    # Chuẩn hoá robust mỗi nhánh để fuse
    recon_n = global_minmax(recon_maps)
    dist_n  = global_minmax(dist_maps)

    results = {}
    results['RECON'] = evaluate_maps(recon_maps, gt_maps, args.max_ratio)
    results['DIST']  = evaluate_maps(dist_maps,  gt_maps, args.max_ratio)

    # Quét alpha cho FUSION, chọn theo metric chính của benchmark MVTec AD 2:
    # AUPRO0.05 (METRIC_NAMES index 7). Đây là metric threshold-independent đầu bảng
    # theo paper dataset (Heckler-Kram et al., IJCV 2026, Sec 4.2/4.3).
    sel = args.select_idx
    best_alpha, best_fused, best_score = None, None, -1
    for alpha in args.alphas:
        fused = alpha * recon_n + (1 - alpha) * dist_n
        r = evaluate_maps(fused, gt_maps, args.max_ratio)
        if r[sel] > best_score:
            best_score, best_alpha, best_fused = r[sel], alpha, r
    results[f'FUSION(a={best_alpha})'] = best_fused

    # In bảng cho category
    print_fn(f'\n  === {cat.upper()} ===')
    print_fn('  {:<16} '.format('Branch') + ' '.join(f'{m:>11}' for m in METRIC_NAMES))
    for name, r in results.items():
        print_fn('  {:<16} '.format(name) + ' '.join(f'{v:>11.4f}' for v in r))
    return results


def main():
    parser = argparse.ArgumentParser('INP-Former + feature-distance fusion eval')
    parser.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',  type=str, default='./reproduced_results')
    parser.add_argument('--encoder',   type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument('--INP_num',    type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_ratio',  type=float, default=0.01)

    # DIST scorer
    parser.add_argument('--scorer', type=str, default='maha', choices=['maha', 'patchcore'])
    parser.add_argument('--n_pca',  type=int, default=50, help='PCA dim cho maha')
    parser.add_argument('--coreset_ratio', type=float, default=0.10, help='tỉ lệ coreset cho patchcore')

    # FUSION
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0],
                        help='trọng số RECON; 0=chỉ DIST, 1=chỉ RECON')
    parser.add_argument('--select_idx', type=int, default=7,
                        help='index metric để chọn alpha tốt nhất. '
                             '7=AUPRO0.05 (mặc định, metric chính MVTec AD 2), '
                             '5=P-F1, 6=AUPRO, 8=AUPRO0.30')

    parser.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn = print

    all_res = {}
    for cat in args.categories:
        if not os.path.exists(os.path.join(args.ckpt_dir, cat, 'model.pth')):
            print_fn(f'[SKIP] {cat}: không thấy checkpoint')
            continue
        all_res[cat] = evaluate_category(args, device, cat, print_fn)

    # Tổng hợp mean theo từng nhánh (gộp theo loại branch)
    print_fn('\n' + '=' * 70)
    print_fn('MEAN ACROSS CATEGORIES (theo loại nhánh)')
    print_fn('=' * 70)
    # Gom FUSION lại bất kể alpha
    agg = {}
    for cat, res in all_res.items():
        for name, r in res.items():
            key = 'FUSION' if name.startswith('FUSION') else name
            agg.setdefault(key, []).append(r)
    print_fn('{:<10} '.format('Branch') + ' '.join(f'{m:>11}' for m in METRIC_NAMES))
    for key, rows in agg.items():
        mean_r = np.array(rows).mean(0)
        print_fn('{:<10} '.format(key) + ' '.join(f'{v:>11.4f}' for v in mean_r))


if __name__ == '__main__':
    main()
