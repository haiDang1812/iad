# quick_test2_mahalanobis.py
# Câu hỏi 2: Pure Mahalanobis scoring trên encoder features
# Không dùng reconstruction — fit multivariate Gaussian từ train features

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from functools import partial
from tqdm import tqdm
import math
from torch.nn import functional as F
from sklearn.decomposition import PCA
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import get_gaussian_kernel, ader_evaluator

import warnings
warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']


def load_model(args, device, cat):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim, num_heads = 768, 12

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
        use_local_contrast=False, use_ortho_loss=False,
    )
    ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return model.to(device).eval()


def extract_fused_features(model, dataloader, device, is_train=False):
    all_feats  = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=80):
            if is_train:
                img, _ = batch
                label  = torch.zeros(img.shape[0]).long()
            else:
                img, gt, label, _ = batch

            img = img.to(device)
            x   = model.encoder.prepare_tokens(img)
            en_list = []
            for i, blk in enumerate(model.encoder.blocks):
                if i <= model.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                if i in model.target_layers:
                    en_list.append(
                        x[:, 1 + model.encoder.num_register_tokens:, :])

            fused = model.fuse_feature(en_list)  # [B, N, C]
            all_feats.append(fused.cpu())
            all_labels.append(label.flatten().cpu())

    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def fit_gaussian(train_feats, n_pca=50):
    """
    train_feats: [N_images, N_patches, C]
    Fit PCA + per-patch mean/precision matrix
    Returns PCA components + per-position Gaussian params
    """
    N, P, C = train_feats.shape
    feats_flat = train_feats.reshape(N * P, C).numpy()

    n_comp = min(n_pca, feats_flat.shape[0] - 1, C)
    pca    = PCA(n_components=n_comp, whiten=False)
    proj   = pca.fit_transform(feats_flat)  # [N*P, n_comp]
    proj   = proj.reshape(N, P, n_comp)

    # Per-patch mean and variance
    mean = proj.mean(axis=0)  # [P, n_comp]
    std  = proj.std(axis=0)   # [P, n_comp]

    return {
        'components': torch.tensor(pca.components_, dtype=torch.float32),
        'mean_feat':  torch.tensor(feats_flat.mean(axis=0), dtype=torch.float32),
        'patch_mean': torch.tensor(mean, dtype=torch.float32),
        'patch_std':  torch.tensor(std,  dtype=torch.float32),
    }


def mahalanobis_anomaly_map(test_feats, gaussian, device, resize_mask=256):
    """
    test_feats: [B, N, C]
    Returns anomaly map [B, 1, resize_mask, resize_mask]
    """
    B, N, C = test_feats.shape
    side = int(math.sqrt(N))

    components = gaussian['components'].to(device)  # [n_comp, C]
    mean_feat  = gaussian['mean_feat'].to(device)   # [C]
    patch_mean = gaussian['patch_mean'].to(device)  # [N, n_comp]
    patch_std  = gaussian['patch_std'].to(device)   # [N, n_comp]

    # Center + project
    x    = test_feats.to(device) - mean_feat          # [B, N, C]
    proj = x @ components.T                            # [B, N, n_comp]

    # Mahalanobis per patch
    normalized = (proj - patch_mean) / (patch_std + 1e-8)  # [B, N, n_comp]
    score      = (normalized ** 2).mean(dim=-1)              # [B, N]

    # Reshape + resize
    score_map = score.reshape(B, 1, side, side)
    score_map = F.interpolate(score_map, size=resize_mask,
                              mode='bilinear', align_corners=False)
    return score_map


def evaluate_category(model, train_loader, test_loader, device,
                       n_pca=50, resize_mask=256, max_ratio=0.01):
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # Fit Gaussian from train features
    print('  Fitting Gaussian from train features...')
    train_feats, _ = extract_fused_features(model, train_loader, device, is_train=True)
    gaussian = fit_gaussian(train_feats, n_pca=n_pca)

    # Evaluate on test
    print('  Evaluating...')
    gt_list_px = []; pr_list_px = []
    gt_list_sp = []; pr_list_sp = []

    with torch.no_grad():
        for img, gt, label, _ in tqdm(test_loader, ncols=80):
            img = img.to(device)

            # Extract test features
            x = model.encoder.prepare_tokens(img)
            en_list = []
            for i, blk in enumerate(model.encoder.blocks):
                if i <= model.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                if i in model.target_layers:
                    en_list.append(
                        x[:, 1 + model.encoder.num_register_tokens:, :])
            fused = model.fuse_feature(en_list)  # [B, N, C]

            # Mahalanobis anomaly map
            anomaly_map = mahalanobis_anomaly_map(
                fused.cpu(), gaussian, device, resize_mask)
            anomaly_map = gaussian_kernel(anomaly_map)

            gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            gt[gt > 0.5] = 1; gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)
            gt_list_sp.append(label)

            flat     = anomaly_map.flatten(1)
            sp_score = torch.sort(flat, dim=1, descending=True)[0][
                       :, :int(flat.shape[1] * max_ratio)].mean(dim=1)
            pr_list_sp.append(sp_score)

    gt_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    pr_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    gt_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    return ader_evaluator(
        pr_px, pr_sp, gt_px, gt_sp,
        use_metrics=['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                     'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30'])


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_transform, gt_transform = get_data_transforms(
        args.input_size, args.crop_size)

    result_list = []
    for cat in VALID_CATEGORIES:
        ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
        if not os.path.exists(ckpt):
            print(f'[SKIP] {cat}'); continue

        print(f'\n=== {cat.upper()} ===')
        train_data = ImageFolder(
            root=os.path.join(args.data_path, cat, 'train'),
            transform=data_transform)
        test_data  = MVTecAD2Dataset(
            root=os.path.join(args.data_path, cat),
            transform=data_transform, gt_transform=gt_transform, phase='test')

        train_loader = DataLoader(train_data, batch_size=8,
                                  shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_data,  batch_size=8,
                                  shuffle=False, num_workers=2)

        model   = load_model(args, device, cat)
        results = evaluate_category(
            model, train_loader, test_loader, device,
            n_pca=args.n_pca)

        auroc_sp, ap_sp, f1_sp, \
            auroc_px, ap_px, f1_px, \
            aupro_px, aupro_005, aupro_030 = results

        print(f'{cat}: I-Auroc:{auroc_sp:.4f} '
              f'P-AUROC:{auroc_px:.4f} '
              f'P-AUPRO:{aupro_px:.4f} '
              f'AUPRO0.05:{aupro_005:.4f} '
              f'AUPRO0.30:{aupro_030:.4f}')
        result_list.append(results)

    if result_list:
        r = np.array(result_list)
        print('\n' + '='*50)
        print('MEAN:')
        print(f'I-AUROC:{r[:,0].mean():.4f} '
              f'P-AUROC:{r[:,3].mean():.4f} '
              f'AUPRO:{r[:,6].mean():.4f} '
              f'AUPRO0.05:{r[:,7].mean():.4f} '
              f'AUPRO0.30:{r[:,8].mean():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',   type=str, default='./reproduced_results')
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)
    parser.add_argument('--n_pca',      type=int, default=50)
    args = parser.parse_args()
    main(args)