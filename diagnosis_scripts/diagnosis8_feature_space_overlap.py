# diagnosis8_feature_space_overlap.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
import math
from torch.nn import functional as F

from dataset import MVTecAD2Dataset, get_data_transforms
from torchvision.datasets import ImageFolder
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

import warnings
warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']


def load_model(args, device):
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
        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP
    )
    ckpt = os.path.join(args.ckpt_dir, args.item, 'model.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    return model.to(device).eval()


def extract_image_features(model, dataloader, device, max_images=200):
    """
    Extract mean-pooled encoder features per image.
    Returns features [N, C], labels [N]
    """
    features, labels = [], []
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=80):
            if len(batch) == 2:
                img, label = batch
                label = torch.zeros(img.shape[0]).long()
            else:
                img, gt, label, _ = batch

            img = img.to(device)
            x = model.encoder.prepare_tokens(img)
            for i, blk in enumerate(model.encoder.blocks):
                if i <= model.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
            # Remove class + register tokens, mean pool
            x = x[:, 1 + model.encoder.num_register_tokens:, :]  # [B, N, C]
            feat = x.mean(dim=1)  # [B, C]
            features.append(feat.cpu())
            labels.append(label.flatten().cpu())
            count += img.shape[0]
            if count >= max_images:
                break

    return torch.cat(features, dim=0)[:max_images], \
           torch.cat(labels,   dim=0)[:max_images]


def compute_overlap_metrics(train_feat, test_normal_feat, test_defect_feat, log_lines):
    """
    Measure overlap between feature distributions using:
    1. Mean cosine distance between train normal and test defect centroids
    2. % of defect features within convex hull of normal features (approximated by PCA distance)
    """
    train_mean   = F.normalize(train_feat.mean(dim=0, keepdim=True), dim=-1)
    normal_mean  = F.normalize(test_normal_feat.mean(dim=0, keepdim=True), dim=-1)
    defect_mean  = F.normalize(test_defect_feat.mean(dim=0, keepdim=True), dim=-1) \
                   if test_defect_feat is not None else None

    train_normal_sim  = (train_mean * normal_mean).sum().item()
    msg = f'  Train-normal vs Test-normal centroid sim : {train_normal_sim:.4f}'
    print(msg); log_lines.append(msg)

    if defect_mean is not None:
        train_defect_sim = (train_mean * defect_mean).sum().item()
        normal_defect_sim = (normal_mean * defect_mean).sum().item()
        msg2 = (f'  Train-normal vs Test-defect centroid sim : {train_defect_sim:.4f}\n'
                f'  Test-normal  vs Test-defect centroid sim : {normal_defect_sim:.4f}')
        if normal_defect_sim > 0.98:
            msg2 += '  ⚠ NEAR IDENTICAL — defect indistinguishable in feature space'
        print(msg2); log_lines.append(msg2)

        # PCA-based overlap: project all onto 2D, measure distance between clouds
        all_feat = torch.cat([train_feat, test_normal_feat, test_defect_feat], dim=0).numpy()
        n_tr = len(train_feat); n_no = len(test_normal_feat); n_de = len(test_defect_feat)
        pca  = PCA(n_components=50)
        all_2d = pca.fit_transform(all_feat)

        tr_2d = all_2d[:n_tr]
        no_2d = all_2d[n_tr:n_tr+n_no]
        de_2d = all_2d[n_tr+n_no:]

        # % defect points within normal std radius
        no_mean, no_std = no_2d.mean(axis=0), no_2d.std(axis=0)
        de_dist = np.abs(de_2d - no_mean) / (no_std + 1e-8)
        within = (de_dist < 2.0).all(axis=1).mean() * 100
        msg3 = f'  Defect points within 2σ of normal distribution (PCA50): {within:.1f}%'
        if within > 60:
            msg3 += '  ⚠ HIGH OVERLAP'
        print(msg3); log_lines.append(msg3)
        return tr_2d, no_2d, de_2d
    return None, None, None


def plot_feature_space(tr_2d, no_2d, de_2d, cat, out_dir):
    if tr_2d is None:
        return
    # Project to 2D for visualization
    all_2d = np.concatenate([tr_2d, no_2d, de_2d], axis=0)
    pca2   = PCA(n_components=2)
    all_vis = pca2.fit_transform(all_2d)

    n_tr = len(tr_2d); n_no = len(no_2d)
    tr_v  = all_vis[:n_tr]
    no_v  = all_vis[n_tr:n_tr+n_no]
    de_v  = all_vis[n_tr+n_no:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(tr_v[:, 0], tr_v[:, 1], c='gray',      alpha=0.3, s=10, label='Train normal')
    ax.scatter(no_v[:, 0], no_v[:, 1], c='steelblue', alpha=0.5, s=15, label='Test normal')
    ax.scatter(de_v[:, 0], de_v[:, 1], c='tomato',    alpha=0.6, s=15, label='Test defect')
    ax.set_title(f'[{cat}] Encoder feature space (PCA 2D)')
    ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{cat}_feature_space.png'), dpi=120)
    plt.close()


def run_category(args, cat, device, log_lines):
    args.item = cat
    ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
    if not os.path.exists(ckpt):
        msg = f'[SKIP] {cat}'; print(msg); log_lines.append(msg); return

    print(f'\n=== {cat.upper()} ==='); log_lines.append(f'\n=== {cat.upper()} ===')

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # Train loader (normal only)
    train_data = ImageFolder(
        root=os.path.join(args.data_path, cat, 'train'),
        transform=data_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=8, shuffle=False, num_workers=2)

    # Test loader
    test_data = MVTecAD2Dataset(
        root=os.path.join(args.data_path, cat),
        transform=data_transform, gt_transform=gt_transform, phase='test'
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2)

    model = load_model(args, device)

    print('  Extracting train features...')
    train_feat, _       = extract_image_features(model, train_loader, device, max_images=300)
    print('  Extracting test features...')
    test_feat, test_lbl = extract_image_features(model, test_loader,  device, max_images=500)

    test_normal_feat = test_feat[test_lbl == 0]
    test_defect_feat = test_feat[test_lbl == 1] if (test_lbl == 1).sum() > 0 else None

    out_cat = os.path.join(args.out_dir, cat)
    tr_2d, no_2d, de_2d = compute_overlap_metrics(
        train_feat, test_normal_feat, test_defect_feat, log_lines)
    plot_feature_space(tr_2d, no_2d, de_2d, cat, out_cat)


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = ['Diagnosis 8 — Normal Training Data Diversity & Feature Space Overlap']

    for cat in VALID_CATEGORIES:
        run_category(args, cat, device, log_lines)

    log_path = os.path.join(args.out_dir, 'diagnosis8_log.txt')
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))
    print(f'\nLog saved to: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',   type=str, default='./reproduced_results')
    parser.add_argument('--out_dir',    type=str, default='./diagnosis/diagnosis8_feature_space')
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)
    args = parser.parse_args()
    main(args)