# quick_test1_patch_lda.py
# Câu hỏi 1: Patch-level LDA AUROC trên encoder features

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
import math

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

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


def extract_patch_features_with_labels(model, dataloader, device, gt_transform, max_images=100):
    """
    Extract per-patch features + per-patch labels from GT mask.
    Returns:
        patch_feats: [N_patches_total, C]
        patch_labels: [N_patches_total] — 1 if patch contains defect, 0 otherwise
        img_feats: [N_images, C] — mean pooled
        img_labels: [N_images]
    """
    all_patch_feats  = []
    all_patch_labels = []
    all_img_feats    = []
    all_img_labels   = []
    count = 0

    with torch.no_grad():
        for img, gt, label, _ in tqdm(dataloader, ncols=80):
            img = img.to(device)
            B   = img.shape[0]

            # Extract fused encoder features
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
            B, N, C = fused.shape
            side = int(math.sqrt(N))

            # Per-patch label from GT mask
            # GT mask: [B, 1, H, W] → resize to [B, 1, side, side]
            import torch.nn.functional as F
            gt_patch = F.interpolate(gt.to(device), size=(side, side),
                                     mode='nearest').squeeze(1)  # [B, side, side]
            gt_patch = (gt_patch > 0.5).float().reshape(B, N)   # [B, N]

            all_patch_feats.append(fused.reshape(B * N, C).cpu())
            all_patch_labels.append(gt_patch.reshape(B * N).cpu())
            all_img_feats.append(fused.mean(dim=1).cpu())
            all_img_labels.append(label.flatten().cpu())

            count += B
            if count >= max_images:
                break

    return (torch.cat(all_patch_feats,  dim=0),
            torch.cat(all_patch_labels, dim=0),
            torch.cat(all_img_feats,    dim=0),
            torch.cat(all_img_labels,   dim=0))


def compute_lda_auroc(feats, labels, n_pca=50):
    normal_feats = feats[labels == 0].numpy()
    anomaly_feats = feats[labels == 1].numpy()

    if len(anomaly_feats) < 2 or len(normal_feats) < 2:
        return None

    all_feats  = np.concatenate([normal_feats, anomaly_feats], axis=0)
    all_labels = np.array([0] * len(normal_feats) + [1] * len(anomaly_feats))

    n_comp = min(n_pca, all_feats.shape[0] - 1, all_feats.shape[1])
    pca    = PCA(n_components=n_comp)
    all_pca = pca.fit_transform(all_feats)

    try:
        lda  = LDA(n_components=1)
        lda.fit(all_pca, all_labels)
        proj  = lda.transform(all_pca).flatten()
        auroc = roc_auc_score(all_labels, proj)
        return auroc
    except:
        return None


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_transform, gt_transform = get_data_transforms(
        args.input_size, args.crop_size)

    print(f'{"Category":<15} {"Patch-LDA AUROC":>16} {"Image-LDA AUROC":>16}')
    print('=' * 50)

    for cat in VALID_CATEGORIES:
        ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
        if not os.path.exists(ckpt):
            print(f'{cat:<15} SKIP')
            continue

        from dataset import MVTecAD2Dataset
        test_data = MVTecAD2Dataset(
            root=os.path.join(args.data_path, cat),
            transform=data_transform, gt_transform=gt_transform, phase='test')
        loader = torch.utils.data.DataLoader(
            test_data, batch_size=8, shuffle=False, num_workers=2)

        model = load_model(args, device, cat)
        patch_feats, patch_labels, img_feats, img_labels = \
            extract_patch_features_with_labels(
                model, loader, device, gt_transform, max_images=200)

        patch_auroc = compute_lda_auroc(patch_feats, patch_labels)
        img_auroc   = compute_lda_auroc(img_feats,   img_labels)

        print(f'{cat:<15} {patch_auroc:>16.4f} {img_auroc:>16.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',   type=str, default='./reproduced_results')
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)
    args = parser.parse_args()
    main(args)