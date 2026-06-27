# diagnosis9_discriminative_direction.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
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


def extract_features_multilevel(model, dataloader, device, max_images=300):
    """
    Extract features at multiple levels:
    1. Image-level: mean pool of encoder output
    2. Patch-level: all patch features (for local analysis)
    3. Per-layer encoder features (to find which layer is most discriminative)
    """
    img_feats   = []   # [N, C] mean pooled
    labels      = []
    layer_feats = {i: [] for i in model.target_layers}  # per encoder layer

    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=80):
            if len(batch) == 2:
                img, lbl = batch
                lbl = torch.zeros(img.shape[0]).long()
            else:
                img, gt, lbl, _ = batch

            img = img.to(device)
            B   = img.shape[0]

            # Forward through encoder, capture per-layer features
            x = model.encoder.prepare_tokens(img)
            for i, blk in enumerate(model.encoder.blocks):
                if i <= model.target_layers[-1]:
                    with torch.no_grad():
                        x = blk(x)
                else:
                    continue
                if i in model.target_layers:
                    # Remove class + register tokens, mean pool
                    feat = x[:, 1 + model.encoder.num_register_tokens:, :]
                    layer_feats[i].append(feat.mean(dim=1).cpu())  # [B, C]

            img_feats.append(layer_feats[model.target_layers[-1]][-1])
            labels.append(lbl.flatten().cpu())
            count += B
            if count >= max_images:
                break

    img_feats = torch.cat(img_feats, dim=0)[:max_images]
    labels    = torch.cat(labels,    dim=0)[:max_images]
    layer_feats_cat = {
        i: torch.cat(v, dim=0)[:max_images]
        for i, v in layer_feats.items()
    }
    return img_feats, labels, layer_feats_cat


def analyze_discriminative_direction(feats, labels, log_lines, prefix=''):
    """
    1. LDA to find most discriminative direction
    2. Project onto LDA direction, compute AUROC
    3. PCA variance analysis: how much variance do normal/defect differ in top PCs
    """
    normal_feats = feats[labels == 0].numpy()
    defect_feats = feats[labels == 1].numpy() if (labels == 1).sum() > 0 else None

    if defect_feats is None or len(defect_feats) < 2:
        msg = f'  {prefix}Not enough defect samples'
        print(msg); log_lines.append(msg)
        return None, None

    all_feats = np.concatenate([normal_feats, defect_feats], axis=0)
    all_labels = np.array([0] * len(normal_feats) + [1] * len(defect_feats))

    # PCA first to reduce dim (LDA needs n_samples > n_features)
    n_components = min(50, all_feats.shape[0] - 1, all_feats.shape[1])
    pca = PCA(n_components=n_components)
    all_pca = pca.fit_transform(all_feats)

    normal_pca = all_pca[:len(normal_feats)]
    defect_pca = all_pca[len(normal_feats):]

    # Per-PC separability: t-stat between normal and defect
    pc_tscore = []
    for pc in range(n_components):
        n_vals = normal_pca[:, pc]
        d_vals = defect_pca[:, pc]
        pooled_std = np.sqrt((n_vals.std()**2 + d_vals.std()**2) / 2 + 1e-8)
        t = abs(n_vals.mean() - d_vals.mean()) / pooled_std
        pc_tscore.append(t)

    top5_pcs   = np.argsort(pc_tscore)[::-1][:5]
    top5_scores = [pc_tscore[i] for i in top5_pcs]

    msg = (f'  {prefix}Top-5 discriminative PCs: {list(top5_pcs)} '
           f'| t-scores: {[f"{s:.3f}" for s in top5_scores]}')
    print(msg); log_lines.append(msg)

    # LDA on PCA features
    try:
        lda = LDA(n_components=1)
        lda.fit(all_pca, all_labels)
        lda_proj = lda.transform(all_pca).flatten()
        lda_normal = lda_proj[:len(normal_feats)]
        lda_defect = lda_proj[len(normal_feats):]

        auroc = roc_auc_score(all_labels, lda_proj)
        gap   = abs(lda_defect.mean() - lda_normal.mean())
        msg2  = (f'  {prefix}LDA direction AUROC: {auroc:.4f}  '
                 f'gap={gap:.4f}  '
                 f'{"✓ SEPARABLE" if auroc > 0.7 else "⚠ NOT SEPARABLE"}')
        print(msg2); log_lines.append(msg2)
        return lda_proj, all_labels, pca, pc_tscore, top5_pcs

    except Exception as e:
        msg = f'  {prefix}LDA failed: {e}'
        print(msg); log_lines.append(msg)
        return None, None, pca, pc_tscore, top5_pcs


def plot_results(lda_proj, all_labels, pc_tscore, top5_pcs, cat, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: LDA projection distribution
    if lda_proj is not None:
        ax = axes[0]
        normal_proj = lda_proj[all_labels == 0]
        defect_proj = lda_proj[all_labels == 1]
        ax.hist(normal_proj, bins=30, alpha=0.6, color='steelblue', label='Normal')
        ax.hist(defect_proj, bins=30, alpha=0.6, color='tomato',    label='Defect')
        ax.set_title(f'[{cat}] LDA projection distribution')
        ax.set_xlabel('LDA score')
        ax.legend()

    # Plot 2: Per-PC t-score (top 20)
    ax2 = axes[1]
    top20 = np.argsort(pc_tscore)[::-1][:20]
    ax2.bar(range(20), [pc_tscore[i] for i in top20],
            color=['tomato' if i in top5_pcs else 'steelblue' for i in top20])
    ax2.set_xticks(range(20))
    ax2.set_xticklabels([f'PC{i}' for i in top20], rotation=45, fontsize=7)
    ax2.set_title(f'[{cat}] Per-PC discriminability (t-score)')
    ax2.set_ylabel('t-score')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{cat}_discriminative_direction.png'), dpi=120)
    plt.close()


def run_category(args, cat, device, log_lines):
    args.item = cat
    ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
    if not os.path.exists(ckpt):
        msg = f'[SKIP] {cat}'; print(msg); log_lines.append(msg); return

    print(f'\n=== {cat.upper()} ==='); log_lines.append(f'\n=== {cat.upper()} ===')

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    test_data = MVTecAD2Dataset(
        root=os.path.join(args.data_path, cat),
        transform=data_transform, gt_transform=gt_transform, phase='test'
    )
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2)

    model = load_model(args, device)
    feats, labels, layer_feats = extract_features_multilevel(model, loader, device)

    out_cat = os.path.join(args.out_dir, cat)

    # Analyze per encoder layer
    msg = '  --- Per encoder layer LDA AUROC ---'
    print(msg); log_lines.append(msg)

    best_auroc = 0
    best_layer = -1
    layer_aurocs = {}

    for layer_idx in sorted(layer_feats.keys()):
        lf = layer_feats[layer_idx]
        normal_f = lf[labels == 0].numpy()
        defect_f = lf[labels == 1].numpy() if (labels == 1).sum() > 0 else None
        if defect_f is None or len(defect_f) < 2:
            continue

        all_f  = np.concatenate([normal_f, defect_f], axis=0)
        all_lb = np.array([0]*len(normal_f) + [1]*len(defect_f))

        n_comp = min(50, all_f.shape[0]-1, all_f.shape[1])
        pca    = PCA(n_components=n_comp)
        all_pca = pca.fit_transform(all_f)

        try:
            lda  = LDA(n_components=1)
            lda.fit(all_pca, all_lb)
            proj = lda.transform(all_pca).flatten()
            auroc = roc_auc_score(all_lb, proj)
            layer_aurocs[layer_idx] = auroc
            verdict = '✓ BEST' if auroc == max(layer_aurocs.values()) else ''
            row = f'    Layer {layer_idx}: AUROC={auroc:.4f}  {verdict}'
            print(row); log_lines.append(row)
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer_idx
        except:
            pass

    msg = f'  → Most discriminative encoder layer: {best_layer} (AUROC={best_auroc:.4f})'
    print(msg); log_lines.append(msg)

    # Full analysis on last layer (image level)
    msg2 = '  --- Image-level feature analysis (last encoder layer) ---'
    print(msg2); log_lines.append(msg2)

    result = analyze_discriminative_direction(feats, labels, log_lines, prefix='  ')
    if result[0] is not None:
        lda_proj, all_labels, pca, pc_tscore, top5_pcs = result
        plot_results(lda_proj, all_labels, pc_tscore, top5_pcs, cat, out_cat)

    # Plot layer AUROC curve
    if layer_aurocs:
        fig, ax = plt.subplots(figsize=(8, 4))
        layers = sorted(layer_aurocs.keys())
        aurocs = [layer_aurocs[l] for l in layers]
        ax.plot(layers, aurocs, 'o-', color='steelblue')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.set_xticks(layers)
        ax.set_xlabel('Encoder layer')
        ax.set_ylabel('LDA AUROC')
        ax.set_title(f'[{cat}] Discriminability per encoder layer')
        ax.legend()
        plt.tight_layout()
        os.makedirs(out_cat, exist_ok=True)
        plt.savefig(os.path.join(out_cat, f'{cat}_layer_auroc.png'), dpi=120)
        plt.close()


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = ['Diagnosis 9 — Discriminative Direction Analysis']

    for cat in VALID_CATEGORIES:
        run_category(args, cat, device, log_lines)

    log_path = os.path.join(args.out_dir, 'diagnosis9_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'\nLog saved to: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',   type=str, default='./reproduced_results')
    parser.add_argument('--out_dir',    type=str,
                        default='./diagnosis/diagnosis9_discriminative_direction')
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)
    args = parser.parse_args()
    main(args)