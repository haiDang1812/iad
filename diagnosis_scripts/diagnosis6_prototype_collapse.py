# diagnosis6_prototype_collapse.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import get_gaussian_kernel, cal_anomaly_maps
from torch.nn import functional as F

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


def get_prototypes(model, dataloader, device):
    """
    Collect aggregated prototype tokens per image.
    Returns:
        prototypes: [N_images, INP_num, embed_dim]
        labels:     [N_images]
    """
    all_prototypes = []
    all_labels     = []

    # Hook on aggregation block to capture agg_prototype after forward
    captured = {}
    def hook_fn(module, input, output):
        captured['prototype'] = output.detach().cpu()

    handle = model.aggregation[0].register_forward_hook(hook_fn)

    with torch.no_grad():
        for img, gt, label, _ in tqdm(dataloader, ncols=80):
            img = img.to(device)
            _ = model(img)
            all_prototypes.append(captured['prototype'])  # [B, INP_num, C]
            all_labels.append(label.flatten())

    handle.remove()
    return torch.cat(all_prototypes, dim=0), torch.cat(all_labels, dim=0)


def plot_prototype_pca(prototypes, labels, cat, out_dir, inp_num=6):
    """
    prototypes: [N, INP_num, C]
    PCA plot: each INP token as different marker, color = normal/defect
    """
    N, K, C = prototypes.shape
    proto_flat = prototypes.reshape(N * K, C).numpy()  # [N*K, C]

    pca = PCA(n_components=2)
    proto_2d = pca.fit_transform(proto_flat)  # [N*K, 2]
    proto_2d = proto_2d.reshape(N, K, 2)      # [N, K, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: color by normal/defect, marker by INP token
    ax = axes[0]
    markers = ['o', 's', '^', 'D', 'v', 'P']
    colors  = {0: 'steelblue', 1: 'tomato'}
    color_labels = {0: 'Normal', 1: 'Defect'}
    for k in range(K):
        for lbl in [0, 1]:
            mask = (labels == lbl).numpy()
            if mask.sum() == 0:
                continue
            ax.scatter(proto_2d[mask, k, 0], proto_2d[mask, k, 1],
                       c=colors[lbl], marker=markers[k % len(markers)],
                       alpha=0.4, s=15,
                       label=f'INP{k}-{color_labels[lbl]}' if k == 0 else f'INP{k}')
    ax.set_title(f'[{cat}] PCA — color=label, marker=INP token')
    ax.legend(fontsize=6, ncol=2)

    # Plot 2: mean prototype per class, with std ellipse
    ax2 = axes[1]
    for lbl in [0, 1]:
        mask = (labels == lbl).numpy()
        if mask.sum() == 0:
            continue
        for k in range(K):
            pts = proto_2d[mask, k, :]
            ax2.scatter(pts[:, 0].mean(), pts[:, 1].mean(),
                        c=colors[lbl], marker=markers[k % len(markers)],
                        s=100, edgecolors='black', linewidths=0.5)
            # std circle
            circle = plt.Circle((pts[:, 0].mean(), pts[:, 1].mean()),
                                 pts.std(), color=colors[lbl], fill=False,
                                 alpha=0.3, linestyle='--')
            ax2.add_patch(circle)
    ax2.set_title(f'[{cat}] Mean ± std per INP token & class')
    ax2.set_aspect('equal')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{cat}_prototype_pca.png'), dpi=120)
    plt.close()


def compute_collapse_metrics(prototypes, labels, inp_num, log_lines):
    """
    1. Pairwise cosine similarity between INP tokens (collapse metric)
    2. Cosine similarity between normal prototype mean and defect prototype mean
    """
    normal_mask  = (labels == 0)
    defect_mask  = (labels == 1)

    # Mean prototype per class [K, C]
    proto_normal = prototypes[normal_mask].mean(dim=0)  # [K, C]
    proto_defect = prototypes[defect_mask].mean(dim=0) if defect_mask.sum() > 0 else None

    # Pairwise sim between INP tokens (collapse)
    proto_n_norm = F.normalize(proto_normal, dim=-1)
    sim_matrix = (proto_n_norm @ proto_n_norm.T).numpy()
    upper = sim_matrix[np.triu_indices(inp_num, k=1)]
    msg = (f'  Prototype pairwise cosine sim (normal): '
           f'mean={upper.mean():.4f}  max={upper.max():.4f}  min={upper.min():.4f}')
    if upper.mean() > 0.9:
        msg += '  ⚠ COLLAPSE'
    print(msg); log_lines.append(msg)

    # Normal vs defect prototype similarity
    if proto_defect is not None:
        proto_d_norm = F.normalize(proto_defect, dim=-1)
        cross_sim = (proto_n_norm * proto_d_norm).sum(dim=-1).numpy()
        msg2 = (f'  Normal vs Defect prototype sim per INP: '
                f'mean={cross_sim.mean():.4f}  min={cross_sim.min():.4f}  max={cross_sim.max():.4f}')
        if cross_sim.mean() > 0.95:
            msg2 += '  ⚠ INDISTINGUISHABLE'
        print(msg2); log_lines.append(msg2)


def run_category(args, cat, device, log_lines):
    args.item = cat
    ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
    if not os.path.exists(ckpt):
        msg = f'[SKIP] {cat}'; print(msg); log_lines.append(msg); return

    print(f'\n=== {cat.upper()} ==='); log_lines.append(f'\n=== {cat.upper()} ===')

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    test_data = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                                transform=data_transform, gt_transform=gt_transform, phase='test')
    loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2)

    model = load_model(args, device)
    prototypes, labels = get_prototypes(model, loader, device)

    compute_collapse_metrics(prototypes, labels, args.INP_num, log_lines)
    plot_prototype_pca(prototypes, labels,
                       cat, os.path.join(args.out_dir, cat), args.INP_num)


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = ['Diagnosis 6 — Prototype Collapse in Feature Space']

    for cat in VALID_CATEGORIES:
        run_category(args, cat, device, log_lines)

    log_path = os.path.join(args.out_dir, 'diagnosis6_log.txt')
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))
    print(f'\nLog saved to: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',   type=str, default='./reproduced_results')
    parser.add_argument('--out_dir',    type=str, default='./diagnosis/diagnosis6_prototype_collapse')
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)
    args = parser.parse_args()
    main(args)