# diagnosis3_score_distribution.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from torch.nn import functional as F
import math

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import get_gaussian_kernel, cal_anomaly_maps

import warnings
warnings.filterwarnings("ignore")


def load_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(args.encoder)
    embed_dim, num_heads = 768, 12

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


def get_scores(model, dataloader, device, max_ratio=0.01, resize_mask=256):
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    scores, labels = [], []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(dataloader, ncols=80):
            img = img.to(device)
            en, de, _ = model(img)  # unpack g_loss
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            if resize_mask:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask,
                                            mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

            flat = anomaly_map.flatten(1)
            sp_score = torch.sort(flat, dim=1, descending=True)[0][
                       :, :int(flat.shape[1] * max_ratio)].mean(dim=1)

            scores.append(sp_score.cpu().numpy())
            labels.append(label.flatten().cpu().numpy())

    return np.concatenate(scores), np.concatenate(labels)


def plot_distribution(scores, labels, cat, out_dir):
    normal_scores = scores[labels == 0]
    defect_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_scores, bins=40, alpha=0.6, color='steelblue',
            label=f'Normal (n={len(normal_scores)})')
    ax.hist(defect_scores, bins=40, alpha=0.6, color='tomato',
            label=f'Defect (n={len(defect_scores)})')

    gap = defect_scores.mean() - normal_scores.mean() if len(defect_scores) else 0
    overlap_pct = (defect_scores < normal_scores.max()).mean() * 100 if len(defect_scores) else 0
    ax.set_title(f'[{cat}]  mean_gap={gap:.4f}  overlap={overlap_pct:.1f}%')
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{cat}_score_dist.png'), dpi=120)
    plt.close()
    return normal_scores, defect_scores


def classify_failure_mode(normal_s, defect_s):
    overlap_pct = (defect_s < normal_s.max()).mean() * 100
    gap = defect_s.mean() - normal_s.mean()
    normal_std = normal_s.std()
    defect_std = defect_s.std()

    if overlap_pct > 60:
        mode = 'A — Miss (overlap cao, model không phân biệt được)'
    elif normal_std > 0.5 * gap:
        mode = 'B — False Positive (normal score phân tán, FP cao)'
    else:
        mode = 'C — Mixed'
    return mode, gap, overlap_pct


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    out_dir = os.path.join(args.ckpt_dir, '..', 'diagnosis3_score_dist')
    out_dir = os.path.normpath(out_dir)

    VALID_CATEGORIES = {'can', 'fabric', 'fruit_jelly', 'rice',
                        'sheet_metal', 'vial', 'wallplugs', 'walnuts'}

    summary = []
    for cat in sorted(VALID_CATEGORIES):
        ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
        if not os.path.exists(ckpt):
            print(f'[SKIP] {cat} — checkpoint not found at {ckpt}')
            continue

        print(f'\n=== {cat.upper()} ===')
        args.item = cat
        test_data = MVTecAD2Dataset(
            root=os.path.join(args.data_path, cat),
            transform=data_transform, gt_transform=gt_transform, phase="test"
        )
        loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        model = load_model(args, device)
        scores, labels = get_scores(model, loader, device)
        normal_s, defect_s = plot_distribution(scores, labels, cat, out_dir)

        mode, gap, overlap_pct = classify_failure_mode(normal_s, defect_s)

        print(f'  Normal : mean={normal_s.mean():.4f}  std={normal_s.std():.4f}')
        print(f'  Defect : mean={defect_s.mean():.4f}  std={defect_s.std():.4f}')
        print(f'  Gap    : {gap:.4f}')
        print(f'  Overlap: {overlap_pct:.1f}%')
        print(f'  Mode   : {mode}')
        summary.append((cat, normal_s.mean(), normal_s.std(),
                        defect_s.mean(), defect_s.std(), gap, overlap_pct, mode))

    print('\n' + '=' * 90)
    print(f'{"Cat":<15} {"N_mean":>8} {"N_std":>7} {"D_mean":>8} {"D_std":>7} '
          f'{"Gap":>8} {"Overlap%":>9}  Mode')
    for r in summary:
        print(f'{r[0]:<15} {r[1]:>8.4f} {r[2]:>7.4f} {r[3]:>8.4f} {r[4]:>7.4f} '
              f'{r[5]:>8.4f} {r[6]:>8.1f}%  {r[7]}')
    print(f'\nPlots saved to: {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',    type=str, default='./reproduced_results',
                        help='Folder chứa các subfolder category, mỗi cái có model.pth')
    parser.add_argument('--encoder',     type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size',  type=int, default=448)
    parser.add_argument('--crop_size',   type=int, default=392)
    parser.add_argument('--INP_num',     type=int, default=6)
    parser.add_argument('--batch_size',  type=int, default=16,
                        help='Batch size inference, 16GB VRAM dùng 16 là ổn')
    args = parser.parse_args()
    main(args)