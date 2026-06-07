# diagnosis7_perlayer_residual.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import math
from torch.nn import functional as F

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import get_gaussian_kernel

import warnings
warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']


class Prototype_Block_SaveOutput(Prototype_Block):
    def forward(self, x, prototype, return_attention=False):
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        out = x + self.drop_path(y)
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        self.last_output = out.detach()
        if return_attention:
            return out, attn
        return out


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
    # patched decoder
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block_SaveOutput(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
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


def compute_per_layer_residual(model, en_fused, side):
    """
    Compute residual between encoder fused features and each decoder layer output.
    en_fused: [B, C, H, W]
    Returns per-layer mean residual [8]
    """
    layer_residuals = []
    for blk in model.decoder:
        de_out = blk.last_output  # [B, N, C]
        de_sp  = de_out.permute(0, 2, 1).reshape(
            de_out.shape[0], -1, side, side).contiguous()
        res = 1.0 - F.cosine_similarity(en_fused, de_sp, dim=1)  # [B, H, W]
        layer_residuals.append(res.mean().item())
    return layer_residuals


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

    normal_layer_res = [[] for _ in range(8)]
    defect_layer_res = [[] for _ in range(8)]

    with torch.no_grad():
        for img, gt, label, _ in tqdm(loader, ncols=80):
            img = img.to(device)
            en, de, _ = model(img)

            # Use first fused encoder feature
            en_fused = en[0]  # [B, C, H, W]
            side = en_fused.shape[-1]

            layer_res = compute_per_layer_residual(model, en_fused, side)

            for i, r in enumerate(layer_res):
                for b in range(img.shape[0]):
                    if label[b].item() == 0:
                        normal_layer_res[i].append(r)
                    else:
                        defect_layer_res[i].append(r)

    # Summary
    header = f'  {"Layer":>6} {"Normal":>10} {"Defect":>10} {"Gap":>8}  Verdict'
    print(header); log_lines.append(header)

    normal_means = [np.mean(v) for v in normal_layer_res]
    defect_means = [np.mean(v) for v in defect_layer_res]

    for i in range(8):
        gap = defect_means[i] - normal_means[i]
        verdict = '⚠ LOW GAP' if gap < 0.01 else ('✓ GOOD' if gap > 0.05 else 'OK')
        row = f'  {i:>6} {normal_means[i]:>10.4f} {defect_means[i]:>10.4f} {gap:>8.4f}  {verdict}'
        print(row); log_lines.append(row)

    # Plot
    out_cat = os.path.join(args.out_dir, cat)
    os.makedirs(out_cat, exist_ok=True)

    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, normal_means, 'o-', color='steelblue', label='Normal')
    ax.plot(x, defect_means, 's-', color='tomato',    label='Defect')
    ax.fill_between(x, normal_means, defect_means, alpha=0.15, color='gray')
    ax.set_xticks(x); ax.set_xticklabels([f'Layer {i}' for i in x], rotation=30)
    ax.set_ylabel('Mean residual (1 - cosine_sim)')
    ax.set_title(f'[{cat}] Per-layer residual: normal vs defect')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_cat, f'{cat}_perlayer_residual.png'), dpi=120)
    plt.close()


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = ['Diagnosis 7 — Per-layer Residual Contribution']

    for cat in VALID_CATEGORIES:
        run_category(args, cat, device, log_lines)

    log_path = os.path.join(args.out_dir, 'diagnosis7_log.txt')
    with open(log_path, 'w') as f: f.write('\n'.join(log_lines))
    print(f'\nLog saved to: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',   type=str, default='./reproduced_results')
    parser.add_argument('--out_dir',    type=str, default='./diagnosis/diagnosis7_perlayer_residual')
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)
    args = parser.parse_args()
    main(args)