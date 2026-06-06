# diagnosis4_inp_contamination.py
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']


# ── Patch Aggregation_Block để expose attention ──────────────────────────────
class Aggregation_Block_WithAttn(Aggregation_Block):
    def forward(self, x, y):
        normed_x = self.norm1(x)
        normed_y = self.norm1(y)
        attn_out, attn_map = self.attn(normed_x, normed_y, return_attn=True)
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        self.last_attn = attn_map  # [B, heads, INP_num, N_tokens]
        return x


from models.vision_transformer import Aggregation_Attention

class Aggregation_Attention_WithReturn(Aggregation_Attention):
    def forward(self, x, y, return_attn=False):
        B, T, C = x.shape
        _, N, _ = y.shape
        q = self.q(x).reshape(B, T, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attnmap = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attnmap)
        out = (attn_drop @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, attnmap  # attnmap: [B, heads, T_inp, N_img]
        return out
# ─────────────────────────────────────────────────────────────────────────────


def build_aggregation_block_with_attn(embed_dim, num_heads):
    blk = Aggregation_Block_WithAttn(
        dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8)
    )
    # swap attention module
    blk.attn = Aggregation_Attention_WithReturn(
        dim=embed_dim, num_heads=num_heads, qkv_bias=True
    )
    return blk


def load_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim, num_heads = 768, 12

    encoder = vit_encoder.load(args.encoder)
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))])

    # patched INP Extractor
    INP_Extractor = nn.ModuleList([
        build_aggregation_block_with_attn(embed_dim, num_heads)
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


def get_inp_attention(model):
    """
    Aggregation_Block_WithAttn lưu last_attn sau mỗi forward.
    Shape: [B, num_heads, INP_num, N_tokens]  (INP attend vào image tokens)
    """
    return model.aggregation[0].last_attn  # [B, H, INP, N]


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def visualize_sample(img_tensor, gt_tensor, attn, anomaly_map,
                     label, img_path, out_path, side=28):
    """
    4 panel: original | GT mask | INP attention heatmap | anomaly map
    attn: [heads, INP_num, N_tokens] → mean over heads & INP → [N_tokens] → reshape [side, side]
    """
    attn_map = attn.mean(dim=0).mean(dim=0)          # [N_tokens]
    attn_map = attn_map.reshape(side, side).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    ano_map = anomaly_map.squeeze().cpu().numpy()
    ano_map = (ano_map - ano_map.min()) / (ano_map.max() - ano_map.min() + 1e-8)

    img_np = denormalize(img_tensor).permute(1, 2, 0).numpy()
    gt_np  = gt_tensor.squeeze().cpu().numpy()

    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    ax0 = fig.add_subplot(gs[0]); ax0.imshow(img_np);              ax0.set_title('Input')
    ax1 = fig.add_subplot(gs[1]); ax1.imshow(gt_np, cmap='gray');  ax1.set_title('GT Mask')
    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(img_np)
    ax2.imshow(attn_map, cmap='jet', alpha=0.55, interpolation='bilinear',
               extent=[0, img_np.shape[1], img_np.shape[0], 0])
    ax2.set_title('INP Attention')
    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(img_np)
    ax3.imshow(ano_map, cmap='jet', alpha=0.55, interpolation='bilinear',
               extent=[0, img_np.shape[1], img_np.shape[0], 0])
    ax3.set_title('Anomaly Map')

    for ax in [ax0, ax1, ax2, ax3]:
        ax.axis('off')

    status = 'DEFECT' if label == 1 else 'NORMAL'
    fig.suptitle(f'{status} | {os.path.basename(img_path)}', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def run_category(args, cat, device, log_lines):
    args.item = cat
    ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
    if not os.path.exists(ckpt):
        msg = f'[SKIP] {cat} — checkpoint not found'
        print(msg); log_lines.append(msg)
        return

    print(f'\n=== {cat.upper()} ===')
    log_lines.append(f'\n=== {cat.upper()} ===')

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    test_data = MVTecAD2Dataset(
        root=os.path.join(args.data_path, cat),
        transform=data_transform, gt_transform=gt_transform, phase='test'
    )
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=2
    )

    model = load_model(args, device)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    out_cat = os.path.join(args.out_dir, cat)
    os.makedirs(os.path.join(out_cat, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(out_cat, 'defect'), exist_ok=True)

    # stats: INP attention overlap với GT mask
    overlap_scores = []   # % attention mass ở vùng defect
    saved_normal = saved_defect = 0

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(loader, ncols=80):
            img = img.to(device)
            en, de, _ = model(img)

            # anomaly map
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(anomaly_map, size=256,
                                        mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

            # INP attention
            attn = get_inp_attention(model)  # [1, heads, INP_num, N]

            # overlap metric (defect only)
            if label.item() == 1:
                side = int(math.sqrt(attn.shape[-1]))
                attn_map_2d = attn[0].mean(0).mean(0).reshape(side, side)  # [28,28]
                attn_map_2d = (attn_map_2d - attn_map_2d.min()) / \
                              (attn_map_2d.max() - attn_map_2d.min() + 1e-8)

                gt_resized = F.interpolate(gt.to(device), size=(side, side),
                                           mode='nearest').squeeze()
                gt_bin = (gt_resized > 0.5).float()

                if gt_bin.sum() > 0:
                    defect_attn  = (attn_map_2d * gt_bin).sum() / (gt_bin.sum() + 1e-8)
                    normal_attn  = (attn_map_2d * (1 - gt_bin)).sum() / ((1 - gt_bin).sum() + 1e-8)
                    overlap_scores.append((defect_attn.item(), normal_attn.item()))

            # save visualizations (top N per class)
            label_val = label.item()
            if label_val == 0 and saved_normal < args.vis_per_class:
                out_path = os.path.join(out_cat, 'normal',
                                        f'{saved_normal:03d}.png')
                visualize_sample(img[0], gt[0], attn[0], anomaly_map[0],
                                 label_val, img_path[0], out_path)
                saved_normal += 1
            elif label_val == 1 and saved_defect < args.vis_per_class:
                out_path = os.path.join(out_cat, 'defect',
                                        f'{saved_defect:03d}.png')
                visualize_sample(img[0], gt[0], attn[0], anomaly_map[0],
                                 label_val, img_path[0], out_path)
                saved_defect += 1

    # summary stats
    if overlap_scores:
        def_attn_vals  = [x[0] for x in overlap_scores]
        norm_attn_vals = [x[1] for x in overlap_scores]
        contamination_ratio = np.mean(def_attn_vals) / (np.mean(norm_attn_vals) + 1e-8)
        msg = (f'  INP attn on defect region  : mean={np.mean(def_attn_vals):.4f}  '
               f'std={np.std(def_attn_vals):.4f}\n'
               f'  INP attn on normal region  : mean={np.mean(norm_attn_vals):.4f}  '
               f'std={np.std(norm_attn_vals):.4f}\n'
               f'  Contamination ratio (def/norm attn): {contamination_ratio:.4f}  '
               f'{"⚠ HIGH" if contamination_ratio > 0.8 else "OK"}')
    else:
        msg = '  No defect images found for overlap analysis'

    print(msg); log_lines.append(msg)
    return overlap_scores


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = ['Diagnosis 4 — INP Contamination Analysis']

    summary = []
    for cat in VALID_CATEGORIES:
        result = run_category(args, cat, device, log_lines)
        if result:
            def_vals  = [x[0] for x in result]
            norm_vals = [x[1] for x in result]
            ratio = np.mean(def_vals) / (np.mean(norm_vals) + 1e-8)
            summary.append((cat, np.mean(def_vals), np.mean(norm_vals), ratio))

    # print & save summary
    header = (f'\n{"="*70}\n'
              f'{"Category":<15} {"Attn@defect":>12} {"Attn@normal":>12} '
              f'{"Contam.ratio":>14}  Verdict\n{"="*70}')
    print(header); log_lines.append(header)

    for cat, d, n, r in summary:
        verdict = '⚠ CONTAMINATED' if r > 0.8 else 'clean'
        row = f'{cat:<15} {d:>12.4f} {n:>12.4f} {r:>14.4f}  {verdict}'
        print(row); log_lines.append(row)

    footer = f'\nVisualizations saved to: {args.out_dir}'
    print(footer); log_lines.append(footer)

    # save log
    log_path = os.path.join(args.out_dir, 'diagnosis4_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',       type=str, default='./reproduced_results')
    parser.add_argument('--out_dir',        type=str, default='./diagnosis/diagnosis4_inp_contamination')
    parser.add_argument('--encoder',        type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size',     type=int, default=448)
    parser.add_argument('--crop_size',      type=int, default=392)
    parser.add_argument('--INP_num',        type=int, default=6)
    parser.add_argument('--vis_per_class',  type=int, default=10,
                        help='Số ảnh visualize per normal/defect per category')
    args = parser.parse_args()
    main(args)