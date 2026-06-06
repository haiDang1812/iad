# diagnosis5_decoder_attention.py
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


# ── Patch Prototype_Block để luôn return attention ───────────────────────────
class Prototype_Block_WithAttn(Prototype_Block):
    def forward(self, x, prototype, return_attention=False):
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        out = x + self.drop_path(y)
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        self.last_attn = attn  # [B, heads, N_tokens, INP_num]
        return out


# ── Patch INP_Former để dùng Prototype_Block_WithAttn ────────────────────────
class INP_Former_D5(INP_Former):
    """Chỉ override forward để lưu decoder attn của block đầu tiên."""
    def forward(self, x):
        en, de, g_loss = super().forward(x)
        return en, de, g_loss
# ─────────────────────────────────────────────────────────────────────────────


def load_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    embed_dim, num_heads = 768, 12

    encoder = vit_encoder.load(args.encoder)
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim))])

    from models.vision_transformer import Aggregation_Block
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])

    # patched decoder — tất cả 8 block đều lưu attn
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block_WithAttn(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
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


def get_decoder_attn(model, block_idx=0):
    """
    Lấy attention từ decoder block chỉ định.
    Shape: [B, heads, N_tokens, INP_num]
    → image token attend vào INP prototype nào
    """
    return model.decoder[block_idx].last_attn


def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def visualize_sample(img_tensor, gt_tensor, decoder_attn, anomaly_map,
                     label, img_path, out_path, side=28, inp_num=6):
    """
    Panel layout:
    Row 1: input | GT mask | anomaly map | decoder attn entropy map
    Row 2: attn map per INP token (inp_num panels)

    decoder_attn: [heads, N_tokens, INP_num]
    """
    # mean over heads → [N_tokens, INP_num]
    attn = decoder_attn.mean(dim=0).cpu()  # [N, INP]

    # entropy map — token nào attend phân tán nhiều INP → uncertain
    attn_prob = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(attn_prob * (attn_prob + 1e-8).log()).sum(dim=-1)  # [N]
    entropy_map = entropy.reshape(side, side).numpy()
    entropy_map = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)

    ano_map = anomaly_map.squeeze().cpu().numpy()
    ano_map = (ano_map - ano_map.min()) / (ano_map.max() - ano_map.min() + 1e-8)

    img_np = denormalize(img_tensor).permute(1, 2, 0).numpy()
    gt_np  = gt_tensor.squeeze().cpu().numpy()

    n_cols = max(4, inp_num)
    fig = plt.figure(figsize=(4 * n_cols, 8))
    gs  = gridspec.GridSpec(2, n_cols, figure=fig)

    # Row 1
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img_np);             ax.set_title('Input');        ax.axis('off')
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(gt_np, cmap='gray'); ax.set_title('GT Mask');      ax.axis('off')
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img_np)
    ax.imshow(ano_map, cmap='jet', alpha=0.55, interpolation='bilinear',
              extent=[0, img_np.shape[1], img_np.shape[0], 0])
    ax.set_title('Anomaly Map'); ax.axis('off')
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(img_np)
    ax.imshow(entropy_map, cmap='hot', alpha=0.55, interpolation='bilinear',
              extent=[0, img_np.shape[1], img_np.shape[0], 0])
    ax.set_title('Decoder Attn Entropy'); ax.axis('off')

    # Row 2 — per INP token attention
    for i in range(inp_num):
        inp_attn = attn[:, i].reshape(side, side).numpy()
        inp_attn = (inp_attn - inp_attn.min()) / (inp_attn.max() - inp_attn.min() + 1e-8)
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img_np)
        ax.imshow(inp_attn, cmap='jet', alpha=0.55, interpolation='bilinear',
                  extent=[0, img_np.shape[1], img_np.shape[0], 0])
        ax.set_title(f'INP token {i}'); ax.axis('off')

    status = 'DEFECT' if label == 1 else 'NORMAL'
    fig.suptitle(f'{status} | {os.path.basename(img_path)}', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def compute_defect_attn_stats(decoder_attn, gt_patch, inp_num=6):
    """
    decoder_attn: [heads, N_tokens, INP_num]
    gt_patch: [N_tokens] binary (defect=1)

    Returns per-INP-token: mean attention weight từ defect tokens vs normal tokens
    """
    attn = decoder_attn.mean(dim=0)  # [N, INP]
    defect_mask  = gt_patch > 0.5
    normal_mask  = ~defect_mask

    results = []
    for i in range(inp_num):
        inp_col = attn[:, i]  # [N]
        d_attn = inp_col[defect_mask].mean().item() if defect_mask.sum() > 0 else 0.
        n_attn = inp_col[normal_mask].mean().item() if normal_mask.sum() > 0 else 0.
        results.append((d_attn, n_attn))
    return results  # list of (defect_attn, normal_attn) per INP token


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

    # accumulate per-INP stats
    inp_stats = [[] for _ in range(args.INP_num)]  # list of (d,n) per INP token
    saved_normal = saved_defect = 0

    with torch.no_grad():
        for img, gt, label, img_path in tqdm(loader, ncols=80):
            img = img.to(device)
            en, de, _ = model(img)

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = F.interpolate(anomaly_map, size=256,
                                        mode='bilinear', align_corners=False)
            anomaly_map = gaussian_kernel(anomaly_map)

            # decoder attn từ block 0 (first decoder layer)
            dec_attn = get_decoder_attn(model, block_idx=0)  # [1, H, N, INP]
            dec_attn_single = dec_attn[0]  # [H, N, INP]

            side = int(math.sqrt(dec_attn_single.shape[1]))

            # stats for defect images
            if label.item() == 1:
                gt_patch = F.interpolate(gt.to(device), size=(side, side),
                                         mode='nearest').squeeze().reshape(-1)
                per_inp = compute_defect_attn_stats(dec_attn_single, gt_patch, args.INP_num)
                for i, (d, n) in enumerate(per_inp):
                    inp_stats[i].append((d, n))

            # visualize
            label_val = label.item()
            if label_val == 0 and saved_normal < args.vis_per_class:
                out_path = os.path.join(out_cat, 'normal', f'{saved_normal:03d}.png')
                visualize_sample(img[0], gt[0], dec_attn_single, anomaly_map[0],
                                 label_val, img_path[0], out_path,
                                 side=side, inp_num=args.INP_num)
                saved_normal += 1
            elif label_val == 1 and saved_defect < args.vis_per_class:
                out_path = os.path.join(out_cat, 'defect', f'{saved_defect:03d}.png')
                visualize_sample(img[0], gt[0], dec_attn_single, anomaly_map[0],
                                 label_val, img_path[0], out_path,
                                 side=side, inp_num=args.INP_num)
                saved_defect += 1

    # summary per INP token
    header = f'  {"INP":>5} {"Attn@defect":>12} {"Attn@normal":>12} {"Ratio":>8}  Verdict'
    print(header); log_lines.append(header)

    ratios = []
    for i, stats in enumerate(inp_stats):
        if not stats:
            continue
        d_vals = [x[0] for x in stats]
        n_vals = [x[1] for x in stats]
        d_mean, n_mean = np.mean(d_vals), np.mean(n_vals)
        ratio = d_mean / (n_mean + 1e-8)
        ratios.append(ratio)
        verdict = '⚠ CONTAMINATED' if ratio > 0.8 else 'clean'
        row = f'  {i:>5} {d_mean:>12.4f} {n_mean:>12.4f} {ratio:>8.4f}  {verdict}'
        print(row); log_lines.append(row)

    if ratios:
        n_contaminated = sum(1 for r in ratios if r > 0.8)
        summary_line = (f'  → {n_contaminated}/{len(ratios)} INP tokens contaminated  '
                        f'| mean ratio={np.mean(ratios):.4f}')
        print(summary_line); log_lines.append(summary_line)

    return ratios


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = ['Diagnosis 5 — Decoder Attention Analysis (which INP token does defect region use?)']

    summary_all = []
    for cat in VALID_CATEGORIES:
        ratios = run_category(args, cat, device, log_lines)
        if ratios:
            n_cont = sum(1 for r in ratios if r > 0.8)
            summary_all.append((cat, np.mean(ratios), n_cont, len(ratios)))

    header = (f'\n{"="*65}\n'
              f'{"Category":<15} {"Mean ratio":>12} {"Contaminated INPs":>18}  Verdict\n'
              f'{"="*65}')
    print(header); log_lines.append(header)

    for cat, mean_r, n_cont, total in summary_all:
        verdict = '⚠ CONTAMINATED' if mean_r > 0.8 else 'clean'
        row = f'{cat:<15} {mean_r:>12.4f} {n_cont:>12}/{total:<5}  {verdict}'
        print(row); log_lines.append(row)

    footer = f'\nVisualizations saved to: {args.out_dir}'
    print(footer); log_lines.append(footer)

    log_path = os.path.join(args.out_dir, 'diagnosis5_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f'Log saved to: {log_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',       type=str, default='./reproduced_results')
    parser.add_argument('--out_dir',        type=str, default='./diagnosis/diagnosis5_decoder_attention')
    parser.add_argument('--encoder',        type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size',     type=int, default=448)
    parser.add_argument('--crop_size',      type=int, default=392)
    parser.add_argument('--INP_num',        type=int, default=6)
    parser.add_argument('--vis_per_class',  type=int, default=10)
    args = parser.parse_args()
    main(args)