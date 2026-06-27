# diagnosis3_score_distribution.py
import sys
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # tránh hang/crash trên server không có display
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from torch.nn import functional as F

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from utils import get_gaussian_kernel, cal_anomaly_maps

import warnings
warnings.filterwarnings("ignore")


def log(msg, logfile):
    print(msg, flush=True)
    print(msg, file=logfile, flush=True)


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
        for img, gt, label, _ in tqdm(dataloader, ncols=80, file=sys.stdout):
            img = img.to(device)
            en, de, _ = model(img)
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


def percentile_table(arr, name, logfile):
    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(arr, pcts)
    row = "  ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals))
    log(f"    {name:<8}: n={len(arr):4d}  mean={arr.mean():.4f}  std={arr.std():.4f}", logfile)
    log(f"               {row}", logfile)


def classify_failure_mode(normal_s, defect_s):
    if len(defect_s) == 0:
        return 'N/A — no defect samples', 0.0, 0.0
    overlap_pct = (defect_s < normal_s.max()).mean() * 100
    gap = defect_s.mean() - normal_s.mean()
    normal_std = normal_s.std()

    if overlap_pct > 60:
        mode = 'A — Miss (overlap cao, model khong phan biet duoc)'
    elif normal_std > 0.5 * gap:
        mode = 'B — False Positive (normal score phan tan, FP cao)'
    else:
        mode = 'C — Mixed'
    return mode, gap, overlap_pct


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)
    out_dir = os.path.normpath(os.path.join(args.ckpt_dir, '..', 'diagnosis3_score_dist'))
    os.makedirs(out_dir, exist_ok=True)

    logfile_path = os.path.join(out_dir, 'diagnosis3_log.txt')
    logfile = open(logfile_path, 'w')

    VALID_CATEGORIES = {'can', 'fabric', 'fruit_jelly', 'rice',
                        'sheet_metal', 'vial', 'wallplugs', 'walnuts'}

    summary = []
    for cat in sorted(VALID_CATEGORIES):
        ckpt = os.path.join(args.ckpt_dir, cat, 'model.pth')
        if not os.path.exists(ckpt):
            log(f'[SKIP] {cat} — checkpoint not found at {ckpt}', logfile)
            continue

        log(f'\n=== {cat.upper()} ===', logfile)
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

        normal_s = scores[labels == 0]
        defect_s = scores[labels == 1]

        # --- dump per-image CSV ---
        csv_path = os.path.join(out_dir, f'{cat}_scores.csv')
        np.savetxt(csv_path, np.column_stack([labels, scores]),
                   delimiter=',', header='label,score', comments='', fmt='%.6f')

        # --- text stats (luon co, du plot co loi hay khong) ---
        percentile_table(normal_s, "Normal", logfile)
        if len(defect_s) > 0:
            percentile_table(defect_s, "Defect", logfile)
        else:
            log("    Defect  : n=0  (KHONG CO ANH DEFECT trong test set nay!)", logfile)

        mode, gap, overlap_pct = classify_failure_mode(normal_s, defect_s)
        log(f'  Gap     : {gap:.4f}', logfile)
        log(f'  Overlap : {overlap_pct:.1f}%   '
            f'(% defect-image co score < max(normal-score))', logfile)
        log(f'  Mode    : {mode}', logfile)

        # --- plot (best-effort, khong lam crash mat log) ---
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(normal_s, bins=40, alpha=0.6, color='steelblue',
                    label=f'Normal (n={len(normal_s)})')
            if len(defect_s) > 0:
                ax.hist(defect_s, bins=40, alpha=0.6, color='tomato',
                        label=f'Defect (n={len(defect_s)})')
            ax.set_title(f'[{cat}]  gap={gap:.4f}  overlap={overlap_pct:.1f}%')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{cat}_score_dist.png'), dpi=120)
            plt.close(fig)
        except Exception as e:
            log(f'  [WARN] plot failed: {e}', logfile)

        summary.append((cat, normal_s.mean(), normal_s.std(),
                        defect_s.mean() if len(defect_s) else float('nan'),
                        defect_s.std() if len(defect_s) else float('nan'),
                        gap, overlap_pct, mode))

    log('\n' + '=' * 100, logfile)
    log(f'{"Cat":<15} {"N_mean":>8} {"N_std":>7} {"D_mean":>8} {"D_std":>7} '
        f'{"Gap":>8} {"Overlap%":>9}  Mode', logfile)
    for r in summary:
        log(f'{r[0]:<15} {r[1]:>8.4f} {r[2]:>7.4f} {r[3]:>8.4f} {r[4]:>7.4f} '
            f'{r[5]:>8.4f} {r[6]:>8.1f}%  {r[7]}', logfile)
    log(f'\nLog file : {logfile_path}', logfile)
    log(f'Per-image CSVs and plots in: {out_dir}', logfile)
    logfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   type=str, default=r'E:\dataset\mvtecad2')
    parser.add_argument('--ckpt_dir',    type=str, default='./reproduced_results')
    parser.add_argument('--encoder',     type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size',  type=int, default=448)
    parser.add_argument('--crop_size',   type=int, default=392)
    parser.add_argument('--INP_num',     type=int, default=6)
    parser.add_argument('--batch_size',  type=int, default=16)
    args = parser.parse_args()
    main(args)