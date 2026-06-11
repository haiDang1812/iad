# ablation_dcpl.py
# Ablation study: LocalContrast only vs OrthoLoss only vs Both vs Baseline
# Chạy trên 1 category để debug nhanh

import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import (evaluation_batch, WarmCosineScheduler,
                   global_cosine_hm_adaptive, setup_seed, get_logger)
from dataset import MVTecAD2Dataset, get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

warnings.filterwarnings("ignore")

SETTINGS = [
    {'name': 'baseline',    'use_local_contrast': False, 'use_ortho_loss': False},
    {'name': 'ortho_only',  'use_local_contrast': False, 'use_ortho_loss': True},
    {'name': 'lc_only',     'use_local_contrast': True,  'use_ortho_loss': False},
    {'name': 'both',        'use_local_contrast': True,  'use_ortho_loss': True},
]


def build_model(args, device, embed_dim, num_heads,
                target_layers, fuse_layer_encoder, fuse_layer_decoder,
                use_local_contrast, use_ortho_loss):
    Bottleneck = nn.ModuleList([
        Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)
    ])
    INP = nn.ParameterList([
        nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)
    ])
    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])
    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])
    encoder = vit_encoder.load(args.encoder)
    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder, target_layers=target_layers,
        remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP,
        use_local_contrast=use_local_contrast,
        use_ortho_loss=use_ortho_loss,
        neighborhood_sizes=args.neighborhood_sizes,
    ).to(device)
    return model, Bottleneck, INP, INP_Extractor, INP_Guided_Decoder


def train_and_eval(args, device, use_local_contrast, use_ortho_loss,
                   setting_name, print_fn):
    setup_seed(1)

    data_transform, gt_transform = get_data_transforms(
        args.input_size, args.crop_size)
    train_data = ImageFolder(
        root=os.path.join(args.data_path, args.item, 'train'),
        transform=data_transform)
    test_data = MVTecAD2Dataset(
        root=os.path.join(args.data_path, args.item),
        transform=data_transform, gt_transform=gt_transform, phase='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4)

    target_layers      = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError()

    model, Bottleneck, INP, INP_Extractor, INP_Guided_Decoder = build_model(
        args, device, embed_dim, num_heads,
        target_layers, fuse_layer_encoder, fuse_layer_decoder,
        use_local_contrast, use_ortho_loss)

    trainable_modules = [Bottleneck, INP_Guided_Decoder, INP_Extractor, INP]
    if use_local_contrast:
        trainable_modules.append(model.local_contrast)
    trainable = nn.ModuleList(trainable_modules)

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW(
        [{'params': trainable.parameters()}],
        lr=1e-3, betas=(0.9, 0.999),
        weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=1e-3, final_value=1e-4,
        total_iters=args.total_epochs * len(train_loader),
        warmup_iters=100)

    print_fn(f'\n--- Setting: {setting_name} | '
             f'LC={use_local_contrast} | Ortho={use_ortho_loss} ---')

    for epoch in range(args.total_epochs):
        model.train()
        loss_list = loss_main_list = loss_g_list = loss_ortho_list = []
        loss_list = []; loss_main_list = []; loss_g_list = []; loss_ortho_list = []

        for img, _ in tqdm(train_loader, ncols=80,
                           desc=f'[{setting_name}] epoch {epoch+1}'):
            img = img.to(device)
            en, de, g_loss, ortho_loss = model(img)

            loss_main = global_cosine_hm_adaptive(en, de, y=3)
            loss = (loss_main
                    + 0.2 * g_loss
                    + args.ortho_loss_weight * ortho_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            lr_scheduler.step()

            loss_list.append(loss.item())
            loss_main_list.append(loss_main.item())
            loss_g_list.append(g_loss.item())
            loss_ortho_list.append(ortho_loss.item())

        if (epoch + 1) % 50 == 0 or epoch == args.total_epochs - 1:
            print_fn(
                'epoch [{}/{}] loss:{:.4f} main:{:.4f} '
                'g:{:.4f} ortho:{:.4f}'.format(
                    epoch+1, args.total_epochs,
                    np.mean(loss_list), np.mean(loss_main_list),
                    np.mean(loss_g_list), np.mean(loss_ortho_list)))

    results = evaluation_batch(model, test_loader, device,
                               max_ratio=0.01, resize_mask=256)
    auroc_sp, ap_sp, f1_sp, \
        auroc_px, ap_px, f1_px, \
        aupro_px, aupro_005, aupro_030 = results

    print_fn(
        '[{}] {}: I-Auroc:{:.4f} I-AP:{:.4f} I-F1:{:.4f} '
        'P-AUROC:{:.4f} P-AP:{:.4f} P-F1:{:.4f} '
        'P-AUPRO:{:.4f} AUPRO0.05:{:.4f} AUPRO0.30:{:.4f}'.format(
            setting_name, args.item,
            auroc_sp, ap_sp, f1_sp,
            auroc_px, ap_px, f1_px,
            aupro_px, aupro_005, aupro_030))
    return results


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--data_path',  type=str, default=r'E:\dataset\mvtecad2')

    # Save
    parser.add_argument('--save_dir',   type=str, default='./saved_results')

    # Model
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)

    # Method
    parser.add_argument('--ortho_loss_weight',  type=float, default=0.1)
    parser.add_argument('--neighborhood_sizes', type=int, nargs='+', default=[3, 5])

    # Training
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size',   type=int, default=16)

    # Ablation target — chạy 1 hoặc nhiều category
    parser.add_argument('--item_list', type=str, nargs='+',
                        default=['vial', 'fruit_jelly'],
                        help='Categories to ablate on')

    args = parser.parse_args()

    save_name = 'DCPL_Ablation'
    logger    = get_logger(save_name, os.path.join(args.save_dir, save_name))
    print_fn  = logger.info
    device    = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Summary table
    all_results = {}  # {category: {setting: results}}

    for item in args.item_list:
        args.item = item
        all_results[item] = {}
        print_fn(f'\n{"="*60}')
        print_fn(f'Category: {item.upper()}')
        print_fn(f'{"="*60}')

        for setting in SETTINGS:
            results = train_and_eval(
                args, device,
                use_local_contrast=setting['use_local_contrast'],
                use_ortho_loss=setting['use_ortho_loss'],
                setting_name=setting['name'],
                print_fn=print_fn)
            all_results[item][setting['name']] = results

    # Print summary table
    print_fn(f'\n{"="*60}')
    print_fn('ABLATION SUMMARY — AUPRO | AUPRO0.05 | AUPRO0.30')
    print_fn(f'{"="*60}')
    metrics_idx = {'AUPRO': 6, 'AUPRO0.05': 7, 'AUPRO0.30': 8}

    for item in args.item_list:
        print_fn(f'\n[{item.upper()}]')
        header = f'  {"Setting":<15} {"AUPRO":>8} {"AUPRO0.05":>10} {"AUPRO0.30":>10}'
        print_fn(header)
        for setting in SETTINGS:
            r = all_results[item][setting['name']]
            print_fn(f'  {setting["name"]:<15} '
                     f'{r[6]:>8.4f} {r[7]:>10.4f} {r[8]:>10.4f}')