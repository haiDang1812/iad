# train_mvtecad2_rcpl.py
# Residual-Calibrated Prototype Learning (RCPL) on MVTec AD 2
# Build on top of INP-Former (CVPR 2025)

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
from utils import evaluation_batch, WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

from dataset import MVTecAD2Dataset, get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

warnings.filterwarnings("ignore")


def build_model(args, device, embed_dim, num_heads, target_layers,
                fuse_layer_encoder, fuse_layer_decoder):
    Bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)])

    INP = nn.ParameterList([
        nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)
    ])

    INP_Extractor = nn.ModuleList([
        Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
    ])

    INP_Guided_Decoder = nn.ModuleList([
        Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        for _ in range(8)
    ])

    encoder = vit_encoder.load(args.encoder)

    model = INP_Former(
        encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
        decoder=INP_Guided_Decoder, target_layers=target_layers,
        remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP,
        use_calibration=args.use_calibration,
    ).to(device)

    return model, Bottleneck, INP, INP_Extractor, INP_Guided_Decoder


def main(args):
    setup_seed(1)

    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    train_path = os.path.join(args.data_path, args.item, 'train')
    test_path  = os.path.join(args.data_path, args.item)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data  = MVTecAD2Dataset(root=test_path, transform=data_transform,
                                  gt_transform=gt_transform, phase="test")

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, drop_last=True)
    test_dataloader  = DataLoader(test_data,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=4)

    # Target layers
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
        raise ValueError("Architecture not in small, base, large.")

    model, Bottleneck, INP, INP_Extractor, INP_Guided_Decoder = build_model(
        args, device, embed_dim, num_heads,
        target_layers, fuse_layer_encoder, fuse_layer_decoder
    )

    if args.phase == 'train':
        # Trainable modules
        trainable_modules = [Bottleneck, INP_Guided_Decoder, INP_Extractor, INP]
        if args.use_calibration:
            trainable_modules.append(model.residual_calibration)

        trainable = nn.ModuleList(trainable_modules)

        # Init weights
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
            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10
        )
        lr_scheduler = WarmCosineScheduler(
            optimizer, base_value=1e-3, final_value=1e-4,
            total_iters=args.total_epochs * len(train_dataloader),
            warmup_iters=100
        )

        print_fn(f'train image number: {len(train_data)}')
        print_fn(f'use_calibration: {args.use_calibration}')
        print_fn(f'cal_loss_weight: {args.cal_loss_weight}')

        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            loss_main_list = []
            loss_g_list    = []
            loss_cal_list  = []

            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)

                if args.use_calibration:
                    en, de, g_loss, cal_loss = model(img)
                else:
                    en, de, g_loss = model(img)
                    cal_loss = torch.tensor(0., device=device)

                loss_main = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss_main + 0.2 * g_loss + args.cal_loss_weight * cal_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()

                loss_list.append(loss.item())
                loss_main_list.append(loss_main.item())
                loss_g_list.append(g_loss.item())
                loss_cal_list.append(cal_loss.item())

            print_fn(
                'epoch [{}/{}]  loss:{:.4f}  main:{:.4f}  g:{:.4f}  cal:{:.4f}'.format(
                    epoch + 1, args.total_epochs,
                    np.mean(loss_list), np.mean(loss_main_list),
                    np.mean(loss_g_list), np.mean(loss_cal_list)
                )
            )

        # Evaluation after training
        results = evaluation_batch(model, test_dataloader, device,
                                   max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030 = results
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, '
            'P-AUPRO:{:.4f}, AUPRO0.05:{:.4f}, AUPRO0.30:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp,
                auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030
            )
        )

        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, args.save_name, args.item, 'model.pth'))
        return results

    elif args.phase == 'test':
        ckpt = os.path.join(args.save_dir, args.save_name, args.item, 'model.pth')
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        model.eval()
        results = evaluation_batch(model, test_dataloader, device,
                                   max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030 = results
        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, '
            'P-AUPRO:{:.4f}, AUPRO0.05:{:.4f}, AUPRO0.30:{:.4f}'.format(
                args.item, auroc_sp, ap_sp, f1_sp,
                auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030
            )
        )
        return results


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    parser = argparse.ArgumentParser(description='RCPL on MVTecAD-2')

    # Dataset
    parser.add_argument('--dataset',   type=str, default='MVTecAD-2')
    parser.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')

    # Save
    parser.add_argument('--save_dir',  type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='RCPL-MVTecAD2')

    # Model
    parser.add_argument('--encoder',    type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size',  type=int, default=392)
    parser.add_argument('--INP_num',    type=int, default=6)

    # Method
    parser.add_argument('--use_calibration',  action='store_true', default=True,
                        help='Enable Residual Calibration Module')
    parser.add_argument('--cal_loss_weight',  type=float, default=0.1,
                        help='Weight for normality contrastive loss')

    # Training
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--phase',        type=str, default='train',
                        choices=['train', 'test'])

    args = parser.parse_args()
    args.save_name = (args.save_name +
                      f'_Encoder={args.encoder}'
                      f'_Crop={args.crop_size}'
                      f'_INP={args.INP_num}'
                      f'_Cal={args.use_calibration}'
                      f'_CalW={args.cal_loss_weight}')

    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device   = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.item_list = ['fabric', 'fruit_jelly', 'rice',
                      'sheet_metal', 'vial', 'wallplugs', 'walnuts']

    result_list = []
    for item in args.item_list:
        args.item = item
        print_fn('=' * 50)
        print_fn(f'Processing category: {item}')
        print_fn('=' * 50)
        results = main(args)
        result_list.append(results)

    result_list = np.array(result_list)
    print_fn('=' * 50)
    print_fn('FINAL RESULTS (Mean across all categories):')
    print_fn('I-AUROC:   {:.4f}'.format(result_list[:, 0].mean()))
    print_fn('I-AP:      {:.4f}'.format(result_list[:, 1].mean()))
    print_fn('I-F1:      {:.4f}'.format(result_list[:, 2].mean()))
    print_fn('P-AUROC:   {:.4f}'.format(result_list[:, 3].mean()))
    print_fn('P-AP:      {:.4f}'.format(result_list[:, 4].mean()))
    print_fn('P-F1:      {:.4f}'.format(result_list[:, 5].mean()))
    print_fn('P-AUPRO:   {:.4f}'.format(result_list[:, 6].mean()))
    print_fn('AUPRO0.05: {:.4f}'.format(result_list[:, 7].mean()))
    print_fn('AUPRO0.30: {:.4f}'.format(result_list[:, 8].mean()))