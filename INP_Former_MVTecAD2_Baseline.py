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

def main(args):
    setup_seed(1)

    # Data transforms - 392x392 for MVTecAD-2
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    # MVTecAD-2 dataset
    train_path = os.path.join(args.data_path, args.item, 'train')
    test_path = os.path.join(args.data_path, args.item)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecAD2Dataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=4)

    # Target layers
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder setup
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    # Model components
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP tokens - M=6 (original INP-Former)
    INP = nn.ParameterList([nn.Parameter(torch.randn(args.INP_num, embed_dim)) for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP-Guided Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    # Build model
    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor,
                      decoder=INP_Guided_Decoder, target_layers=target_layers,
                      remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                      fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    if args.phase == 'train':
        # Model initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4,
                                          total_iters=args.total_epochs*len(train_dataloader),
                                          warmup_iters=100)

        print_fn('train image number:{}'.format(len(train_data)))

        # Training loop
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, _ in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))

        # Evaluation
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030 = results
        print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, AUPRO0.05:{:.4f}, AUPRO0.30:{:.4f}'.format(
            args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030))

        os.makedirs(os.path.join(args.save_dir, args.save_name, args.item), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, args.item, 'model.pth'))
        return results

    elif args.phase == 'test':
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, args.item, 'model.pth')),
                            strict=True)
        model.eval()
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030 = results
        print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, AUPRO0.05:{:.4f}, AUPRO0.30:{:.4f}'.format(
            args.item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, aupro_005, aupro_030))
        return results


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='INP-Former Baseline on MVTecAD-2')

    # Dataset
    parser.add_argument('--dataset', type=str, default='MVTecAD-2')
    parser.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')

    # Save
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='INP-Former-Baseline-MVTecAD2')

    # Model - use M=6 (original) and 392x392
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # Training
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_Encoder={args.encoder}_Crop={args.crop_size}_INP={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # MVTecAD-2 categories
    args.item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wall_plugs', 'walnuts']

    result_list = []
    for item in args.item_list:
        args.item = item
        print_fn('='*50)
        print_fn(f'Processing category: {item}')
        print_fn('='*50)
        results = main(args)
        result_list.append(results)

    # Summary
    result_list = np.array(result_list)
    print_fn('='*50)
    print_fn('FINAL RESULTS (Mean across all categories):')
    print_fn('I-AUROC: {:.4f}'.format(result_list[:, 0].mean()))
    print_fn('I-AP: {:.4f}'.format(result_list[:, 1].mean()))
    print_fn('I-F1: {:.4f}'.format(result_list[:, 2].mean()))
    print_fn('P-AUROC: {:.4f}'.format(result_list[:, 3].mean()))
    print_fn('P-AP: {:.4f}'.format(result_list[:, 4].mean()))
    print_fn('P-F1: {:.4f}'.format(result_list[:, 5].mean()))
    print_fn('P-AUPRO: {:.4f}'.format(result_list[:, 6].mean()))
    print_fn('AUPRO0.05: {:.4f}'.format(result_list[:, 7].mean()))
    print_fn('AUPRO0.30: {:.4f}'.format(result_list[:, 8].mean()))
