# train_lowfpr_head.py
# -----------------------------------------------------------------------------
# METHOD (thesis #1+#2): head phân biệt nhẹ trên FROZEN DINOv2, học bằng
# SYNTHETIC ANOMALY (SimpleNet/GLASS-style) + LOSS NHẮM LOW-FPR (pAUC surrogate).
#
# Động lực (từ diagnosis + 5 negative của ta):
#   - D9: tín hiệu phân biệt nằm ở chiều phương sai thấp -> distance Euclid bỏ qua.
#     Synthetic anomaly cho "hướng" để học chiều đó mà KHÔNG cần nhãn lỗi thật.
#   - 5 negative: mọi enhancement đổi low-FPR lấy broad recall. Method này tối ưu
#     TRỰC TIẾP low-FPR: loss pAUC chỉ phạt mạnh các patch NORMAL điểm cao
#     (= false-positive ở FPR thấp), thay vì tối ưu phân tách trung bình (BCE/hinge).
#
# So sánh trong 1 lần chạy: loss 'hinge' (chuẩn) vs 'pauc' (của ta).
# Mốc cần vượt: DIST Euclid PatchCore-DINOv2 = AUPRO0.05 0.4357.
#
# KHÔNG cần checkpoint INP-Former (encoder frozen). Train head nhỏ trên feature cache.
#
# Chạy:
#   python train_lowfpr_head.py --data_path ../data --out_dir ./method_lowfpr
# -----------------------------------------------------------------------------

import os
import math
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dataset import MVTecAD2Dataset, get_data_transforms
from models import vit_encoder
from utils import ader_evaluator, get_gaussian_kernel, get_logger, setup_seed

warnings.filterwarnings("ignore")

VALID_CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice',
                    'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP',
                'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
DEFAULT_LAYERS = {'base': [2, 3, 4, 5, 6, 7, 8, 9],
                  'large': [5, 11, 17, 23]}
LOSSES = ['hinge', 'pauc']   # hinge = baseline; pauc = method (low-FPR-targeted)


def get_arch(enc):
    for a in ('small', 'base', 'large'):
        if a in enc:
            return a
    raise ValueError(enc)


@torch.no_grad()
def extract_feature(encoder, img, layers, n_reg):
    x = encoder.prepare_tokens(img)
    feats, last = [], max(layers)
    for i, blk in enumerate(encoder.blocks):
        if i <= last:
            x = blk(x)
        if i in layers:
            feats.append(x[:, 1 + n_reg:, :])
    return torch.stack(feats, dim=1).mean(dim=1)   # [B, N, C]


class DiscHead(nn.Module):
    """Adaptor (Linear) + discriminator MLP. Output: anomaly logit/patch (cao = bất thường)."""
    def __init__(self, C, hidden=256):
        super().__init__()
        self.adaptor = nn.Linear(C, C)
        self.disc = nn.Sequential(
            nn.Linear(C, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.disc(self.adaptor(x)).squeeze(-1)


def compute_loss(s_norm, s_anom, mode, margin, topk):
    # normal -> điểm THẤP (< -margin), anomaly -> điểm CAO (> margin)
    l_anom = F.relu(margin - s_anom).mean()
    if mode == 'hinge':
        l_norm = F.relu(margin + s_norm).mean()
    elif mode == 'pauc':
        # CHỈ phạt mạnh top-k% normal điểm cao = false-positive ở low-FPR
        k = max(1, int(s_norm.numel() * topk))
        s_top = torch.topk(s_norm, k)[0]
        l_norm = F.relu(margin + s_top).mean()
    else:
        raise ValueError(mode)
    return l_norm + l_anom


def sp_score_from_map(m, max_ratio=0.01):
    flat = m.reshape(-1)
    k = max(1, int(flat.shape[0] * max_ratio))
    return np.sort(flat)[::-1][:k].mean()


def train_and_eval(args, device, cat, encoder, layers, n_reg, gk, loss_mode, print_fn):
    setup_seed(1)
    dtf, gtf = get_data_transforms(args.input_size, args.crop_size)
    train_data = ImageFolder(root=os.path.join(args.data_path, cat, 'train'), transform=dtf)
    test_data = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                                transform=dtf, gt_transform=gtf, phase='test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 1) Cache feature train-normal
    print_fn(f'  [{cat}/{loss_mode}] caching train features ({len(train_data)} ảnh)...')
    tf = []
    with torch.no_grad():
        for img, _ in tqdm(train_loader, ncols=80, desc='  train-feat'):
            tf.append(extract_feature(encoder, img.to(device), layers, n_reg).cpu())
    tf = torch.cat(tf, 0)                       # [Ntr, N, C]
    C = tf.shape[-1]
    flat = tf.reshape(-1, C)                    # [M, C]

    # Standardize (mean/std từ train) -> noise có thang đo nhất quán
    mu = flat.mean(0, keepdim=True)
    sd = flat.std(0, keepdim=True) + 1e-6
    flat_std = ((flat - mu) / sd)               # [M, C] trên CPU
    M = flat_std.shape[0]

    # 2) Train head nhỏ trên feature cache (encoder frozen, rất nhanh)
    head = DiscHead(C, args.hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-5)
    head.train()
    for it in tqdm(range(args.iters), ncols=80, desc=f'  train-{loss_mode}'):
        idx = torch.randint(0, M, (args.feat_bs,))
        f = flat_std[idx].to(device)
        fa = f + args.noise_std * torch.randn_like(f)   # synthetic anomaly
        s_n = head(f)
        s_a = head(fa)
        loss = compute_loss(s_n, s_a, loss_mode, args.margin, args.pauc_topk)
        opt.zero_grad(); loss.backward(); opt.step()

    # 3) Test
    head.eval()
    mu_d, sd_d = mu.to(device), sd.to(device)
    pr_maps, gt_maps = [], []
    with torch.no_grad():
        for img, gt, label, _ in tqdm(test_loader, ncols=80, desc='  test'):
            img = img.to(device)
            feats = extract_feature(encoder, img, layers, n_reg)     # [B,N,C]
            side = int(math.sqrt(feats.shape[1]))
            fs = (feats - mu_d) / sd_d
            s = head(fs.reshape(-1, C)).reshape(feats.shape[0], 1, side, side)
            s = F.interpolate(s, size=args.resize_mask, mode='bilinear', align_corners=False)
            s = gk(s)
            gt = F.interpolate(gt, size=args.resize_mask, mode='nearest')
            gt[gt > 0.5] = 1; gt[gt <= 0.5] = 0
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]
            pr_maps.append(s[:, 0].cpu().numpy())
            gt_maps.append(gt[:, 0].cpu().numpy())

    pr = np.concatenate(pr_maps, 0)
    gt = np.concatenate(gt_maps, 0).astype(np.uint8)
    pr_sp = np.array([sp_score_from_map(m, args.max_ratio) for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    r = ader_evaluator(pr, pr_sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    print_fn(f'  [{cat}/{loss_mode}] ' + ' '.join(f'{n}:{v:.4f}' for n, v in zip(METRIC_NAMES, r)))
    return r


def main():
    ap = argparse.ArgumentParser('Low-FPR synthetic-anomaly head trên frozen DINOv2')
    ap.add_argument('--data_path', type=str, default=r'E:\dataset\mvtecad2')
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    ap.add_argument('--layers', type=int, nargs='+', default=None)
    ap.add_argument('--input_size', type=int, default=448)
    ap.add_argument('--crop_size', type=int, default=392)
    ap.add_argument('--resize_mask', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_ratio', type=float, default=0.01)
    # head / train
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--feat_bs', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--noise_std', type=float, default=0.2, help='thang đo synthetic anomaly (đv std)')
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--pauc_topk', type=float, default=0.1, help='tỉ lệ normal điểm-cao bị phạt (loss pauc)')
    ap.add_argument('--losses', type=str, nargs='+', default=LOSSES,
                    help='hinge=baseline, pauc=method low-FPR')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID_CATEGORIES)
    ap.add_argument('--out_dir', type=str, default='./method_lowfpr')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "\n❌ CUDA không khả dụng (torch.cuda.is_available()=False).\n"
            "   Lỗi 'driver too old' = PyTorch trong .venv build cho CUDA mới hơn driver.\n"
            "   adeval (EvalAccumulatorCuda) BẮT BUỘC GPU, nên không chạy CPU được.\n"
            "   Fix: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
            "   rồi kiểm tra: python -c \"import torch; print(torch.cuda.is_available())\"\n")
    device = 'cuda:0'
    arch = get_arch(args.encoder)
    layers = args.layers if args.layers is not None else DEFAULT_LAYERS[arch]
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger('lowfpr', args.out_dir)
    print_fn = logger.info
    print_fn('=' * 70)
    print_fn(f'LOW-FPR HEAD | encoder={args.encoder} | layers={layers} | losses={args.losses}')
    print_fn(f'iters={args.iters} noise_std={args.noise_std} margin={args.margin} pauc_topk={args.pauc_topk}')
    print_fn('=' * 70)

    encoder = vit_encoder.load(args.encoder).to(device).eval()
    n_reg = getattr(encoder, 'num_register_tokens', 0)
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    all_res = {lm: {} for lm in args.losses}
    for cat in args.categories:
        for lm in args.losses:
            all_res[lm][cat] = train_and_eval(args, device, cat, encoder, layers, n_reg, gk, lm, print_fn)

    print_fn('\n' + '=' * 70)
    print_fn('MEAN theo loss')
    print_fn('{:<8} '.format('') + ' '.join(f'{m:>10}' for m in METRIC_NAMES))
    for lm in args.losses:
        mean_r = np.array(list(all_res[lm].values())).mean(0)
        print_fn('{:<8} '.format(lm) + ' '.join(f'{v:>10.4f}' for v in mean_r))

    csv = os.path.join(args.out_dir, 'results.csv')
    with open(csv, 'w') as f:
        f.write('loss,category,' + ','.join(METRIC_NAMES) + '\n')
        for lm in args.losses:
            for cat, r in all_res[lm].items():
                f.write(f'{lm},{cat},' + ','.join(f'{v:.4f}' for v in r) + '\n')
            mean_r = np.array(list(all_res[lm].values())).mean(0)
            f.write(f'{lm},MEAN,' + ','.join(f'{v:.4f}' for v in mean_r) + '\n')
    print_fn(f'\nĐã lưu: {csv}')


if __name__ == '__main__':
    main()
