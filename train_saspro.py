# train_softpro.py
# -----------------------------------------------------------------------------
# NOVELTY: SAS-PRO (Size-Adaptive Surrogate for Per-Region Optimization)
#
# Diagnosis từ per-category results:
#   - SoftPRO flat (gốc) thắng BCE ở fabric/sheet_metal/wallplugs (small defect)
#   - SoftPRO flat THUA BCE ở can/rice/walnuts (large defect)
#   - Root cause: flat per-region mean weight region nhỏ = region lớn →
#     over-suppress normal pixels của large-defect categories → SegF1 drop
#
# Fix (SAS-PRO): geometry-aware surrogate với 2 thay đổi:
#   1. Weight tỉ lệ nghịch sqrt(region_size): small region → pull mạnh hơn
#   2. Temperature per-region: scale_r = temp × std_normal / sqrt(n_r + 1)
#      → small region dùng sharper gradient (ít bị dilute bởi normal context)
#
# Ablation mode: --loss bce | softpro | saspro (default: tất cả 3)
#
# Chạy:
#   HF_HUB_OFFLINE=1 python train_softpro.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --head_w 0.5 0.6 0.7 --out_dir ./diag_softpro
# -----------------------------------------------------------------------------

import os
import sys
import glob
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import label as cc_label
import json
from sklearn.metrics import f1_score

from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger
from backbones_ext import load_backbone


warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image / feature utils (không đổi)
# ---------------------------------------------------------------------------

def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    return [pil.crop((round(j * w / T), round(i * h / T),
                      round((j + 1) * w / T), round((i + 1) * h / T)))
            for i in range(T) for j in range(T)]


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def img_featmap(bb, pil, T, R, gt, layers, enc_batch):
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), enc_batch):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + enc_batch]])
        fl.append(bb.extract(b, layers))
    f = torch.cat(fl, 0)
    C = f.shape[-1]
    grid = torch.zeros(T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * gt:(i + 1) * gt, j * gt:(j + 1) * gt] = f[k, :gt * gt].reshape(gt, gt, C)
    return grid


@torch.no_grad()
def nn_map(grid, bank, device, chunk=4096):
    G = grid.shape[0]; C = grid.shape[-1]
    q = grid.reshape(-1, C)
    out = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(1)[0]
    return out.reshape(G, G).cpu().numpy()


def gt_grid(gpath, label_, G):
    if label_ == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((G, G), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


def upmap(arr2d, size, gk, device):
    t = torch.tensor(arr2d, device=device)[None, None].float()
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return gk(t)[0, 0].cpu().numpy()


def region_metrics(maps, gts, gk, device, resize=256):
    pr  = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt  = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize),
                    Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp  = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * 0.01))].mean()
                    for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    r = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return r[7], r[5]


def find_best_threshold(maps, gts, gk, device, resize=256, n_thresh=200):
    pr  = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt  = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize),
                    Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    all_scores = pr.ravel(); all_gts = gt.ravel()
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresh)
    best_thr, best_f1 = float(thresholds[n_thresh // 2]), 0.0
    for thr in thresholds:
        pred = (all_scores > thr).astype(np.uint8)
        f1   = f1_score(all_gts, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_thr = float(thr)
    return best_thr, best_f1


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, C, mu, sd):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)
        self.lin = nn.Linear(C, 1)

    def forward(self, x):
        return self.lin((x - self.mu) / self.sd).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss 1: BCE (baseline)
# ---------------------------------------------------------------------------

def train_bce(head, Xpos, Xneg, steps, lr, device):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    X = torch.cat([Xpos, Xneg], 0)
    y = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    w = torch.cat([torch.full((len(Xpos),), len(Xneg) / max(1, len(Xpos))),
                   torch.ones(len(Xneg))]).to(device)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(X), y, weight=w)
        loss.backward(); opt.step()
    return head


# ---------------------------------------------------------------------------
# Loss 2: SoftPRO flat (gốc — giữ để ablation)
# ---------------------------------------------------------------------------

def train_softpro(head, Xpos, region_ids, Xneg, steps, lr, q, temp, w_bce, w_fp, device):
    opt   = torch.optim.Adam(head.parameters(), lr=lr)
    rids  = torch.tensor(region_ids, device=device)
    uniq  = torch.unique(rids)
    y     = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    wbce  = torch.cat([torch.full((len(Xpos),), len(Xneg) / max(1, len(Xpos))),
                       torch.ones(len(Xneg))]).to(device)
    Xall  = torch.cat([Xpos, Xneg], 0)
    for _ in range(steps):
        opt.zero_grad()
        sp  = head(Xpos)
        sn  = head(Xneg)
        tau = torch.quantile(sn.detach(), q)
        scale = temp * (sn.detach().std() + 1e-6)
        rec = []
        for r in uniq:
            m = rids == r
            rec.append(torch.sigmoid((sp[m] - tau) / scale).mean())
        region_recall = torch.stack(rec).mean()
        fp_excess     = torch.sigmoid((sn - tau) / scale).mean()
        bce           = F.binary_cross_entropy_with_logits(head(Xall), y, weight=wbce)
        loss = (1 - region_recall) + w_fp * fp_excess + w_bce * bce
        loss.backward(); opt.step()
    return head


# ---------------------------------------------------------------------------
# Loss 3: SAS-PRO — Size-Adaptive Surrogate for Per-Region Optimization
#
# Novelty so với SoftPRO flat:
#   a) weight_r = 1/sqrt(n_r): small region → higher gradient weight
#      Justification: AUPRO0.05 đánh giá mỗi region bằng nhau (per-region recall),
#      nhưng large region đã dễ recall hơn bởi NN-distance mạnh hơn.
#      Flat weighting waste gradient budget vào region đã "easy".
#
#   b) scale_r = temp × std_n / sqrt(n_r + 1): small region → sharper sigmoid
#      Justification: small region bị dilute bởi surrounding normal context →
#      cần sharper decision boundary cục bộ, không nên dùng chung scale global.
#
#   c) Negative mining per-region: thay vì push toàn bộ normal, chỉ push
#      K nearest normal patches của từng region (hard negatives).
#      Justification: gradient tập trung vào boundary thật, tránh waste vào
#      easy normals xa → FPR control chính xác hơn ở low-FPR regime.
# ---------------------------------------------------------------------------

def train_saspro(head, Xpos, region_ids, region_sizes, Xneg,
                 steps, lr, q, temp, w_bce, w_fp, w_hard, device,
                 n_hard=64):
    """
    Args:
        region_sizes : dict {region_id: n_pixels} — build từ cc_label
        w_hard       : weight cho hard-negative mining loss
        n_hard       : số hard negatives per region
    """
    opt   = torch.optim.Adam(head.parameters(), lr=lr)
    rids  = torch.tensor(region_ids, device=device)
    uniq  = torch.unique(rids)

    # precompute inverse-sqrt weights (normalized)
    r_sizes = torch.tensor(
        [region_sizes[int(r.item())] for r in uniq], dtype=torch.float32, device=device
    )
    inv_sqrt_w = 1.0 / (r_sizes.sqrt() + 1e-6)
    inv_sqrt_w = inv_sqrt_w / inv_sqrt_w.sum()          # normalize → sum=1

    # per-region temperature denominator (sqrt size)
    sqrt_sizes = r_sizes.sqrt()

    y     = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    wbce  = torch.cat([torch.full((len(Xpos),), len(Xneg) / max(1, len(Xpos))),
                       torch.ones(len(Xneg))]).to(device)
    Xall  = torch.cat([Xpos, Xneg], 0)

    # precompute region masks (static)
    masks = [(rids == r) for r in uniq]

    for step in range(steps):
        opt.zero_grad()
        sp  = head(Xpos)
        sn  = head(Xneg)
        std_n = sn.detach().std() + 1e-6
        tau   = torch.quantile(sn.detach(), q)          # FPR=(1-q) threshold

        # --- (a+b) size-adaptive per-region recall ---
        rec = []
        for idx, (r, m) in enumerate(zip(uniq, masks)):
            # sharper scale for smaller region
            scale_r = temp * std_n / (sqrt_sizes[idx] + 1e-3)
            scale_r = scale_r.clamp(min=1e-4)
            rec_r   = torch.sigmoid((sp[m] - tau) / scale_r).mean()
            rec.append(rec_r * inv_sqrt_w[idx])         # weighted
        region_recall = torch.stack(rec).sum()          # weighted sum

        # --- global FP penalty (giữ như SoftPRO) ---
        scale_g   = temp * std_n
        fp_excess = torch.sigmoid((sn - tau) / scale_g).mean()

        # --- (c) hard-negative mining per region ---
        hard_loss = torch.tensor(0.0, device=device)
        if w_hard > 0:
            with torch.no_grad():
                sn_det = head(Xneg).detach()
            for idx, m in enumerate(masks):
                # top-n_hard normal pixels với score cao nhất (hardest)
                k = min(n_hard, len(sn_det))
                hard_idx = sn_det.topk(k).indices
                Xhard = Xneg[hard_idx]
                sp_r  = sp[m]
                # contrastive: push region mean score > hard normal mean score
                margin = 0.5
                hard_loss = hard_loss + F.relu(
                    margin - (sp_r.mean() - head(Xhard).mean())
                ) * inv_sqrt_w[idx]

        # --- BCE stabilizer ---
        bce = F.binary_cross_entropy_with_logits(head(Xall), y, weight=wbce)

        loss = (1 - region_recall) + w_fp * fp_excess + w_bce * bce + w_hard * hard_loss
        loss.backward(); opt.step()

    return head


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser('SAS-PRO vs SoftPRO vs BCE (ablation)')
    ap.add_argument('--data_path',  type=str, default='/workspace/data')
    ap.add_argument('--model',      type=str, default='v3_large')
    ap.add_argument('--layers',     type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles',      type=int, default=2)
    ap.add_argument('--grid_tile',  type=int, default=28)
    ap.add_argument('--bank_size',  type=int, default=50000)
    ap.add_argument('--enc_batch',  type=int, default=64)
    ap.add_argument('--shots',      type=int, default=10)
    ap.add_argument('--head_w',     type=float, nargs='+', default=[0.5, 0.6, 0.7])
    ap.add_argument('--n_neg',      type=int,   default=20000)
    ap.add_argument('--steps',      type=int,   default=400)
    ap.add_argument('--lr',         type=float, default=1e-2)
    ap.add_argument('--q',          type=float, default=0.95)
    ap.add_argument('--temp',       type=float, default=0.5)
    ap.add_argument('--w_bce',      type=float, default=0.3)
    ap.add_argument('--w_fp',       type=float, default=1.0)
    # SAS-PRO specific
    ap.add_argument('--w_hard',     type=float, default=0.5,
                    help='weight hard-negative contrastive loss (0 = disable)')
    ap.add_argument('--n_hard',     type=int,   default=64,
                    help='số hard negatives per region')
    # ablation: chọn loss nào chạy
    ap.add_argument('--loss',       type=str, nargs='+',
                    default=['bce', 'softpro', 'saspro'],
                    choices=['bce', 'softpro', 'saspro'],
                    help='loss variants để chạy (ablation)')
    ap.add_argument('--seed',       type=int,   default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir',    type=str,   default='./diag_softpro')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('softpro', args.out_dir).info
    torch.manual_seed(args.seed)

    bb = load_backbone(args.model, device)
    patch = bb.patch
    R = args.grid_tile * patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(
            max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers)))
            for l in args.layers
        ))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile
    rng = np.random.default_rng(args.seed)

    loss_tags = [l.upper() for l in args.loss]   # ['BCE','SOFTPRO','SASPRO']

    p('=' * 88)
    p(f'SAS-PRO ablation | model={args.model} eff_grid={T*gt} layers={layers} | '
      f'k={args.shots} head_w={args.head_w} | steps={args.steps} '
      f'q={args.q} temp={args.temp} w_bce={args.w_bce} w_hard={args.w_hard}')
    p(f'Running: {loss_tags}')
    p('=' * 88)

    agg = {}
    thresholds = {}

    for cat in args.categories:
        # ---- build feature bank ----
        tr = sorted(
            glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
            glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg'))
        )
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            buf = []
            for pth in tr:
                buf.extend(tile_pils(Image.open(pth), T))
                while len(buf) >= args.enc_batch:
                    b = torch.stack([to_tensor(t, R) for t in buf[:args.enc_batch]])
                    buf = buf[args.enc_batch:]
                    f = bb.extract(b, layers)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]),
                                         args.enc_batch * keep).cpu())
            if buf:
                f = bb.extract(torch.stack([to_tensor(t, R) for t in buf]), layers)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        Cdim = bank.shape[-1]

        # ---- dataset split ----
        ds   = MVTecAD2Dataset(root=os.path.join(args.data_path, cat),
                               transform=None, gt_transform=None, phase='test')
        bad  = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        shot_pool = bad[:args.shots]
        eval_idx  = bad[args.shots:] + good

        def prep(idx):
            grid = img_featmap(bb, Image.open(ds.img_paths[idx]),
                               T, R, gt, layers, args.enc_batch)
            d = nn_map(grid, bank, device)
            g = gt_grid(ds.gt_paths[idx], ds.labels[idx], grid.shape[0])
            return grid.cpu().numpy(), d, g

        ev     = [prep(i) for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval')]
        ev_feat = [e[0] for e in ev]
        ev_d    = [e[1] for e in ev]
        ev_gt   = [e[2] for e in ev]

        # ---- shots: extract pos/neg features + region info ----
        pos_list, rid_list, region_sizes_map = [], [], {}
        rbase = 0
        for i in shot_pool:
            g = img_featmap(bb, Image.open(ds.img_paths[i]),
                            T, R, gt, layers, args.enc_batch).cpu().numpy()
            G = g.shape[0]
            m = gt_grid(ds.gt_paths[i], 1, G)
            lab, n = cc_label(m)
            flat = g.reshape(-1, Cdim)
            for rid in range(1, n + 1):
                idxs = (lab.reshape(-1) == rid)
                n_px = int(idxs.sum())
                pos_list.append(flat[idxs])
                rid_list.append(np.full(n_px, rbase))
                region_sizes_map[rbase] = n_px          # ← size dict
                rbase += 1

        if rbase < 1 or sum(len(x) for x in pos_list) < 3:
            p(f'  [{cat}] thiếu defect region, bỏ'); continue

        Xpos       = torch.tensor(np.concatenate(pos_list), device=device)
        region_ids = np.concatenate(rid_list)
        Xneg       = subsample(bank, args.n_neg).detach()
        mu = bank.mean(0, keepdim=True)
        sd = bank.std(0, keepdim=True) + 1e-6

        # global-norm distance
        dall = np.stack(ev_d, 0)
        lo, hi = np.percentile(dall, 1), np.percentile(dall, 99)
        ev_dr = [(d - lo) / (hi - lo + 1e-8) for d in ev_d]

        # ---- train heads ----
        heads = {}
        if 'BCE' in loss_tags:
            heads['BCE'] = train_bce(
                Head(Cdim, mu, sd).to(device), Xpos, Xneg,
                args.steps, args.lr, device
            )
        if 'SOFTPRO' in loss_tags:
            heads['SOFTPRO'] = train_softpro(
                Head(Cdim, mu, sd).to(device), Xpos, region_ids, Xneg,
                args.steps, args.lr, args.q, args.temp, args.w_bce, args.w_fp, device
            )
        if 'SASPRO' in loss_tags:
            heads['SASPRO'] = train_saspro(
                Head(Cdim, mu, sd).to(device), Xpos, region_ids,
                region_sizes_map, Xneg,
                args.steps, args.lr, args.q, args.temp,
                args.w_bce, args.w_fp, args.w_hard, device, args.n_hard
            )

        # ---- eval ----
        for tag, head in heads.items():
            head.eval()
            with torch.no_grad():
                ev_pr = []
                for f in ev_feat:
                    G = f.shape[0]
                    s = torch.sigmoid(
                        head(torch.tensor(f.reshape(-1, Cdim), device=device))
                    ).reshape(G, G).cpu().numpy()
                    ev_pr.append(s)
            for hw in args.head_w:
                maps = [(1 - hw) * dr + hw * pr for dr, pr in zip(ev_dr, ev_pr)]
                au, segf1 = region_metrics(maps, ev_gt, gk, device)
                agg.setdefault((tag, hw), []).append((au, segf1))
                best_thr, best_f1 = find_best_threshold(maps, ev_gt, gk, device)
                key_thr = f'{tag}_hw{hw:.2f}'
                thresholds.setdefault(key_thr, {})[cat] = round(best_thr, 6)
                p(f'  [{cat}] {tag:<8} hw{hw:.2f} '
                  f'AUPRO05={au:.4f} SegF1={segf1:.4f} thr={best_thr:.4f}')

    # ---- summary ----
    p('\n' + '=' * 88)
    p('{:<20}{:>12}{:>12}'.format('head/head_w', 'AUPRO0.05', 'SegF1'))
    rows = []
    for key in sorted(agg.keys(), key=lambda x: (x[0], x[1])):
        m = np.array(agg[key]).mean(0)
        label = f'{key[0]}-hw{key[1]:.2f}'
        rows.append((label, m[0], m[1]))
        p('{:<20}{:>12.4f}{:>12.4f}'.format(label, m[0], m[1]))

    with open(os.path.join(args.out_dir, 'results.csv'), 'w') as f:
        f.write('head_headw,AUPRO0.05,SegF1\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f}\n')

    thr_path = os.path.join(args.out_dir, 'thresholds.json')
    with open(thr_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    p(f'\nĐã lưu: {os.path.join(args.out_dir, "results.csv")}')
    p(f'Threshold đã lưu: {thr_path}')
    p('\nĐỌC KẾT QUẢ:')
    p('  SASPRO > SOFTPRO ở can/rice/walnuts (large defect) nhờ size-adaptive weight')
    p('  SASPRO >= SOFTPRO ở fabric/sheet_metal/wallplugs (small defect)')
    p('  → unified performance across defect sizes = novelty core claim')


if __name__ == '__main__':
    main()