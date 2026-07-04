# diag22_shiftaug_premise.py
# -----------------------------------------------------------------------------
# PREMISE TEST cho SHIFT-AUG FEW-SHOT — lever AUPRO DUY NHẤT còn lại sau 2 negative.
#
# BỐI CẢNH (đã chốt): diag20 (bank-adapt) + diag21 (per-image norm) đều FAIL cho AUPRO
#   -> AUPRO dưới shift KHÔNG cứu được bằng test-time UNSUPERVISED (khớp diag10/11:
#      rare-normal chỉ SUPERVISED-recoverable). Lever AUPRO = few-shot/SoftPRO head.
#   Câu hỏi private thật: head few-shot (train trên shot SẠCH) có TRANSFER dưới shift không?
#
# METHOD: augment 10 shot bằng photometric_shift khi train head:
#   - Xpos: pixel-defect trích từ shot ở nhiều mức shift (clean + n_aug mức ngẫu nhiên)
#     -> head học boundary defect BẤT BIẾN với sáng.
#   - Xneg: normal-pixel của shot ĐÃ SHIFT thêm vào negative (dạy "shifted-normal vẫn là normal").
#   Kỳ vọng: head-AUG transfer tốt hơn head-NOAUG khi eval dưới shift -> nâng AUPRO fused.
#
# Đối chứng (eval LUÔN dưới shift dị-nhất per-image, mô phỏng private; fuse global-norm khớp submit):
#   CLEAN-REF   : eval KHÔNG shift, head-NOAUG, fuse hw  -> trần "public".
#   SHIFT DIST  : eval shift, chỉ distance (hw=0)         -> sàn (bất khả unsup, ~diag20/21).
#   SHIFT NOAUG : eval shift, head-NOAUG (train sạch)     -> pipeline hiện tại dưới private.
#   SHIFT AUG   : eval shift, head-AUG (train shift-aug)  -> method.
#   + HEAD-ONLY (hw=1) NOAUG vs AUG: cô lập khả năng transfer của RIÊNG head.
#   SegF1 báo ở ngưỡng self-cal percentile (lever SegF1 đã chốt ở diag20/21).
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag22_shiftaug_premise.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shots 10 --n_aug 3 --shift_lo 0.3 --shift_hi 1.2 \
#     --head_w 0.6 --out_dir ./diag22
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
from PIL import Image, ImageEnhance
from scipy.ndimage import label as cc_label

from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ===== core: y hệt train_softpro / diag21 ====================================
def photometric_shift(pil, s):
    if s <= 0:
        return pil
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    return [pil.crop((round(j * w / T), round(i * h / T), round((j + 1) * w / T), round((i + 1) * h / T)))
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
    G = grid.shape[0]
    C = grid.shape[-1]
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
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST))
                   for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * 0.01))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    r = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return r[7]   # AUPRO0.05


def segf1_fixed(maps, gts, thr, gk, device, resize=256):
    ss, yy = [], []
    for m, g in zip(maps, gts):
        ss.append(upmap(m, resize, gk, device).reshape(-1))
        yy.append((np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) > 0)
                  .reshape(-1).astype(np.uint8))
    s = np.concatenate(ss)
    y = np.concatenate(yy)
    if y.sum() == 0:
        return 0.0
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / (prec + rec + 1e-12)


class Head(nn.Module):
    def __init__(self, C, mu, sd):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)
        self.lin = nn.Linear(C, 1)

    def forward(self, x):
        return self.lin((x - self.mu) / self.sd).squeeze(-1)


def train_softpro(head, Xpos, region_ids, Xneg, steps, lr, q, temp, w_bce, w_fp, device):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    rids = torch.tensor(region_ids, device=device)
    uniq = torch.unique(rids)
    y = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    wbce = torch.cat([torch.full((len(Xpos),), len(Xneg) / max(1, len(Xpos))), torch.ones(len(Xneg))]).to(device)
    Xall = torch.cat([Xpos, Xneg], 0)
    for _ in range(steps):
        opt.zero_grad()
        sp = head(Xpos)
        sn = head(Xneg)
        tau = torch.quantile(sn.detach(), q)
        scale = temp * (sn.detach().std() + 1e-6)
        rec = []
        for r in uniq:
            m = rids == r
            rec.append(torch.sigmoid((sp[m] - tau) / scale).mean())
        region_recall = torch.stack(rec).mean()
        fp_excess = torch.sigmoid((sn - tau) / scale).mean()
        bce = F.binary_cross_entropy_with_logits(head(Xall), y, weight=wbce)
        loss = (1 - region_recall) + w_fp * fp_excess + w_bce * bce
        loss.backward()
        opt.step()
    return head


def main():
    ap = argparse.ArgumentParser('diag22 — shift-augmented few-shot head (AUPRO transfer under shift)')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--n_aug', type=int, default=3, help='số bản shift-augment mỗi shot (ngoài bản clean)')
    ap.add_argument('--shift_lo', type=float, default=0.3)
    ap.add_argument('--shift_hi', type=float, default=1.2)
    ap.add_argument('--head_w', type=float, default=0.6, help='trọng số fuse head (khớp submit)')
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--selfcal_pct', type=float, default=98.0)
    ap.add_argument('--max_eval', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag22')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag22', args.out_dir).info
    torch.manual_seed(args.seed)

    bb = load_backbone(args.model, device)
    R = args.grid_tile * bb.patch
    if args.layers_fixed or not bb.n_layers:
        layers = [ly for ly in args.layers if ly < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(ly / 12 * bb.n_layers))) for ly in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles
    gt = args.grid_tile
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    shift_rng = np.random.default_rng(args.seed + 12345)

    p('=' * 116)
    p(f'DIAG22 shift-aug few-shot | model={args.model} eff_grid={T*gt} layers={layers} | k={args.shots} '
      f'n_aug={args.n_aug} shift~U[{args.shift_lo},{args.shift_hi}] hw={hw} steps={args.steps}')
    p('CLEAN-REF | SHIFT DIST(hw0) | SHIFT NOAUG(head train sạch) | SHIFT AUG(head shift-aug) | +HEAD-ONLY(hw1)')
    p('=' * 116)

    agg = {k: [] for k in ['clean_au', 'clean_f1', 'sh_dist_au', 'noaug_au', 'noaug_f1',
                           'aug_au', 'aug_f1', 'noaug_head_au', 'aug_head_au']}

    for cat in args.categories:
        # ---- bank train (clean) ----
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        acc = []
        keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            buf = []
            for pth in tr:
                buf.extend(tile_pils(Image.open(pth), T))
                while len(buf) >= args.enc_batch:
                    b = torch.stack([to_tensor(t, R) for t in buf[:args.enc_batch]])
                    buf = buf[args.enc_batch:]
                    f = bb.extract(b, layers)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), args.enc_batch * keep).cpu())
            if buf:
                f = bb.extract(torch.stack([to_tensor(t, R) for t in buf]), layers)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        Cdim = bank.shape[-1]
        mu = bank.mean(0, keepdim=True)
        sd = bank.std(0, keepdim=True) + 1e-6
        neg_bank = subsample(bank, args.n_neg).detach()

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        rng.shuffle(good)
        shot_pool = bad[:args.shots]
        eval_bad = bad[args.shots:args.shots + args.max_eval]
        eval_idx = eval_bad + good[:args.max_eval // 2]

        # ---- eval: feature CLEAN + SHIFT dị-nhất per-image ----
        ev_feat_c, ev_feat_s, ev_d_c, ev_d_s, ev_gt, s_list = [], [], [], [], [], []
        for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval'):
            pil = Image.open(ds.img_paths[i])
            s_i = float(shift_rng.uniform(args.shift_lo, args.shift_hi))
            s_list.append(s_i)
            gc = img_featmap(bb, pil, T, R, gt, layers, args.enc_batch)
            gsf = img_featmap(bb, photometric_shift(pil, s_i), T, R, gt, layers, args.enc_batch)
            ev_feat_c.append(gc.cpu().numpy())
            ev_feat_s.append(gsf.cpu().numpy())
            ev_d_c.append(nn_map(gc, bank, device))
            ev_d_s.append(nn_map(gsf, bank, device))
            ev_gt.append(gt_grid(ds.gt_paths[i], ds.labels[i], gc.shape[0]))

        # ---- shots -> Xpos/region_ids/Xneg, cho NOAUG (chỉ clean) và AUG (clean + n_aug shift) ----
        def collect_shots(n_aug):
            pos_list, rid_list, negsh_list, rbase = [], [], [], 0
            for i in shot_pool:
                pil0 = Image.open(ds.img_paths[i])
                shifts = [0.0] + [float(shift_rng.uniform(args.shift_lo, args.shift_hi)) for _ in range(n_aug)]
                for s in shifts:
                    pil = photometric_shift(pil0, s) if s > 0 else pil0
                    g = img_featmap(bb, pil, T, R, gt, layers, args.enc_batch).cpu().numpy()
                    G = g.shape[0]
                    m = gt_grid(ds.gt_paths[i], 1, G)
                    lab, n = cc_label(m)
                    flat = g.reshape(-1, Cdim)
                    for rid in range(1, n + 1):
                        idxs = (lab.reshape(-1) == rid)
                        if idxs.sum() == 0:
                            continue
                        pos_list.append(flat[idxs])
                        rid_list.append(np.full(int(idxs.sum()), rbase))
                        rbase += 1
                    if s > 0:                                    # normal-pixel ĐÃ SHIFT -> negative aug
                        negsh_list.append(flat[m.reshape(-1) == 0])
            return pos_list, rid_list, negsh_list, rbase

        pos_n, rid_n, _, rb_n = collect_shots(0)             # NOAUG: chỉ clean
        pos_a, rid_a, negsh_a, rb_a = collect_shots(args.n_aug)   # AUG: clean + shift
        if rb_n < 1 or rb_a < 1:
            p(f'  [{cat}] thiếu defect region, bỏ')
            continue

        Xpos_n = torch.tensor(np.concatenate(pos_n), device=device)
        Xpos_a = torch.tensor(np.concatenate(pos_a), device=device)
        Xneg_n = neg_bank
        if negsh_a:
            neg_sh = subsample(torch.tensor(np.concatenate(negsh_a), device=device), args.n_neg // 2)
            Xneg_a = torch.cat([neg_bank, neg_sh], 0)
        else:
            Xneg_a = neg_bank

        head_noaug = train_softpro(Head(Cdim, mu, sd).to(device), Xpos_n, np.concatenate(rid_n), Xneg_n,
                                   args.steps, args.lr, args.q, args.temp, args.w_bce, args.w_fp, device)
        head_aug = train_softpro(Head(Cdim, mu, sd).to(device), Xpos_a, np.concatenate(rid_a), Xneg_a,
                                 args.steps, args.lr, args.q, args.temp, args.w_bce, args.w_fp, device)

        def head_scores(head, feat_list):
            head.eval()
            out = []
            with torch.no_grad():
                for f in feat_list:
                    G = f.shape[0]
                    s = torch.sigmoid(head(torch.tensor(f.reshape(-1, Cdim), device=device))).reshape(G, G).cpu().numpy()
                    out.append(s)
            return out

        pr_noaug_c = head_scores(head_noaug, ev_feat_c)
        pr_noaug_s = head_scores(head_noaug, ev_feat_s)
        pr_aug_s = head_scores(head_aug, ev_feat_s)

        # global-norm distance (khớp submit): clean và shift chuẩn riêng
        def gn(dlist):
            dd = np.stack(dlist)
            lo, hi = np.percentile(dd, 1), np.percentile(dd, 99)
            return [(d - lo) / (hi - lo + 1e-8) for d in dlist]
        dr_c = gn(ev_d_c)
        dr_s = gn(ev_d_s)

        def fuse_eval(drl, prl, w):
            maps = [(1 - w) * dr + w * pr for dr, pr in zip(drl, prl)]
            au = region_metrics(maps, ev_gt, gk, device)
            pooled = np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in maps])
            thr = float(np.percentile(pooled, args.selfcal_pct))
            f1 = segf1_fixed(maps, ev_gt, thr, gk, device)
            return au, f1

        clean_au, clean_f1 = fuse_eval(dr_c, pr_noaug_c, hw)
        sh_dist_au, _ = fuse_eval(dr_s, pr_noaug_s, 0.0)             # hw=0 -> distance-only
        noaug_au, noaug_f1 = fuse_eval(dr_s, pr_noaug_s, hw)
        aug_au, aug_f1 = fuse_eval(dr_s, pr_aug_s, hw)
        noaug_head_au, _ = fuse_eval(dr_s, pr_noaug_s, 1.0)         # hw=1 -> head-only
        aug_head_au, _ = fuse_eval(dr_s, pr_aug_s, 1.0)

        for kk, vv in [('clean_au', clean_au), ('clean_f1', clean_f1), ('sh_dist_au', sh_dist_au),
                       ('noaug_au', noaug_au), ('noaug_f1', noaug_f1), ('aug_au', aug_au), ('aug_f1', aug_f1),
                       ('noaug_head_au', noaug_head_au), ('aug_head_au', aug_head_au)]:
            agg[kk].append(vv)

        p(f'  [{cat:<11}] AUPRO05 clean={clean_au:.3f} dist={sh_dist_au:.3f} NOAUG={noaug_au:.3f} '
          f'AUG={aug_au:.3f} ({aug_au-noaug_au:+.3f}) | head-only NOAUG={noaug_head_au:.3f} AUG={aug_head_au:.3f} '
          f'| SegF1 NOAUG={noaug_f1:.3f} AUG={aug_f1:.3f}')

    p('\n' + '=' * 116)
    M = {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    p('{:<18}{:>12}{:>12}'.format('condition', 'AUPRO0.05', 'SegF1'))
    p('{:<18}{:>12.4f}{:>12.4f}'.format('CLEAN-REF', M['clean_au'], M['clean_f1']))
    p('{:<18}{:>12.4f}{:>12}'.format('SHIFT DIST(hw0)', M['sh_dist_au'], '-'))
    p('{:<18}{:>12.4f}{:>12.4f}'.format('SHIFT NOAUG', M['noaug_au'], M['noaug_f1']))
    p('{:<18}{:>12.4f}{:>12.4f}'.format('SHIFT AUG', M['aug_au'], M['aug_f1']))
    p('{:<18}{:>12.4f}{:>12.4f}'.format('  head-only NOAUG', M['noaug_head_au'], float('nan')))
    p('{:<18}{:>12.4f}{:>12.4f}'.format('  head-only AUG', M['aug_head_au'], float('nan')))

    drop = M['clean_au'] - M['noaug_au']
    rec_fused = M['aug_au'] - M['noaug_au']
    rec_head = M['aug_head_au'] - M['noaug_head_au']
    rec_f1 = M['aug_f1'] - M['noaug_f1']
    close = M['aug_au'] - M['clean_au']
    v_fail = 'PASS' if drop > 0.02 else 'FAIL'
    v_trans = 'PASS' if rec_fused > 0.01 else 'FAIL'
    v_head = 'PASS' if rec_head > 0.01 else 'FAIL'
    p('=' * 116)
    p(f'PREMISE FAILURE   (shift hạ pipeline hiện tại)      : {v_fail}  (dAUPRO05={-drop:+.3f})')
    p(f'PREMISE TRANSFER-AUPRO (shift-aug nâng fused)        : {v_trans}  ({rec_fused:+.3f})')
    p(f'PREMISE TRANSFER-HEAD  (shift-aug nâng head-only)    : {v_head}  ({rec_head:+.3f})')
    p(f'INFO SegF1 aug vs noaug = {rec_f1:+.3f} | AUG còn cách trần CLEAN {close:+.3f}')
    p('=> BUILD shift-aug nếu FAILURE + (TRANSFER-AUPRO hoặc TRANSFER-HEAD) PASS.')
    p('   Nếu cả hai FAIL: shift đẩy defect-feature ra ngoài cả head shift-aug -> feature backbone bất-biến-sáng')
    p('     mới là gốc (đổi backbone/normalize input), few-shot head không đủ.')

    with open(os.path.join(args.out_dir, 'diag22.csv'), 'w') as fcsv:
        fcsv.write('condition,AUPRO05,SegF1\n')
        fcsv.write(f'clean_ref,{M["clean_au"]:.4f},{M["clean_f1"]:.4f}\n')
        fcsv.write(f'shift_dist,{M["sh_dist_au"]:.4f},\n')
        fcsv.write(f'shift_noaug,{M["noaug_au"]:.4f},{M["noaug_f1"]:.4f}\n')
        fcsv.write(f'shift_aug,{M["aug_au"]:.4f},{M["aug_f1"]:.4f}\n')
        fcsv.write(f'headonly_noaug,{M["noaug_head_au"]:.4f},\n')
        fcsv.write(f'headonly_aug,{M["aug_head_au"]:.4f},\n')
    p(f'Đã lưu: {os.path.join(args.out_dir, "diag22.csv")}')


if __name__ == '__main__':
    main()
