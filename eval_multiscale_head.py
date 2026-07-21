# eval_multiscale_head.py
# -----------------------------------------------------------------------------
# NOVELTY (premise diag): LEARNED per-pixel multi-scale routing.
#
# eval_multiscale đã chốt: các scale bổ khuyết per-pixel (ORACLE +0.147 AUPRO/+0.113 SegF1
#   vs single) NHƯNG toán tử CỐ ĐỊNH thất bại — max_z khuếch đại nhiễu FP (SegF1 sập),
#   mean_z pha loãng. Cần HỌC "tin scale nào ở đâu" thay vì toán tử cứng.
#
# Ý tưởng: train 1 router NHỎ trên vector đa-scale mỗi pixel [z_1..z_K + min/max/mean/std]
#   bằng 10 shot (defect=1 / normal=0, có GT của shot). Router học đúng cái ORACLE làm
#   (triệt normal chọn lọc + nâng defect) nhưng LABEL-FREE lúc test. = mở rộng head few-shot
#   SoftPRO sang đa-scale. TEST trên eval (KHÔNG chứa shot).
#
# So — CẢ 2 metric — single*(scale đơn tốt nhất) / mean_z(SuperADD) / max_z / ROUTER / oracle.
# PASS = router > mean_z VÀ > single* trên CẢ HAI, tiến gần oracle => learned routing bắt được
#        headroom => build đa-scale head vào infer = đóng góp chính paper.
# FAIL = router ~ mean_z => routing không học được từ 10 shot => bỏ, lấy (A) uniform 3:24.
#
#   python eval_multiscale_head.py --data_path ../data --out_dir ./msh --scales 2:28 3:24 4:24
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, score_grid, gt_grid, up_to, VALID, IMG_EXT, SMOOTH_RES,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def aupro05(maps, gts):
    sp = np.array([float(m.max()) for m in maps])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(maps), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def segf1_ksig(maps, gts, k):
    P = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    thr = float(P.mean() + k * P.std())
    TP = FP = FN = 0.0
    for m, g in zip(maps, gts):
        pred = m >= thr; gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
    return 2 * TP / (2 * TP + FP + FN + 1e-9)


def metrics(maps, gts, k):
    return aupro05(maps, gts), segf1_ksig(maps, gts, k)


def scale_all_maps(bb, cat, T, gt, args, layers, gk, device, ds, shot_idx, all_idx):
    """Fused-map @SMOOTH_RES cho MỌI ảnh trong all_idx, head train trên shot_idx. None nếu head fail."""
    args.tiles, args.grid_tile = T, gt
    R = gt * bb.patch; hw = args.head_w
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train and len(tr) > args.max_train:
        tr = tr[:args.max_train]
    bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)
    head = build_head(bb, ds, shot_idx, bank, args, layers, device)
    if head is None:
        return None
    recs = [score_grid(bb, Image.open(ds.img_paths[i]), bank, head, args, layers, device) for i in all_idx]
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in recs]), [1, 99])
    out = []
    for d, pr in recs:
        fused = (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + (hw * pr if pr is not None else 0)
        out.append(up_to(fused, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32))
    return out


def pix_feats(zmaps_one):
    """zmaps_one: list[K] HxW (z của 1 ảnh) -> [H*W, K+4] (z_s + min/max/mean/std qua scale)."""
    st = np.stack(zmaps_one)                                       # [K,H,W]
    extra = np.stack([st.min(0), st.max(0), st.mean(0), st.std(0)])  # [4,H,W]
    Fm = np.concatenate([st, extra], 0)                           # [K+4,H,W]
    return Fm.reshape(Fm.shape[0], -1).T.astype(np.float32)       # [H*W, K+4]


class Router(nn.Module):
    def __init__(self, D, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_router(X, y, steps, lr, device):
    Xt = torch.tensor(X, device=device); yt = torch.tensor(y, device=device).float()
    r = Router(X.shape[1]).to(device)
    opt = torch.optim.Adam(r.parameters(), lr=lr)
    npos = float(yt.sum()); nneg = float((yt == 0).sum())
    w = torch.where(yt > 0, torch.tensor(nneg / max(npos, 1.0), device=device), torch.tensor(1.0, device=device))
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(r(Xt), yt, weight=w)
        loss.backward(); opt.step()
    r.eval()
    return r


def run_cat(bb, cat, args, layers, gk, device):
    rng = np.random.default_rng(args.seed)
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shot_idx = bad[:args.shots]
    eval_idx = [i for i in bad if i not in set(shot_idx)][:args.max_eval]
    eval_idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    all_idx = shot_idx + eval_idx
    ns = len(shot_idx)

    cfgs = [(int(c.split(':')[0]), int(c.split(':')[1])) for c in args.scales]
    maps_scales, kept = [], []
    for cstr, (T, gt) in zip(args.scales, cfgs):
        m = scale_all_maps(bb, cat, T, gt, args, layers, gk, device, ds, shot_idx, all_idx)
        if m is not None:
            maps_scales.append(m); kept.append(cstr)
    if len(maps_scales) < 2:
        return None
    S = len(maps_scales)

    # z-chuẩn hóa per scale bằng μ,σ pooled của phần EVAL (transductive, normal-dominant)
    Z = []
    for s in range(S):
        P = np.concatenate([maps_scales[s][ns + n].reshape(-1) for n in range(len(eval_idx))])
        mu, sd = float(P.mean()), float(P.std() + 1e-8)
        Z.append([(m - mu) / sd for m in maps_scales[s]])         # aligned với all_idx

    shot_gts = [gt_grid(ds.gt_paths[i], 1, SMOOTH_RES).astype(np.uint8) for i in shot_idx]
    eval_gts = [gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8) for i in eval_idx]
    k = args.thr_sigma

    # ---- train router trên SHOT ----
    Xs, ys = [], []
    for n in range(ns):
        Xs.append(pix_feats([Z[s][n] for s in range(S)]))
        ys.append(shot_gts[n].reshape(-1))
    X = np.concatenate(Xs); y = np.concatenate(ys).astype(np.float32)
    pos = np.where(y > 0)[0]; neg = np.where(y == 0)[0]
    if pos.size < 3:
        return None
    neg = neg[rng.integers(0, neg.size, min(neg.size, args.router_nneg))]
    sel = np.concatenate([pos, neg])
    router = train_router(X[sel], y[sel], args.router_steps, args.router_lr, device)

    # ---- áp trên EVAL ----
    router_maps = []
    with torch.no_grad():
        for n in range(len(eval_idx)):
            Fm = pix_feats([Z[s][ns + n] for s in range(S)])
            sc = torch.sigmoid(router(torch.tensor(Fm, device=device))).cpu().numpy()
            router_maps.append(sc.reshape(SMOOTH_RES, SMOOTH_RES).astype(np.float32))

    # ---- baselines trên EVAL (dùng z) ----
    Ze = [[Z[s][ns + n] for s in range(S)] for n in range(len(eval_idx))]   # [N][S]
    singles = [metrics([Z[s][ns + n] for n in range(len(eval_idx))], eval_gts, k) for s in range(S)]
    mean_maps = [np.mean(np.stack(Ze[n]), 0) for n in range(len(eval_idx))]
    max_maps = [np.max(np.stack(Ze[n]), 0) for n in range(len(eval_idx))]
    orc_maps = [np.where(eval_gts[n].astype(bool), np.stack(Ze[n]).max(0), np.stack(Ze[n]).min(0))
                for n in range(len(eval_idx))]

    return {'scales': kept,
            'single': max(singles, key=lambda ab: ab[0] + ab[1]),
            'mean_z': metrics(mean_maps, eval_gts, k),
            'max_z':  metrics(max_maps, eval_gts, k),
            'router': metrics(router_maps, eval_gts, k),
            'oracle': metrics(orc_maps, eval_gts, k)}


def main():
    ap = argparse.ArgumentParser('eval_multiscale_head: learned per-pixel scale routing')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--scales', type=str, nargs='+', default=['2:28', '3:24', '4:24'])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400, help='steps head SoftPRO đơn-scale')
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--router_steps', type=int, default=600)
    ap.add_argument('--router_lr', type=float, default=5e-3)
    ap.add_argument('--router_nneg', type=int, default=30000)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./msh')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('msh', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} scales={args.scales} layers={layers} head_w={args.head_w} k={args.thr_sigma}')

    # dùng router_* qua args (train_router đọc args.steps/n_neg gián tiếp qua tham số truyền)
    variants = ['single', 'mean_z', 'max_z', 'router', 'oracle']
    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        s = '  '.join(f'{v}=({r[v][0]:.3f}/{r[v][1]:.3f})' for v in variants)
        p(f'  [{cat:11s}] {s}')
    if not res:
        return

    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig) qua category =====')
    m = {v: (float(np.mean([res[c][v][0] for c in res])), float(np.mean([res[c][v][1] for c in res]))) for v in variants}
    for v in variants:
        d = '' if v == 'single' else f'   ΔAUPRO={m[v][0]-m["single"][0]:+.4f}  ΔSegF1={m[v][1]-m["single"][1]:+.4f}'
        tag = {'single': '  <- scale đơn', 'mean_z': '  (SuperADD)', 'router': '  <- LEARNED ROUTER',
               'oracle': '  <- TRẦN'}.get(v, '')
        p(f'  {v:7s}: AUPRO0.05={m[v][0]:.4f}  SegF1={m[v][1]:.4f}{d}{tag}')

    p('\nĐỌC (theo Δ THẬT, đừng tin template):')
    p(' - router > mean_z VÀ > single* trên CẢ HAI => learned routing bắt được headroom oracle')
    p('   => novelty nhấc cả hai => build đa-scale head vào infer.')
    p(' - router ~ mean_z => 10 shot không đủ học routing => bỏ, lấy uniform 3:24.')
    p(' - khoảng cách router -> oracle = phần còn lại (thêm scale / router mạnh hơn).')


if __name__ == '__main__':
    main()
