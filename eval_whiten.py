# eval_whiten.py
# -----------------------------------------------------------------------------
# LEVER gốc D9: chiều phân biệt normal/defect là chiều PHƯƠNG SAI THẤP của frozen feature.
#   Nhánh distance hiện dùng NN Euclidean THÔ -> bị các chiều phương sai CAO át -> bỏ sót
#   tín hiệu defect (đó là lý do phải gánh bằng head). Nhánh distance lại SHIFT-STABLE
#   (transfer tốt private). => WHITEN feature trước NN (chuẩn hóa phương sai -> phơi bày
#   chiều thấp) sẽ làm distance branch tự thấy defect -> map tốt hơn -> CẢ HAI metric lên,
#   và transfer. Novelty có nguyên lý (khác PaDiM per-position: đây là NN bank đã whiten).
#
# So nhánh distance: plain NN / diag-whiten ((x-μ)/σ) / pca-whiten (full-cov, top-q).
#   Đo trên test_public (có GT): FUSED (head_w) và DIST-only (head_w=0) — cả AUPRO0.05 &
#   SegF1@test_ksig. PASS = whiten nâng CẢ HAI so plain (nhất là DIST-only => nhánh distance
#   thật sự khỏe hơn, không phải head kéo).
#
#   python eval_whiten.py --data_path ../data --out_dir ./whiten --tiles 3 --grid_tile 24
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, gt_grid, up_to, VALID, IMG_EXT, SMOOTH_RES,
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


@torch.no_grad()
def _proj(x, mu, Wproj):
    xc = x - mu
    return xc * Wproj if Wproj.dim() == 1 else xc @ Wproj    # diag (elementwise) vs pca (matmul)


@torch.no_grad()
def nn_white(grid, bank_w, mu, Wproj, device, chunk=4096):
    """NN Euclidean trong không gian đã whiten. grid:[G,G,C] -> [G,G] distance."""
    G = grid.shape[0]; C = grid.shape[-1]
    g = grid.reshape(-1, C).to(device)
    gw = _proj(g, mu, Wproj)                                 # [G*G, q hoặc C]
    out = torch.empty(gw.shape[0], device=device)
    for i in range(0, gw.shape[0], chunk):
        d = torch.cdist(gw[i:i + chunk], bank_w)             # [chunk, Nb]
        out[i:i + chunk] = d.min(1).values
    return out.reshape(G, G).cpu().numpy()


def make_whiten(bank, mode, q, device):
    """Trả (mu, Wproj, bank_w) cho chế độ whiten. plain -> None (dùng nn_map thô)."""
    mu = bank.mean(0, keepdim=True)                          # [1,C]
    if mode == 'plain':
        return None
    if mode == 'diag':
        Wproj = (1.0 / (bank.std(0) + 1e-6))                 # [C] -> whiten elementwise
    else:  # pca-whiten (full-cov, top-q)
        _, S, V = torch.pca_lowrank(bank - mu, q=min(q, bank.shape[1] - 1), center=False)
        scale = S / np.sqrt(max(bank.shape[0] - 1, 1))       # std dọc mỗi PC
        Wproj = V / (scale.unsqueeze(0) + 1e-6)              # [C,q]
    bank_w = _proj(bank, mu, Wproj)
    return mu, Wproj, bank_w


def run_cat(bb, cat, args, layers, gk, device):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch; hw = args.head_w; k = args.thr_sigma
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train and len(tr) > args.max_train:
        tr = tr[:args.max_train]
    bank = build_bank(bb, tr, T, R, gt, layers, args.enc_batch, args.bank_size, device)

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shot_idx = bad[:args.shots]
    head = build_head(bb, ds, shot_idx, bank, args, layers, device)
    if head is None:
        return None
    idx = [i for i in bad if i not in set(shot_idx)][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    gts = [gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8) for i in idx]

    # feature grid + head prob (tính 1 lần), và distance thô để chuẩn hóa lo/hi mỗi biến thể
    grids, prs = [], []
    for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
        g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch)
        grids.append(g)
        C = g.shape[-1]
        with torch.no_grad():
            prs.append(torch.sigmoid(head(g.reshape(-1, C))).reshape(g.shape[0], g.shape[0]).cpu().numpy())

    out = {}
    for mode in args.modes:
        wh = make_whiten(bank, mode, args.pca_q, device)
        dmaps = []
        for g in grids:
            if wh is None:
                dmaps.append(np.asarray(nn_map(g, bank, device)))
            else:
                mu, Wproj, bank_w = wh
                dmaps.append(nn_white(g, bank_w, mu, Wproj, device))
        # chuẩn hóa lo/hi trên chính tập eval (transductive) rồi fuse + upsample
        allc = np.concatenate([d.reshape(-1) for d in dmaps])
        lo, hi = np.percentile(allc, [1, 99])
        for tag, w in [('fuse', hw), ('dist', 0.0)]:
            maps = []
            for d, pr in zip(dmaps, prs):
                fused = (1 - w) * ((d - lo) / (hi - lo + 1e-8)) + w * pr
                maps.append(up_to(fused, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32))
            out[f'{mode}/{tag}'] = metrics(maps, gts, k)
    return out


def main():
    ap = argparse.ArgumentParser('eval_whiten: whitened bank-distance (gốc D9) có nâng cả hai?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--modes', type=str, nargs='+', default=['plain', 'diag', 'pca'])
    ap.add_argument('--pca_q', type=int, default=256, help='số PC cho pca-whiten')
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./whiten')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('whiten', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles*args.grid_tile} layers={layers} head_w={args.head_w} '
      f'modes={args.modes} pca_q={args.pca_q} k={args.thr_sigma}')

    cols = [f'{m}/{t}' for m in args.modes for t in ('fuse', 'dist')]
    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        s = '  '.join(f'{c}=({r[c][0]:.3f}/{r[c][1]:.3f})' for c in cols)
        p(f'  [{cat:11s}] {s}')
    if not res:
        return

    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig) =====')
    m = {c: (float(np.mean([res[k][c][0] for k in res])), float(np.mean([res[k][c][1] for k in res]))) for c in cols}
    base_f = m.get('plain/fuse'); base_d = m.get('plain/dist')
    for c in cols:
        ref = base_f if c.endswith('fuse') else base_d
        d = '' if c.startswith('plain') else f'   Δ={m[c][0]-ref[0]:+.4f}/{m[c][1]-ref[1]:+.4f} vs plain'
        p(f'  {c:12s}: AUPRO0.05={m[c][0]:.4f}  SegF1={m[c][1]:.4f}{d}')

    p('\nĐỌC (Δ thật):')
    p(' - diag/pca *_fuse nâng CẢ HAI so plain/fuse => whitened distance = lever nhấc cả hai,')
    p('   transfer được (nhánh distance shift-stable). Build vào infer (nn_map -> nn_white).')
    p(' - Nhìn *_dist: nếu whiten nâng distance-only MẠNH => nhánh distance thật sự khỏe hơn')
    p('   (không phải head kéo) => bằng chứng D9 + novelty "low-variance discriminative subspace".')
    p(' - Nếu whiten ~ plain hoặc chỉ pca giúp mà diag không => ghi rõ full-cov mới ăn.')


if __name__ == '__main__':
    main()
