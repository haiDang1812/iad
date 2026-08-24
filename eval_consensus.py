# eval_consensus.py
# -----------------------------------------------------------------------------
# METHOD candidate (fair, unsup, non-huge): operator ĐỒNG THUẬN ĐA TẦNG.
#
#   Pipeline hiện tại: bb.extract AVERAGE các layer -> 1 feature -> NN. Average = tổ hợp
#   CỘNG TÍNH: một layer lệch mạnh đủ kéo score lên. rare-normal thường chỉ lệch ở VÀI tầng
#   (vd màu/texture lạ nhưng cấu trúc bình thường) -> average cho điểm cao -> FP ở FPR thấp
#   -> dập AUPRO0.05.
#
#   Đổi operator: NN-distance RIÊNG từng layer, z-score per-layer (label-free, fair), rồi hợp
#   bằng ĐỒNG THUẬN (phi cộng tính) thay vì cộng:
#     defect      = lệch manifold ở MỌI tầng  -> mọi layer đồng ý cao -> giữ.
#     rare-normal = lệch chỉ VÀI tầng         -> đồng thuận thấp      -> dập (hết FP).
#
#   KHÁC nối/average: đó là additive (một tầng đủ kéo lên). min/quantile là AND mềm -> đòi
#   NHIỀU tầng cùng đồng ý -> đúng cơ chế lọc rare-normal. Không nằm trong họ đã chết nào
#   (density/subspace/aggregation/shift/geometry đều khác).
#
#   Biến thể (cùng bank per-layer, biến duy nhất = cách HỢP layer):
#     avg   = NN trên feature TRUNG BÌNH (= pipeline hiện tại), z-score      [BASELINE]
#     meanz = mean_l z_l   (additive trên z per-layer)                       [additive control]
#     minz  = min_l z_l    (AND cứng: đòi mọi tầng đồng ý)                    [consensus]
#     q25z  = quantile_0.25_l z_l  (AND mềm)                                  [consensus]
#
#   FAIR: bank chỉ train/good; GT chỉ để chấm. KHÔNG head/nhãn/shot/train.
#   ĐỌC: minz/q25z > avg RÕ ở AUPRO0.05 (và > meanz) -> đồng thuận (AND-ness) là cơ chế thật
#        = novelty method. ~avg -> layer DINOv3 tương quan quá cao -> loại dứt.
#
#   python eval_consensus.py --data_path ../data --out_dir ./consensus --max_eval 30 \
#       --categories can sheet_metal fruit_jelly vial fabric rice wallplugs walnuts
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
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import (build_bank, tile_pils, to_tensor, IMG_EXT)                 # noqa: E402
from diag30_thin_premise import eval_sgrids, norm01                                            # noqa: E402
from dataset import MVTecAD2Dataset                                                             # noqa: E402
from utils import get_gaussian_kernel, get_logger                                               # noqa: E402
from backbones_ext import load_backbone                                                         # noqa: E402

warnings.filterwarnings('ignore')
VARIANTS = ['avg', 'meanz', 'minz', 'q25z']


@torch.no_grad()
def _extract_stack(bb, imgs, layers):
    """GIỐNG bb.extract nhưng KHÔNG average -> giữ per-layer [B,L,N,C].
    Inline ở đây để probe CHỈ phụ thuộc 1 file (không cần sync backbones_ext)."""
    out = bb.model(pixel_values=imgs.to(bb.device), output_hidden_states=True)
    hs = out.hidden_states
    skip = 1 + bb.n_reg
    feats = [hs[min(ll, len(hs) - 1)][:, skip:, :] for ll in layers]
    return torch.stack(feats, dim=1)                         # [B, L, N, C]


def img_featgrid_stack(bb, pil, T, R, gt, layers, eb):
    """GIỐNG img_featgrid nhưng giữ per-layer -> [L, G, G, C]."""
    tiles = tile_pils(pil, T)
    parts = []
    for s in range(0, len(tiles), eb):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + eb]])
        parts.append(_extract_stack(bb, b, layers))          # [b, L, N, C]
    f = torch.cat(parts, 0)                                   # [T*T, L, N, C]
    L, C = f.shape[1], f.shape[-1]
    grid = torch.zeros(L, T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[:, i * gt:(i + 1) * gt, j * gt:(j + 1) * gt, :] = f[k, :, :gt * gt, :].reshape(L, gt, gt, C)
    return grid                                               # [L, G, G, C]


@torch.no_grad()
def build_bank_stack(bb, tr, T, R, gt, layers, eb, bank_size, device):
    """Bank RIÊNG từng layer từ train/good. Trả list[L] tensor (Ni,C) trên device."""
    L = len(layers)
    keep = max(64, bank_size * 4 // max(1, len(tr) * T * T))
    pools = [[] for _ in range(L)]
    for pth in tr:
        g = img_featgrid_stack(bb, Image.open(pth), T, R, gt, layers, eb)     # [L,G,G,C]
        C = g.shape[-1]
        for l in range(L):
            p = g[l].reshape(-1, C)
            idx = torch.randperm(p.shape[0], device=p.device)[:keep]
            pools[l].append(p[idx].cpu())
        del g
    torch.cuda.empty_cache()
    banks = []
    for l in range(L):
        b = torch.cat(pools[l], 0)
        if b.shape[0] > bank_size:
            b = b[torch.randperm(b.shape[0])[:bank_size]]
        banks.append(b.to(device))
    return banks


@torch.no_grad()
def _nn_z(q, bank, chunk=2048):
    """min NN-distance của q (Np,C) tới bank, rồi z-score per-image (fair). Trả (Np,)."""
    Np = q.shape[0]
    d = torch.empty(Np, device=q.device)
    for s in range(0, Np, chunk):
        d[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(dim=1).values
    return (d - d.mean()) / d.std().clamp_min(1e-6)


@torch.no_grad()
def score_consensus(gstack, banks, gavg, bank_avg, device):
    L, G, _, C = gstack.shape
    Np = G * G
    z = torch.empty(L, Np, device=device)
    for l in range(L):
        z[l] = _nn_z(gstack[l].reshape(Np, C).to(device), banks[l])
    out = {}
    out['avg'] = _nn_z(gavg.reshape(Np, C).to(device), bank_avg)        # baseline hiện tại
    out['meanz'] = z.mean(0)                                            # additive control
    out['minz'] = z.min(0).values                                      # AND cứng
    out['q25z'] = z.quantile(0.25, dim=0)                              # AND mềm
    return {k: v.reshape(G, G).cpu().numpy() for k, v in out.items()}


def run_cat(bb, cat, args, layers, gk, device, p, rng):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    sizes = [(Image.open(ds.img_paths[i]).size[1], Image.open(ds.img_paths[i]).size[0]) for i in idx]
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={T * gt_} L={len(layers)}')

    banks = build_bank_stack(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    bank_avg = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    p(f'    [{cat}] bank per-layer={banks[0].shape[0]} avg={bank_avg.shape[0]}')

    raws = {v: [] for v in VARIANTS}
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
            pil = Image.open(ds.img_paths[i])
            gstack = img_featgrid_stack(bb, pil, T, R, gt_, layers, args.enc_batch)
            gavg = gstack.mean(0)                                       # feature trung bình = pipeline hiện tại
            sv = score_consensus(gstack, banks, gavg, bank_avg, device)
            for v in VARIANTS:
                raws[v].append(sv[v])
            del gstack, gavg
    del banks, bank_avg; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        sg, _ = norm01(raws[v])
        m = eval_sgrids(sg, sizes, idx, ds, args.canvas, gk, args.aupro_res, args.thr_sigma, device, rng)
        out[v] = m
        db = '' if v == 'avg' else f'   Δaupro={m["aupro"] - out["avg"]["aupro"]:+.4f}'
        p(f'    [{cat}] {v:6s}: AUPRO0.05={m["aupro"]:.4f}  SegF1={m["segf1"]:.4f}  trần={m["segf1_max"]:.4f}{db}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_consensus: cross-layer consensus operator (fair, unsup)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split (fair). Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./consensus')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('consensus', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} layers={layers} '
      f'bank={args.bank_size} aupro_res={args.aupro_res} k={args.thr_sigma}')
    p('  FAIR: bank CHỈ train/good; GT chỉ để chấm. avg=baseline (NN feature trung bình = pipeline hiện tại).')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (AUPRO0.05 / SegF1 / trần) — avg=baseline =====')
    base = float(np.mean([res[c]['avg']['aupro'] for c in res]))
    for v in VARIANTS:
        au = float(np.mean([res[c][v]['aupro'] for c in res]))
        f1 = float(np.mean([res[c][v]['segf1'] for c in res]))
        fm = float(np.mean([res[c][v]['segf1_max'] for c in res]))
        db = '' if v == 'avg' else f'   Δ={au - base:+.4f}'
        p(f'  {v:6s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}  trần={fm:.4f}{db}')
    p('\n  Per-cat (biến thể tốt nhất theo AUPRO0.05):')
    for c in res:
        bv = max(VARIANTS, key=lambda vv: res[c][vv]['aupro'])
        r = res[c][bv]
        p(f'    [{c:11s}] best={bv:6s} AUPRO0.05={r["aupro"]:.4f} (avg={res[c]["avg"]["aupro"]:.3f})  SegF1={r["segf1"]:.4f}')
    p('\nĐỌC: minz/q25z>avg RÕ (và >meanz) -> đồng thuận đa tầng = novelty method. ~avg -> loại.')


if __name__ == '__main__':
    main()
