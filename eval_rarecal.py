# eval_rarecal.py
# -----------------------------------------------------------------------------
# ĐÓNG GÓP candidate: RARITY-CALIBRATED anomaly scoring (fair, unsup, non-huge).
#   CHẨN ĐOÁN ĐỘC QUYỀN của tụi mình: ở low-FPR (AUPRO0.05) bank-distance thất bại KHÔNG phải
#   vì thiếu tín hiệu defect, mà vì patch NORMAL-nhưng-HIẾM bị chấm khoảng-cách cao -> false pos
#   -> dập đúng AUPRO0.05. => Hiệu chỉnh khoảng cách theo ĐỘ CÔ LẬP CỤC BỘ của bank:
#     m* (điểm bank khớp gần nhất) ở vùng THƯA (rare-normal) -> hạ điểm;
#     vùng DÀY mà test vẫn xa (defect thật)                 -> giữ điểm.
#   FAIR: chỉ dùng bank train/good; GT chỉ để chấm. Không head, không nhãn, không shot.
#
# 4 biến thể cô lập cùng một nguyên lý (đo lever sạch):
#   raw      = min NN distance (baseline = số fair đã đo: can 0.15, sheet 0.47)
#   ratio    = s* / iso(m*)         iso = mean dist m* tới k hàng xóm bank của nó
#   zcal     = s* - iso(m*)         trừ baseline độ-cô-lập vùng
#   reweight = PatchCore-style ổn định số: w=1-1/(1+Σ_nb exp(s*-||q-nb||)); score=w*s*
#              (nb = k hàng xóm bank của m*; m* thưa -> ||q-nb|| lớn -> w->0 -> dập)
#
# ĐỌC:
#   ratio/zcal/reweight > raw RÕ ở AUPRO0.05 -> tiền đề rare-normal-FP ĐÚNG + cơ chế sống
#                                               -> đây là đóng góp để viết (grounded, fair, standard).
#   ~ raw / âm -> tiền đề sai (như LOF diag31 từng rụng) -> báo cáo thẳng, đổi hướng.
#
#   python eval_rarecal.py --data_path ../data --out_dir ./rarecal --max_eval 30 \
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
from infer_submit_mvtec_ad2 import build_bank, img_featgrid, IMG_EXT                   # noqa: E402
from diag30_thin_premise import eval_sgrids, norm01                                    # noqa: E402
from dataset import MVTecAD2Dataset                                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                                       # noqa: E402
from backbones_ext import load_backbone                                                 # noqa: E402

warnings.filterwarnings('ignore')
VARIANTS = ['raw', 'ratio', 'zcal', 'reweight']


@torch.no_grad()
def bank_self_stats(bank, k, device, chunk=2048):
    """Với mỗi điểm bank: idx k hàng xóm bank gần nhất (loại chính nó) + iso = mean dist tới chúng.
    iso lớn = vùng THƯA (rare-normal)."""
    N = bank.shape[0]
    knn_idx = torch.empty(N, k, dtype=torch.long, device=device)
    iso = torch.empty(N, device=device)
    for s in range(0, N, chunk):
        d = torch.cdist(bank[s:s + chunk], bank)                 # (c, N)
        vals, idx = d.topk(k + 1, largest=False)                 # gồm chính nó (dist~0) ở cột 0
        knn_idx[s:s + chunk] = idx[:, 1:k + 1]
        iso[s:s + chunk] = vals[:, 1:k + 1].mean(1)
    return knn_idx, iso


@torch.no_grad()
def score_variants(grid, bank, knn_idx, iso, device, chunk=4096):
    """grid (G,G,C) -> dict{name:(G,G)} cho 4 biến thể. Ổn định số, chunk theo patch."""
    G = grid.shape[0]; C = grid.shape[-1]
    q = grid.reshape(-1, C)
    Np = q.shape[0]
    out = {v: torch.empty(Np, device=device) for v in VARIANTS}
    for s in range(0, Np, chunk):
        qc = q[s:s + chunk]                                      # (c, C)
        d = torch.cdist(qc, bank)                                # (c, N)
        sstar, mi = d.min(1)                                     # NN dist + index m*
        im = iso[mi]                                             # iso(m*)  (c,)
        out['raw'][s:s + chunk] = sstar
        out['ratio'][s:s + chunk] = sstar / (im + 1e-6)
        out['zcal'][s:s + chunk] = sstar - im
        nb = bank[knn_idx[mi]]                                   # (c, k, C) hàng xóm bank của m*
        dn = (qc.unsqueeze(1) - nb).norm(dim=-1)                 # (c, k) = ||q - nb||, luôn >= sstar
        denom = 1.0 + torch.exp((sstar.unsqueeze(1) - dn)).sum(1)  # exp(<=0) in (0,1]
        w = 1.0 - 1.0 / denom
        out['reweight'][s:s + chunk] = w * sstar
    return {v: out[v].reshape(G, G).cpu().numpy() for v in VARIANTS}


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
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={T * gt_} k={args.knn}')

    bank = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    knn_idx, iso = bank_self_stats(bank, args.knn, device)
    p(f'    [{cat}] bank={bank.shape[0]} iso mean={iso.mean():.3f} (thưa cao=rare-normal nhiều)')

    raws = {v: [] for v in VARIANTS}
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
            g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
            sv = score_variants(g, bank, knn_idx, iso, device)
            for v in VARIANTS:
                raws[v].append(sv[v])
            del g
    del bank; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        sg, _ = norm01(raws[v])
        m = eval_sgrids(sg, sizes, idx, ds, args.canvas, gk, args.aupro_res, args.thr_sigma, device, rng)
        out[v] = m
        db = '' if v == 'raw' else f'   Δaupro={m["aupro"] - out["raw"]["aupro"]:+.4f}'
        p(f'    [{cat}] {v:8s}: AUPRO0.05={m["aupro"]:.4f}  SegF1={m["segf1"]:.4f}  trần={m["segf1_max"]:.4f}{db}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_rarecal: rarity-calibrated anomaly scoring (fair, unsup)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--knn', type=int, default=10, help='số hàng xóm bank để đo độ cô lập')
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
    ap.add_argument('--out_dir', type=str, default='./rarecal')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('rarecal', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} layers={layers} knn={args.knn} '
      f'bank={args.bank_size} aupro_res={args.aupro_res} k={args.thr_sigma}')
    p('  FAIR: rarity-calibration CHỈ từ bank train/good; GT chỉ để chấm. raw=baseline.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (AUPRO0.05 / SegF1 / trần) — target ~0.55 fair | raw=baseline =====')
    for v in VARIANTS:
        au = float(np.mean([res[c][v]['aupro'] for c in res]))
        f1 = float(np.mean([res[c][v]['segf1'] for c in res]))
        fm = float(np.mean([res[c][v]['segf1_max'] for c in res]))
        db = '' if v == 'raw' else f'   Δ={au - float(np.mean([res[c]["raw"]["aupro"] for c in res])):+.4f}'
        p(f'  {v:8s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}  trần={fm:.4f}{db}')
    p('\n  Per-cat (biến thể tốt nhất theo AUPRO0.05):')
    for c in res:
        bv = max(VARIANTS, key=lambda vv: res[c][vv]['aupro'])
        r = res[c][bv]
        p(f'    [{c:11s}] best={bv:8s} AUPRO0.05={r["aupro"]:.4f} (raw={res[c]["raw"]["aupro"]:.3f})  '
          f'SegF1={r["segf1"]:.4f}')
    p('\nĐỌC: cal>raw rõ -> rare-normal-FP đúng, cơ chế sống = đóng góp. ~raw -> tiền đề sai, đổi hướng.')


if __name__ == '__main__':
    main()
