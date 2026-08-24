# eval_subspace.py
# -----------------------------------------------------------------------------
# NOVELTY candidate: LOCAL SUBSPACE RECONSTRUCTION thay NN-distance (fair, unsup, non-huge).
#   Chẩn đoán: rare-normal (texture normal THẬT nhưng hiếm) xa MỌI điểm bank đơn lẻ -> NN cao
#   -> FP ở FPR thấp -> dập AUPRO0.05. NHƯNG rare-normal vẫn TRÊN manifold normal -> tái tạo
#   được từ tổ hợp hàng xóm bank. defect = ngoài manifold -> KHÔNG tổ hợp normal nào dựng lại.
#   NN không phân biệt "xa Euclid" vs "ngoài manifold". Subspace-residual thì CÓ:
#     score = khoảng cách từ q tới BAO AFFINE của k hàng xóm bank gần nhất.
#       rare-normal -> trong bao -> dư NHỎ (hết FP);  defect -> ngoài bao -> dư LỚN (giữ).
#   dư² = ‖v‖² − vᵀM(MᵀM)⁻¹Mᵀv ,  v=q−b₁,  M=[b₂−b₁,…,bₖ−b₁]  (hệ (k−1)² tí hon, batch).
#
#   FAIR: chỉ bank train/good; GT chỉ để chấm. KHÔNG head/nhãn/shot/train.
#   KHÁC 8 lever đã chết: chúng reweight ĐƠN ĐIỆU cùng NN (cùng thứ hạng); cái này là HÀM ĐIỂM
#   khác -> ĐỔI thứ hạng (rare-normal tụt dưới defect) = đúng cái AUPRO0.05 cần.
#
# Biến thể (cùng bank, biến duy nhất = hàm điểm):
#   nn     = min NN distance                       (baseline = cả leaderboard AD2)
#   aff{k} = subspace residual, bao affine k-NN    (k = 8, 16)
#   gap    = nn − aff8  (chẩn đoán: lớn = rare-normal nhiều; chỉ để soi, không phải score)
#
# ĐỌC: aff{k} > nn RÕ ở AUPRO0.05 -> cơ chế "ngoài-manifold" ĐÚNG = novelty carry paper.
#      ~ nn / âm -> rare-normal-FP không phải chuyện manifold -> loại dứt.
#
#   python eval_subspace.py --data_path ../data --out_dir ./subspace --max_eval 30 \
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
from infer_submit_mvtec_ad2 import build_bank, img_featgrid, IMG_EXT                    # noqa: E402
from diag30_thin_premise import eval_sgrids, norm01                                     # noqa: E402
from dataset import MVTecAD2Dataset                                                      # noqa: E402
from utils import get_gaussian_kernel, get_logger                                        # noqa: E402
from backbones_ext import load_backbone                                                  # noqa: E402

warnings.filterwarnings('ignore')
KLIST = [8, 16]
VARIANTS = ['nn'] + [f'aff{k}' for k in KLIST] + [f'cvx{k}' for k in KLIST] + ['gap']


@torch.no_grad()
def affine_residual(v, M, eps=1e-3):
    """Dư của v (m,C) tới span các hướng M (m,r,C).  dư² = ‖v‖² − vᵀMᵀ(MMᵀ)⁻¹Mv.
    Ổn định số: ridge TƯƠNG ĐỐI theo scale của G (feature norm lớn -> eps tuyệt đối vô nghĩa),
    và solve_ex không ném lỗi -> hàng xóm trùng nhau (texture lặp) => proj=0 (dư = full norm)."""
    G = torch.bmm(M, M.transpose(1, 2))                                # (m,r,r) = M Mᵀ
    r = G.shape[1]
    I = torch.eye(r, device=G.device).unsqueeze(0)
    diag_mean = G.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-6)  # (m,) scale mỗi hệ
    G = G + (eps * diag_mean).view(-1, 1, 1) * I                       # ridge tương đối -> luôn PD
    rhs = torch.bmm(M, v.unsqueeze(2))                                 # (m,r,1) = M v
    a, info = torch.linalg.solve_ex(G, rhs)                            # không raise; info!=0 = hỏng
    proj_sq = (rhs * a).sum(dim=(1, 2))                                # vᵀMᵀ(MMᵀ)⁻¹Mv
    bad = (info != 0) | ~torch.isfinite(proj_sq)
    if bad.any():
        proj_sq = torch.where(bad, torch.zeros_like(proj_sq), proj_sq)  # suy biến -> chiếu 0
    res_sq = (v * v).sum(1) - proj_sq
    return res_sq.clamp_min(0).sqrt()                                  # (m,)


@torch.no_grad()
def convex_residual(q, nb, n_iter=20, eps=1e-8):
    """Dư của q (m,C) tới BAO LỒI của k hàng xóm nb (m,k,C) — simplex-constrained LSQ (Frank-Wolfe).
    q = Σ wᵢ bᵢ, wᵢ≥0, Σwᵢ=1 -> q phải NỘI SUY mẫu thật. KHÁC affine: affine cho ngoại suy tự do
    (cost 0 dọc hướng hợp lệ), convex PHẠT ngoại suy -> defect vượt biên dữ liệu bị bắt kể cả khi
    nằm dọc hướng tiếp tuyến. rare-normal = nội suy -> dư nhỏ; defect = ngoại suy -> dư lớn."""
    m, k, C = nb.shape
    w = torch.full((m, k), 1.0 / k, device=nb.device, dtype=nb.dtype)
    for _ in range(n_iter):
        recon = torch.einsum('mk,mkc->mc', w, nb)                     # (m,C) tổ hợp lồi hiện tại
        r = q - recon                                                 # (m,C) dư
        grad = -2.0 * torch.einsum('mkc,mc->mk', nb, r)               # (m,k) ∇_w ‖r‖²
        j = grad.argmin(dim=1)                                        # (m,) đỉnh Frank-Wolfe
        s = torch.zeros_like(w).scatter_(1, j.unsqueeze(1), 1.0)      # (m,k) one-hot
        d = s - w                                                     # (m,k) hướng về đỉnh
        delta = torch.einsum('mk,mkc->mc', d, nb)                     # (m,C) đổi recon
        num = (r * delta).sum(1)
        den = (delta * delta).sum(1).clamp_min(eps)
        gamma = (num / den).clamp(0.0, 1.0)                           # line-search bậc 2 chính xác
        w = w + gamma.unsqueeze(1) * d
    recon = torch.einsum('mk,mkc->mc', w, nb)
    return (q - recon).norm(dim=1)                                    # (m,)


@torch.no_grad()
def score_variants(grid, bank, device, chunk=2048):
    G = grid.shape[0]; C = grid.shape[-1]
    q = grid.reshape(-1, C)
    Np = q.shape[0]
    kmax = max(KLIST)
    out = {v: torch.empty(Np, device=device) for v in VARIANTS}
    for s in range(0, Np, chunk):
        qc = q[s:s + chunk]                                            # (m,C)
        d = torch.cdist(qc, bank)                                      # (m,N)
        vals, idx = d.topk(kmax, largest=False)                        # (m,kmax)
        out['nn'][s:s + chunk] = vals[:, 0]
        nb = bank[idx]                                                 # (m,kmax,C)
        b1 = nb[:, 0, :]                                               # (m,C) neo = NN
        v = qc - b1                                                    # (m,C)
        for k in KLIST:
            M = nb[:, 1:k, :] - b1.unsqueeze(1)                        # (m,k-1,C) hướng
            out[f'aff{k}'][s:s + chunk] = affine_residual(v, M)
            out[f'cvx{k}'][s:s + chunk] = convex_residual(qc, nb[:, :k, :])  # bao lồi k hàng xóm
        out['gap'][s:s + chunk] = out['nn'][s:s + chunk] - out[f'aff{KLIST[0]}'][s:s + chunk]
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
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={T * gt_} K={KLIST}')

    bank = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    p(f'    [{cat}] bank={bank.shape[0]}')

    raws = {v: [] for v in VARIANTS}
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
            g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
            sv = score_variants(g, bank, device)
            for v in VARIANTS:
                raws[v].append(sv[v])
            del g
    del bank; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        sg, _ = norm01(raws[v])
        m = eval_sgrids(sg, sizes, idx, ds, args.canvas, gk, args.aupro_res, args.thr_sigma, device, rng)
        out[v] = m
        db = '' if v == 'nn' else f'   Δaupro={m["aupro"] - out["nn"]["aupro"]:+.4f}'
        p(f'    [{cat}] {v:6s}: AUPRO0.05={m["aupro"]:.4f}  SegF1={m["segf1"]:.4f}  trần={m["segf1_max"]:.4f}{db}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_subspace: local subspace reconstruction (fair, unsup)')
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
    ap.add_argument('--out_dir', type=str, default='./subspace')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('subspace', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} layers={layers} K={KLIST} '
      f'bank={args.bank_size} aupro_res={args.aupro_res} k={args.thr_sigma}')
    p('  FAIR: bank CHỈ train/good; GT chỉ để chấm. nn=baseline (NN-distance = cả leaderboard AD2).')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (AUPRO0.05 / SegF1 / trần) — nn=baseline =====')
    for v in VARIANTS:
        au = float(np.mean([res[c][v]['aupro'] for c in res]))
        f1 = float(np.mean([res[c][v]['segf1'] for c in res]))
        fm = float(np.mean([res[c][v]['segf1_max'] for c in res]))
        db = '' if v == 'nn' else f'   Δ={au - float(np.mean([res[c]["nn"]["aupro"] for c in res])):+.4f}'
        p(f'  {v:6s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}  trần={fm:.4f}{db}')
    p('\n  Per-cat (biến thể tốt nhất theo AUPRO0.05):')
    for c in res:
        bv = max([x for x in VARIANTS if x != 'gap'], key=lambda vv: res[c][vv]['aupro'])
        r = res[c][bv]
        p(f'    [{c:11s}] best={bv:6s} AUPRO0.05={r["aupro"]:.4f} (nn={res[c]["nn"]["aupro"]:.3f})  SegF1={r["segf1"]:.4f}')
    p('\nĐỌC: aff{k}>nn rõ -> cơ chế ngoài-manifold ĐÚNG = novelty. ~nn -> loại dứt.')


if __name__ == '__main__':
    main()
