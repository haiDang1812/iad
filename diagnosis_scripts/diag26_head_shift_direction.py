# diag26_head_shift_direction.py
# -----------------------------------------------------------------------------
# CÂU HỎI: vì sao head few-shot mua được +0.252 AUPRO0.05 trên public nhưng chỉ chuyển
#   giao 6% sang private (+0.016), và ÂM trên SegF1 (0.3515 -> 0.3306)?
#
# GIẢ THUYẾT (ghép 2 kết quả đã đo của chính ta):
#   - D9: chiều phân biệt normal/defect là chiều PHƯƠNG SAI THẤP của frozen feature.
#   - shift-gap ~= 1.0 (eval_target_subspace): private normal KHÔNG xa bank hơn public normal.
#   NN-distance bị chi phối bởi các chiều phương sai CAO -> light-shift gần như không đụng.
#   Head đọc đúng chiều phương sai THẤP -> một dịch chuyển tuyệt đối rất nhỏ (không đủ làm
#   nhúc nhích NN-distance) vẫn là NHIỀU σ theo σ của chính chiều đó. => head trôi, distance không.
#
# PHÉP ĐO (trên private THẬT, không cần GT, không đốt lượt submit):
#   u_head = hướng hiệu dụng của head. head(x) = w·((x-mu)/sd) => u_head ∝ w/sd.
#   u_pc1.. = top eigvec của Cov(bank)  (chiều phương sai CAO).
#   δ       = mean(feat private) − mean(feat validation/good)   (vector shift thật)
#
#   drift theo σ của CHÍNH chiều đó:   |δ·û| / σ_bank(û)
#     - u_head  : kỳ vọng LỚN
#     - u_pc1..k: kỳ vọng NHỎ
#     - u_rand  : baseline
#   Kèm 2 số kiểm chứng chéo: shift-gap (median NN-dist priv/val) và độ trôi logit theo σ_val.
#
# PASS  = drift_head >> drift_pc1 VÀ shift_gap ~ 1.0  -> cơ chế được xác nhận.
# FAIL  = drift_head ~ drift_pc1                      -> head trôi vì lý do khác, bỏ giả thuyết.
#
#   python diag26_head_shift_direction.py --data_path ../data --out_dir ./diag26
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
    build_bank, build_head, img_featgrid, nn_map, list_split_files,
    VALID, SPLITS, IMG_EXT,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_logger                                            # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')


@torch.no_grad()
def collect(bb, paths, args, layers, n_per_img, device, desc):
    """Gom feature patch (đã subsample) + NN-dist thô của một tập ảnh."""
    T, gt, R = args.tiles, args.grid_tile, args.grid_tile * bb.patch
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    feats = []
    for pth in tqdm(paths, ncols=70, desc=desc, leave=False):
        grid = img_featgrid(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
        f = grid.reshape(-1, grid.shape[-1])
        idx = torch.randperm(f.shape[0], generator=g)[:n_per_img].to(f.device)
        feats.append(f[idx])
    return torch.cat(feats, 0)


@torch.no_grad()
def median_nndist(bb, paths, bank, args, layers, device, desc):
    T, gt, R = args.tiles, args.grid_tile, args.grid_tile * bb.patch
    vals = []
    for pth in tqdm(paths, ncols=70, desc=desc, leave=False):
        grid = img_featgrid(bb, Image.open(pth), T, R, gt, layers, args.enc_batch)
        vals.append(np.asarray(nn_map(grid, bank, device)).reshape(-1))
    return float(np.median(np.concatenate(vals)))


def drift_sigma(delta, u, X):
    """|δ·û| đo bằng σ của phân bố X dọc chính û."""
    u = u / (u.norm() + 1e-12)
    return float((delta @ u).abs() / (X @ u).std().clamp_min(1e-12))


def run_cat(bb, cat, args, layers, device):
    # KHÔNG @torch.no_grad() ở đây: build_head phải train (loss.backward()).
    T, gt = args.tiles, args.grid_tile
    rng_np = np.random.default_rng(args.seed)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr, T, gt * bb.patch, gt, layers, args.enc_batch, args.bank_size, device)

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng_np.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None

    val = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e)) for e in IMG_EXT], []))[:args.n_val]
    priv = []
    for s in SPLITS:
        files, _ = list_split_files(args.data_path, cat, s)
        if files:
            priv += files[:args.n_priv]
    if not val or not priv:
        return None

    Fv = collect(bb, val, args, layers, args.px_per_img, device, f'    {cat}/val')
    Fp = collect(bb, priv, args, layers, args.px_per_img, device, f'    {cat}/priv')

    with torch.no_grad():                                           # head đã train xong -> khỏi dựng graph
        delta = Fp.mean(0) - Fv.mean(0)                             # vector shift thật

        # --- hướng hiệu dụng của head: logit = w·((x-mu)/sd) = (w/sd)·x + const ---
        u_head = (head.lin.weight.squeeze(0) / head.sd.squeeze(0)).float()
        # --- chiều phương sai CAO của bank ---
        Bc = bank - bank.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(Bc, q=max(args.n_pc, 8), center=False)
        pcs = [V[:, i] for i in range(args.n_pc)]
        gcpu = torch.Generator(device='cpu').manual_seed(args.seed)
        u_rand = torch.randn(bank.shape[1], generator=gcpu).to(device)

        d_head = drift_sigma(delta, u_head, bank)
        d_pcs = [drift_sigma(delta, u, bank) for u in pcs]
        d_rand = drift_sigma(delta, u_rand, bank)

        # head có thật sự là chiều phương sai THẤP không? (kiểm D9 trên head ĐÃ train)
        sig = lambda u: float((bank @ (u / (u.norm() + 1e-12))).std())    # noqa: E731
        var_ratio = sig(u_head) / max(sig(pcs[0]), 1e-12)

        # --- kiểm chứng chéo: shift-gap + độ trôi logit ---
        gap = (median_nndist(bb, priv, bank, args, layers, device, f'    {cat}/gap-p')
               / max(median_nndist(bb, val, bank, args, layers, device, f'    {cat}/gap-v'), 1e-9))
        lv, lp = head(Fv), head(Fp)
        logit_drift = float((lp.mean() - lv.mean()) / lv.std().clamp_min(1e-9))

    return {'d_head': d_head, 'd_pc1': d_pcs[0], 'd_pcs': d_pcs, 'd_rand': d_rand,
            'var_ratio': var_ratio, 'shift_gap': gap, 'logit_drift': logit_drift}


def main():
    ap = argparse.ArgumentParser('diag26: shift nằm ở chiều nào, cắn nhánh nào?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--n_val', type=int, default=30)
    ap.add_argument('--n_priv', type=int, default=40, help='ảnh mỗi private split')
    ap.add_argument('--px_per_img', type=int, default=400)
    ap.add_argument('--n_pc', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag26')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag26', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} layers={layers}')
    p('drift = |δ·û| / σ_bank(û)  — dịch chuyển nguồn->đích đo bằng σ của CHÍNH chiều đó.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        pcs = ' '.join(f'{v:.2f}' for v in r['d_pcs'])
        p(f'  [{cat}] drift_head={r["d_head"]:.2f}σ  drift_pc[1..{args.n_pc}]=[{pcs}]  drift_rand={r["d_rand"]:.2f}σ')
        p(f'          σ_head/σ_pc1={r["var_ratio"]:.3f}   shift_gap={r["shift_gap"]:.3f}   '
          f'logit_drift={r["logit_drift"]:+.2f}σ_val')

    if not res:
        return
    g = lambda k: float(np.mean([res[c][k] for c in res]))                                # noqa: E731
    p('\n' + '=' * 78 + '\n===== MEAN qua category =====')
    p(f'  drift_head = {g("d_head"):.2f}σ      <- head đọc chiều này')
    p(f'  drift_pc1  = {g("d_pc1"):.2f}σ      <- NN-distance sống ở đây')
    p(f'  drift_rand = {g("d_rand"):.2f}σ      <- baseline')
    p(f'  σ_head/σ_pc1 = {g("var_ratio"):.3f}  (<<1 => head ĐÚNG là chiều phương sai thấp, khớp D9)')
    p(f'  shift_gap = {g("shift_gap"):.3f}    (~1.0 => distance đứng yên dưới shift)')
    p(f'  logit_drift = {g("logit_drift"):+.2f}σ_val')

    ratio = g('d_head') / max(g('d_pc1'), 1e-9)
    p(f'\n  drift_head / drift_pc1 = {ratio:.1f}x')
    p('\nĐỌC:')
    p('  PASS (ratio >> 1, shift_gap ~ 1.0): shift né NN-distance nhưng cắn thẳng chiều head đọc.')
    p('    => photproj thất bại vì đặt phép chiếu lên nhánh DISTANCE (nơi không có gì để bỏ).')
    p('    => đặt nó lên INPUT CỦA HEAD mới đúng chỗ. Cùng 1 phép chiếu, 2 nhánh, 2 kết cục')
    p('       -> negative result cũ thành control experiment.')
    p('  FAIL (ratio ~ 1): head trôi vì lý do khác (overfit 10 shot, scale sigmoid...) -> bỏ giả thuyết,')
    p('    quay về thuần threshold-transfer (diag25).')


if __name__ == '__main__':
    main()
