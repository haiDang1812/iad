# eval_twoscale.py
# -----------------------------------------------------------------------------
# TWO-SCALE Z-ROUTING: fine-scale (3:48) + context-scale (2:36), fuse per-pixel max-z.
#
# BẰNG CHỨNG (autopsy can, đã trả tiền trong các run trước):
#   - can KHÔNG feature-blind: head 10-shot trên đúng feature này AUPRO 0.9474 — nhưng đó
#     là đường cấm (feed GT). Unsup NN-distance ở 3:48 chỉ 0.177.
#   - Res tăng can càng tệ (3:64=0.142 < 3:48 < 3:28=0.235) → không phải sub-cell.
#   - CONTEXT effect: cùng mật độ lấy mẫu (eff_grid 72), 2:36 = 0.3285 vs 3:24 = 0.17-0.19
#     (lặp lại ở 2-tile submit-era: dist-only 0.388). Tile to = pattern phản xạ diện rộng
#     thành feature → defect móp/biến dạng trên kim loại phản quang hiện ra.
#   - Multi-scale bổ khuyết per-pixel đã chứng minh mức oracle (+0.147 AUPRO, eval_multiscale
#     line cũ); mean-fuse pha loãng, low-FPR cần peak-preserving (scalepyramid) → chọn MAX-z.
#
# METHOD (fair, KHÔNG per-cat, không GT):
#   - Mỗi scale chạy ĐÚNG pipeline nền: cand overlap-windows -> coreset 50k -> 1-NN ->
#     overlap-averaging. Cùng 2 scale + cùng rule fuse cho cả 8 cat.
#   - z-stats (mu, sd) per-scale từ pixel heldout train/good. Fuse ở mức grid:
#       maxz = mu3 + sd3 * max(z3, z2up)   (đưa về đơn vị raw scale fine để rule p95x1.15
#                                            giữ nguyên ngữ nghĩa; monotone -> AUPRO/trần
#                                            không đổi bản chất)
#   - Variants: base (3:48) | ctx (2:36 upsample 144) | maxz | meanz (đối chứng dilution).
#   - Threshold per-variant: RULE ĐÓNG BĂNG p95 heldout x 1.15 + closing (không đổi).
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   1) MAXZ VÀO NỀN nếu: Δmean(maxz-base) AUPRO >= +0.010 VÀ Δmean trần >= -0.005.
#   2) Cơ chế context XÁC NHẬN nếu: can AUPRO (ctx hoặc maxz) >= can base + 0.08.
#   3) meanz chỉ để đối chiếu (kỳ vọng thua maxz — tiền lệ low-FPR peak-preserving).
#   4) (2) đậu mà (1) trượt -> đọc per-cat tìm cat bị maxz kéo tụt, KHÔNG tune thêm trong run.
#
#   python eval_twoscale.py --data_path ../data --out_dir ./twoscale --max_eval 30 \
#       --categories can sheet_metal fruit_jelly vial fabric rice wallplugs walnuts
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import IMG_EXT                                         # noqa: E402
from eval_bankmap import coreset                                                   # noqa: E402
from eval_overlapmap import build_cand_overlap, overlap_score                      # noqa: E402
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from diag30_thin_premise import aupro05                                            # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

RULE_P, RULE_G = 95.0, 1.15          # ĐÓNG BĂNG. KHÔNG ĐỔI.
SCALES = [(3, 48), (2, 36)]          # (tiles, grid_tile): fine eff144 + context eff72. CỐ ĐỊNH.
VARIANTS = ['base', 'ctx', 'maxz', 'meanz']


def up_grid(g, G, device):
    t = torch.from_numpy(np.ascontiguousarray(g))[None, None].to(device)
    return F.interpolate(t, size=(G, G), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def fuse_variants(g3, g2u, st):
    (mu3, sd3), (mu2, sd2) = st
    z3 = (g3 - mu3) / sd3
    z2 = (g2u - mu2) / sd2
    return {'base': g3, 'ctx': g2u,
            'maxz': mu3 + sd3 * np.maximum(z3, z2),
            'meanz': mu3 + sd3 * 0.5 * (z3 + z2)}


def run_cat(bb, cat, args, layers, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr
    va = tr[args.max_train:args.max_train + args.n_val] if args.max_train else []
    overlap_warn = False
    if len(va) < 3:
        va, overlap_warn = tr_use[-args.n_val:], True

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | fine={G3} ctx={SCALES[1][0] * SCALES[1][1]} '
      f'heldout={len(va)}{" [TRÙNG BANK]" if overlap_warn else ""}')

    # ---- per-scale: bank -> score heldout + test (tuần tự để đỡ VRAM) ----
    va_g, te_g = {}, {}
    for si, (T, gt) in enumerate(SCALES):
        R = gt * bb.patch
        cand = build_cand_overlap(bb, tr_use, T, R, gt, layers, args.enc_batch, args.cand_size, device).to(device)
        bank = coreset(cand, args.bank_size, device, seed=args.seed)
        del cand; torch.cuda.empty_cache()
        p(f'    [{cat}] scale {T}:{gt} bank(coreset,overlap)={bank.shape[0]} tu cand={args.cand_size}')
        va_g[si] = [overlap_score(bb, Image.open(path), T, R, gt, layers, args.enc_batch, bank, device)
                    for path in tqdm(va, ncols=70, desc=f'    {cat} s{T}:{gt} heldout', leave=False)]
        te_g[si] = [overlap_score(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch, bank, device)
                    for i in tqdm(idx, ncols=70, desc=f'    {cat} s{T}:{gt} test', leave=False)]
        del bank; torch.cuda.empty_cache()

    # ---- z-stats per-scale từ heldout (fair: chỉ train/good) ----
    st = []
    for si in range(len(SCALES)):
        px = np.concatenate([g.ravel() for g in va_g[si]])
        st.append((float(px.mean()), float(px.std()) + 1e-6))
    p(f'    [{cat}] z-stats: fine mu={st[0][0]:.3f} sd={st[0][1]:.3f} | ctx mu={st[1][0]:.3f} sd={st[1][1]:.3f}')

    # ---- heldout: ngưỡng per-variant theo RULE đóng băng ----
    tr_px = {v: [] for v in VARIANTS}
    for k, path in enumerate(va):
        pil = Image.open(path)
        W, H = pil.size
        var = fuse_variants(va_g[0][k], up_grid(va_g[1][k], G3, device), st)
        for v in VARIANTS:
            m = make_map(var[v], args.canvas, gk, (H, W), device)
            tr_px[v].append(m.cpu().numpy().ravel()[::4])
            del m
    thr = {v: float(np.percentile(np.concatenate(tr_px[v]), RULE_P)) * RULE_G for v in VARIANTS}
    del tr_px, va_g

    # ---- test: fuse grid -> Hist(trần) + F1@rule + AUPRO@512 ----
    te_var = [fuse_variants(te_g[0][k], up_grid(te_g[1][k], G3, device), st) for k in range(len(idx))]
    del te_g
    h = {}
    for v in VARIANTS:
        vmin = min(float(d[v].min()) for d in te_var)
        vmax = max(float(d[v].max()) for d in te_var)
        h[v] = Hist(vmin - 0.05, vmax + 0.05)
    mst = {v: np.zeros(3, np.float64) for v in VARIANTS}
    ap_preds = {v: [] for v in VARIANTS}
    ap_gts = []
    for d, i in zip(tqdm(te_var, ncols=70, desc=f'    {cat} maps', leave=False), idx):
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), np.uint8)
            g_ap = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
        else:
            gpil = Image.open(ds.gt_paths[i]).convert('L')
            g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
            g_ap = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
        ap_gts.append(g_ap)
        g_t = torch.from_numpy(g_nat).to(device) > 0
        r = max(1, round(min(H, W) / G3))
        for v in VARIANTS:
            nat = make_map(d[v], args.canvas, gk, (H, W), device)
            h[v].add(nat.cpu().numpy().reshape(-1), g_nat.reshape(-1))
            ap_preds[v].append(make_map(d[v], args.canvas, gk, (args.aupro_res, args.aupro_res), device).cpu().numpy())
            pred = closing(nat > thr[v], 2 * r + 1)
            mst[v] += ((pred & g_t).sum().item(), (pred & ~g_t).sum().item(), ((~pred) & g_t).sum().item())
            del nat, pred
        del g_t
    del te_var

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = {'aupro': aupro05(ap_preds[v], ap_gts, rng), 'f1_max': h[v].f1_max(),
                  'f1': float(2 * tp / (2 * tp + fp + fn + 1e-9))}
        p(f'    [{cat}] {v:6s}: AUPRO0.05={out[v]["aupro"]:.4f}  trần={out[v]["f1_max"]:.4f}  F1@rule={out[v]["f1"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_twoscale: fine 3:48 + context 2:36, fuse max-z per-pixel (fair, không per-cat)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split. Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./twoscale')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('twoscale', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} scales={SCALES} bank={args.bank_size}(coreset,overlap)x2 cand={args.cand_size} '
      f'heldout={args.n_val} | RULE p={RULE_P} g={RULE_G} morph=CÓ | fuse=max-z (z-stats heldout, fair)')
    p('  variants: base(3:48) | ctx(2:36) | maxz | meanz. Cùng 2 scale + cùng rule cho MỌI cat, không GT.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN theo variant =====')
    mean = {}
    for v in VARIANTS:
        mean[v] = {k: float(np.mean([res[c][v][k] for c in res])) for k in ('aupro', 'f1_max', 'f1')}
        p(f'  {v:6s}: AUPRO0.05={mean[v]["aupro"]:.4f}  trần={mean[v]["f1_max"]:.4f}  F1@rule={mean[v]["f1"]:.4f}')
    d = {k: mean['maxz'][k] - mean['base'][k] for k in mean['base']}
    p(f'\n  Δ MAXZ (maxz - base): AUPRO{d["aupro"]:+.4f}  trần{d["f1_max"]:+.4f}  F1{d["f1"]:+.4f}')
    if 'can' in res:
        p(f'  can: base={res["can"]["base"]["aupro"]:.4f}  ctx={res["can"]["ctx"]["aupro"]:.4f}  '
          f'maxz={res["can"]["maxz"]["aupro"]:.4f}  (cơ chế context: cần >= base+0.08)')
    p('\nĐỌC (pre-registered): (1) MAXZ vào nền nếu Δmean AUPRO>=+0.010 VÀ Δmean trần>=-0.005. '
      '(2) Cơ chế context xác nhận nếu can ctx/maxz >= base+0.08. (3) meanz chỉ đối chứng dilution. '
      '(4) (2) đậu (1) trượt -> đọc per-cat, KHÔNG tune trong run.')


if __name__ == '__main__':
    main()
