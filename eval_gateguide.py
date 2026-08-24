# eval_gateguide.py
# -----------------------------------------------------------------------------
# NOVELTY v2: SCORE-GATED GUIDED REFINEMENT (sửa đúng chỗ guided thuần chết).
#
# Kết quả eval_guidedup (pre-registered): guided KHÔNG vào nền — trần +0.016 (7/8 cat TĂNG,
#   cơ chế biên THẬT) nhưng AUPRO −0.021 vì halo: guided bôi score theo cạnh ảnh ở VÙNG NỀN
#   -> FPR vùng thấp tăng (can −.073, wallplugs −.039, fabric −.037).
#
# Fix = gate theo score: chỉ cho guided sửa biên nơi map đã nóng, nền giữ bilinear nguyên vẹn:
#     gated = m + gate * (guided_eps0.001(m) - m),  gate = clamp((m - t0)/(t1 - t0), 0, 1)
#   t0 = percentile-95 pixel heldout train/good (map base), t1 = t0 * 1.15 = CHÍNH threshold
#   của rule đóng băng -> KHÔNG thêm hyperparameter mới, fair, global, không per-cat.
#   eps đóng băng 0.001 (argmax trần guidedup, đúng luật freeze đã khai trước).
#   Kỳ vọng: vùng dưới t0 không đổi -> AUPRO ~ base; vùng defect lấy biên guided -> giữ Δtrần.
#
# NỀN = config thắng overlapmap (3:48, bank coreset 50k/cand200k cửa sổ chồng lấn, 1-NN,
#   overlap-averaging, RULE p95 g1.15 morph). Encode+score 1 lần/ảnh; variants chỉ khác upsample:
#     base | guided (=g0.001 guidedup, tái lập trong-run) | gated.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   - gated vs base: Δtrần >= +0.015 VÀ ΔAUPRO >= -0.003 -> GATED VÀO NỀN (1 map cho cả 2 metric).
#   - gated fail -> fallback ĐÃ KHAI TRƯỚC: per-metric split (AUPRO = map base 0.6469,
#     SegF1 = map guided: trần 0.4080 / F1@rule 0.3505) — tiền lệ per-metric head trong pipeline.
#   - gated làm AUPRO sập như guided thuần -> gate không chặn được halo -> đọc per-cat tìm cat gãy.
#
#   python eval_gateguide.py --data_path ../data --out_dir ./gateguide --max_eval 30 \
#       --categories vial wallplugs sheet_metal fabric rice walnuts can fruit_jelly
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
from infer_submit_mvtec_ad2 import IMG_EXT                                         # noqa: E402
from eval_bankmap import coreset                                                   # noqa: E402
from eval_overlapmap import build_cand_overlap, overlap_score                      # noqa: E402
from eval_guidedup import guided_up, load_gray                                     # noqa: E402
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from diag30_thin_premise import aupro05                                            # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

RULE_P, RULE_G = 95.0, 1.15          # ĐÓNG BĂNG. KHÔNG ĐỔI.
GKEY = 'g0.001'                      # eps đóng băng theo argmax trần guidedup.
VARIANTS = ['base', 'guided', 'gated']


def gated_map(m, mg, t0, t1):
    gate = ((m - t0) / max(t1 - t0, 1e-9)).clamp(0, 1)
    return m + gate * (mg - m)


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
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
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={G} heldout={len(va)}{" [TRÙNG BANK]" if overlap_warn else ""}')

    cand = build_cand_overlap(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.cand_size, device).to(device)
    bank = coreset(cand, args.bank_size, device, seed=args.seed)
    del cand; torch.cuda.empty_cache()
    p(f'    [{cat}] bank(coreset,overlap)={bank.shape[0]} tu cand={args.cand_size}')

    r512 = max(1, round(args.aupro_res / G))

    # ---- heldout pass 1: score + map base -> t0/t1 (nguyên liệu gate, từ rule) ----
    va_s = []
    px_base = []
    for path in tqdm(va, ncols=70, desc=f'    {cat} heldout1', leave=False):
        pil = Image.open(path)
        W, H = pil.size
        s = overlap_score(bb, pil, T, R, gt_, layers, args.enc_batch, bank, device)
        va_s.append((s, path))
        px_base.append(make_map(s, args.canvas, gk, (H, W), device).cpu().numpy().ravel()[::4])
    t0 = float(np.percentile(np.concatenate(px_base), RULE_P))
    t1 = t0 * RULE_G
    p(f'    [{cat}] gate: t0={t0:.4f} t1={t1:.4f} (p95 heldout x {RULE_G})')

    def all_maps(s, pil, W, H):
        m = make_map(s, args.canvas, gk, (H, W), device)
        r_nat = max(1, round(min(H, W) / G))
        mg = guided_up(m, load_gray(pil, device), r_nat)[GKEY]
        nat = {'base': m, 'guided': mg, 'gated': gated_map(m, mg, t0, t1)}
        m5 = make_map(s, args.canvas, gk, (args.aupro_res, args.aupro_res), device)
        mg5 = guided_up(m5, load_gray(pil, device, (args.aupro_res, args.aupro_res)), r512)[GKEY]
        ap = {'base': m5, 'guided': mg5, 'gated': gated_map(m5, mg5, t0, t1)}
        return nat, {v: ap[v].cpu().numpy() for v in ap}

    # ---- heldout pass 2: ngưỡng per-variant theo RULE (từ score đã lưu) ----
    tr_px = {v: [] for v in VARIANTS}
    for s, path in va_s:
        pil = Image.open(path)
        W, H = pil.size
        nat, _ = all_maps(s, pil, W, H)
        for v in VARIANTS:
            tr_px[v].append(nat[v].cpu().numpy().ravel()[::4])
        del nat
    thr = {v: float(np.percentile(np.concatenate(tr_px[v]), RULE_P)) * RULE_G for v in VARIANTS}
    del tr_px, va_s, px_base

    # ---- test ----
    s_grids = []
    for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
        s_grids.append(overlap_score(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch, bank, device))
    del bank; torch.cuda.empty_cache()

    gmin = min(float(s.min()) for s in s_grids)
    gmax = max(float(s.max()) for s in s_grids)
    h = {v: Hist(gmin - 0.05, gmax + 0.05) for v in VARIANTS}
    mst = {v: np.zeros(3, np.float64) for v in VARIANTS}
    ap_preds = {v: [] for v in VARIANTS}
    ap_gts = []
    for s, i in zip(tqdm(s_grids, ncols=70, desc=f'    {cat} maps', leave=False), idx):
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
        nat, ap = all_maps(s, pil, W, H)
        g_t = torch.from_numpy(g_nat).to(device) > 0
        r = max(1, round(min(H, W) / G))
        for v in VARIANTS:
            h[v].add(nat[v].cpu().numpy().reshape(-1), g_nat.reshape(-1))
            ap_preds[v].append(ap[v])
            pred = closing(nat[v] > thr[v], 2 * r + 1)
            mst[v] += ((pred & g_t).sum().item(), (pred & ~g_t).sum().item(), ((~pred) & g_t).sum().item())
        del nat, g_t
    del s_grids

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = {'aupro': aupro05(ap_preds[v], ap_gts, rng), 'f1_max': h[v].f1_max(),
                  'f1': float(2 * tp / (2 * tp + fp + fn + 1e-9))}
        p(f'    [{cat}] {v:6s}: AUPRO0.05={out[v]["aupro"]:.4f}  trần={out[v]["f1_max"]:.4f}  F1@rule={out[v]["f1"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_gateguide: base vs guided vs score-gated guided (nền ov+coreset, rule đóng băng)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
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
    ap.add_argument('--out_dir', type=str, default='./gateguide')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('gateguide', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} bank={args.bank_size}(coreset,overlap) '
      f'cand={args.cand_size} heldout={args.n_val} | RULE p={RULE_P} g={RULE_G} morph=CÓ | eps={GKEY} '
      f'gate=[t0,p95heldout -> t1=t0x{RULE_G}]')
    p('  variants: base | guided (=g0.001) | gated (guided chỉ nơi map nóng, nền giữ base). FAIR, không per-cat.')

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
    d = {k: mean['gated'][k] - mean['base'][k] for k in mean['base']}
    p(f'\n  Δ GATED (gated - base): AUPRO{d["aupro"]:+.4f}  trần{d["f1_max"]:+.4f}  F1{d["f1"]:+.4f}')
    p('\nĐỌC (pre-registered): Δtrần>=+0.015 VÀ ΔAUPRO>=-0.003 -> GATED VÀO NỀN (1 map cho cả 2 metric). '
      'Fail -> fallback per-metric split (AUPRO=base, SegF1=guided) đã khai trước từ guidedup.')


if __name__ == '__main__':
    main()
