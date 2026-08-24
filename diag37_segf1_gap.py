# diag37_segf1_gap.py
# -----------------------------------------------------------------------------
# MỔ GAP SegF1 (fabric/wallplugs) — chẩn đoán, KHÔNG phải lever.
#
# Sự kiện (fullscale 2026-08-22, full test_public, chain chốt maxz/gmaxz):
#   - fabric : AUPRO 0.86 nhưng TRẦN map chỉ 0.2135 (SuperADD pub F1 = 0.937!).
#     Quick-30 trần từng là 0.497 → thêm 60 bad + 36 good nghiền nát pooled F1.
#   - wallplugs: trần 0.3137 vs SuperADD 0.792.
#   Oracle threshold đã tính trong trần → KHÔNG phải bệnh threshold. Là bệnh MAP.
#
# Câu hỏi duy nhất: FP hay FN?
#   (A) FP-dominated: ảnh good nổi hotspot ở mọi ngưỡng → bệnh coverage/normal-variation
#       → thuốc phía bank/chuẩn hóa cho nhánh png.
#   (B) FN-dominated: defect nhỏ recall ~0 → bệnh chi tiết/sub-cell ở grid 144
#       → thuốc độ phân giải RIÊNG cho nhánh png (AUPRO không đụng).
#
# Đo trên map gmaxz (nhánh png đã chốt), cùng chain/bank/z-stats fullscale:
#   1) trần full + trần trên đúng subset quick-30 (cùng rng seed → cùng 30/30 ảnh)
#      -> xác nhận hiệu ứng mẫu.
#   2) Ngưỡng oracle t* (argmax F1 full) -> pooled P/R; FP tách theo ảnh good vs bad.
#   3) Recall theo bucket diện tích GT: <2k / 2k-20k / >=20k px; số ảnh bad recall<0.1.
#   4) Top-10 ảnh đóng góp FP nhiều nhất (tên file, label).
#
# ĐỌC (pre-register): FP từ ảnh good >= 50% tổng FP -> kết luận (A). Recall bucket
#   nhỏ < 0.15 VÀ FP-good < 30% -> kết luận (B). Lưng chừng -> bệnh kép, báo cả hai số.
#   KHÔNG đề xuất lever trong run này.
#
#   python diag37_segf1_gap.py --data_path ../data --out_dir ./diag37 --categories fabric wallplugs
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
from eval_native import Hist, make_map, NB                                         # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, fuse2, up_grid, guided1                         # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

SUB_N = 30                                   # tái lập subset quick-30
AREA_BUCKETS = [0, 2000, 20000, 10**12]      # px GT native


def f1_argmax(h):
    """(f1_max, thr*) từ Hist: quét NB ngưỡng bin."""
    cpos = np.cumsum(h.pos[::-1])[::-1].astype(np.float64)
    cneg = np.cumsum(h.neg[::-1])[::-1].astype(np.float64)
    tot = float(h.pos.sum())
    f1 = 2 * cpos / (2 * cpos + cneg + (tot - cpos) + 1e-9)
    b = int(np.argmax(f1))
    return float(f1[b]), float(h.lo + (h.hi - h.lo) * b / NB)


def run_cat(bb, cat, args, layers, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if args.max_train and len(tr) > args.max_train + 3:
        tr_use, va = tr[:args.max_train], tr[args.max_train:args.max_train + args.n_val]
    else:
        tr_use, va = tr[:-args.n_val], tr[-args.n_val:]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    idx = bad + good
    sub = set(bad[:SUB_N] + good[:SUB_N])    # đúng subset quick-30 (cùng rng)
    p(f'  [{cat}] bad={len(bad)} good={len(good)} sub30={len(sub)} | train_bank={len(tr_use)} heldout={len(va)}')

    # ---- banks + grids (y hệt fullscale) ----
    va_g, te_g = {}, {}
    for si, (T, gt) in enumerate(SCALES):
        R = gt * bb.patch
        cand = build_cand_overlap(bb, tr_use, T, R, gt, layers, args.enc_batch, args.cand_size, device).to(device)
        bank = coreset(cand, args.bank_size, device, seed=args.seed)
        del cand; torch.cuda.empty_cache()
        va_g[si] = [overlap_score(bb, Image.open(path), T, R, gt, layers, args.enc_batch, bank, device)
                    for path in tqdm(va, ncols=70, desc=f'    {cat} s{T}:{gt} heldout', leave=False)]
        te_g[si] = [overlap_score(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch, bank, device)
                    for i in tqdm(idx, ncols=70, desc=f'    {cat} s{T}:{gt} test', leave=False)]
        del bank; torch.cuda.empty_cache()
    st = []
    for si in range(len(SCALES)):
        px = np.concatenate([g.ravel() for g in va_g[si]])
        st.append((float(px.mean()), float(px.std()) + 1e-6))
    del va_g

    def gmaxz_nat(k, pil, W, H):
        fused = fuse2(te_g[0][k], up_grid(te_g[1][k], G3, device), st)
        nat = make_map(fused['maxz'], args.canvas, gk, (H, W), device)
        return guided1(nat, load_gray(pil, device), max(1, round(min(H, W) / G3)))

    # ---- pass 1: hist full + hist sub30 ----
    vmin = min(min(float(te_g[0][k].min()), float(te_g[1][k].min())) for k in range(len(idx)))
    vmax = max(max(float(te_g[0][k].max()), float(te_g[1][k].max())) for k in range(len(idx)))
    h_full, h_sub = Hist(vmin - 0.55, vmax + 0.55), Hist(vmin - 0.55, vmax + 0.55)
    for k, i in enumerate(tqdm(idx, ncols=70, desc=f'    {cat} pass1', leave=False)):
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), np.uint8)
        else:
            g_nat = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)
        m = gmaxz_nat(k, pil, W, H).cpu().numpy().reshape(-1)
        h_full.add(m, g_nat.reshape(-1))
        if i in sub:
            h_sub.add(m, g_nat.reshape(-1))
        del m
    f1_full, t_star = f1_argmax(h_full)
    f1_sub, _ = f1_argmax(h_sub)
    p(f'    [{cat}] trần FULL={f1_full:.4f} (t*={t_star:.3f}) | trần SUB30={f1_sub:.4f} '
      f'(quick-30 cũ ~{0.497 if cat == "fabric" else 0.31:.2f})')

    # ---- pass 2: per-image tại t* (không closing — mổ map thuần) ----
    rows = []
    for k, i in enumerate(tqdm(idx, ncols=70, desc=f'    {cat} pass2', leave=False)):
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        if ds.labels[i] == 0:
            g_t = torch.zeros((H, W), dtype=torch.bool, device=device)
            area = 0
        else:
            g_nat = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)
            area = int(g_nat.sum())
            g_t = torch.from_numpy(g_nat).to(device) > 0
        pred = gmaxz_nat(k, pil, W, H) > t_star
        tp = int((pred & g_t).sum()); fp = int((pred & ~g_t).sum()); fn = int(((~pred) & g_t).sum())
        rows.append((os.path.basename(ds.img_paths[i]), int(ds.labels[i]), area, tp, fp, fn))
        del pred, g_t
    del te_g

    TP = sum(r[3] for r in rows); FP = sum(r[4] for r in rows); FN = sum(r[5] for r in rows)
    fp_good = sum(r[4] for r in rows if r[1] == 0)
    p(f'    [{cat}] tại t*: P={TP / (TP + FP + 1e-9):.4f} R={TP / (TP + FN + 1e-9):.4f} | '
      f'FP tổng={FP:,} — từ ảnh GOOD: {100.0 * fp_good / (FP + 1e-9):.1f}%')
    for lo, hi in zip(AREA_BUCKETS[:-1], AREA_BUCKETS[1:]):
        rs = [r for r in rows if r[1] == 1 and lo <= r[2] < hi]
        if rs:
            btp = sum(r[3] for r in rs); bfn = sum(r[5] for r in rs)
            miss = sum(1 for r in rs if r[3] / (r[3] + r[5] + 1e-9) < 0.1)
            p(f'    [{cat}] GT area [{lo:>6},{hi if hi < 10**12 else "inf":>6}): n={len(rs):3d} '
              f'recall={btp / (btp + bfn + 1e-9):.4f}  ảnh recall<0.1: {miss}')
    top = sorted(rows, key=lambda r: -r[4])[:10]
    p(f'    [{cat}] top-10 FP: ' + ', '.join(f'{r[0]}({"bad" if r[1] else "good"},{r[4] // 1000}k)' for r in top))


def main():
    ap = argparse.ArgumentParser('diag37: mổ trần SegF1 fabric/wallplugs — FP hay FN?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=['fabric', 'wallplugs'])
    ap.add_argument('--out_dir', type=str, default='./diag37')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag37', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} DIAG (không lever): map gmaxz nhánh png, chain fullscale nguyên vẹn. '
      f'Câu hỏi: trần thấp do FP (ảnh good) hay FN (defect nhỏ)?')
    for cat in args.categories:
        run_cat(bb, cat, args, layers, gk, device, p)
    p('\nĐỌC (pre-registered): FP-good>=50% -> bệnh (A) coverage/normal-variation. '
      'Recall bucket <2k px <0.15 VÀ FP-good<30% -> bệnh (B) chi tiết/sub-cell. Lưng chừng -> kép.')


if __name__ == '__main__':
    main()
