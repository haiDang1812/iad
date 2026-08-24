# eval_fullscale.py
# -----------------------------------------------------------------------------
# FULL-SCALE RUN của chain đã chốt (không hypothesis mới — đây là run ĐO, scale lên):
#   nền = 3:48 + context 2:36, fuse per-pixel MAX-Z (twoscale verdict 2026-08-21:
#   Δmean AUPRO +0.0277 / trần +0.0190, 8/8 cat AUPRO tăng → maxz VÀO NỀN).
#   Mỗi scale: cand overlap-windows -> coreset 50k -> 1-NN -> overlap-averaging.
#   RULE ĐÓNG BĂNG p95 heldout × 1.15 + closing 1 cell. FAIR: bank/z-stats/threshold
#   CHỈ train/good; GT chỉ chấm điểm. KHÔNG per-cat.
#
# Khác quick harness: max_eval=0 (FULL test_public = số paper), max_train=200 (nhiều
#   train hơn cho cand coreset — dùng thêm data train là fair chuẩn).
#
# VARIANTS:
#   base : 3:48 (ablation cho paper, kiểm transfer)
#   maxz : two-scale max-z  -> ứng viên map liên tục (tiff/AUPRO) VÀ map nhị phân
#   gmaxz: guided filter (eps=1e-3 đóng băng từ run guidedup) TRÊN map maxz
#          -> ứng viên nhánh nhị phân (png/SegF1) theo tiền lệ per-metric split
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   1) Transfer: kỳ vọng Δ(maxz-base) AUPRO >= +0.010 giữ ở full-scale; nếu < +0.005
#      -> maxz mong manh, ghi nhận, không tune.
#   2) NHÁNH PNG (SegF1) = gmaxz nếu Δmean F1@rule(gmaxz - maxz) >= +0.005; ngược lại
#      png = maxz luôn (bỏ per-metric split). Trần chỉ đọc tham khảo. AUPRO của gmaxz
#      không dùng để quyết (tiff luôn = maxz).
#   3) Số MEAN maxz (AUPRO) + nhánh png thắng (F1@rule) = số fair chính thức trên
#      test_public -> đưa thẳng vào script submit. KHÔNG tune gì thêm trong run.
#
#   python eval_fullscale.py --data_path ../data --out_dir ./fullscale \
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
from eval_guidedup import boxf, load_gray                                          # noqa: E402
from diag30_thin_premise import aupro05                                            # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

RULE_P, RULE_G = 95.0, 1.15          # ĐÓNG BĂNG. KHÔNG ĐỔI.
SCALES = [(3, 48), (2, 36)]          # fine eff144 + context eff72. CỐ ĐỊNH (twoscale).
GUID_EPS = 1e-3                      # ĐÓNG BĂNG (argmax run guidedup). KHÔNG ĐỔI.
VARIANTS = ['base', 'maxz', 'gmaxz']


def up_grid(g, G, device):
    t = torch.from_numpy(np.ascontiguousarray(g))[None, None].to(device)
    return F.interpolate(t, size=(G, G), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def fuse2(g3, g2u, st):
    (mu3, sd3), (mu2, sd2) = st
    z2 = (g2u - mu2) / sd2
    return {'base': g3, 'maxz': mu3 + sd3 * np.maximum((g3 - mu3) / sd3, z2)}


@torch.no_grad()
def guided1(m_t, gray, r, eps=GUID_EPS):
    I = gray[None, None]; pp = m_t[None, None]
    mI = boxf(I, r); mp = boxf(pp, r)
    varI = boxf(I * I, r) - mI * mI
    cov = boxf(I * pp, r) - mI * mp
    a = cov / (varI + eps)
    b = mp - a * mI
    return (boxf(a, r) * I + boxf(b, r)).squeeze(0).squeeze(0)


def run_cat(bb, cat, args, layers, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    if args.max_train and len(tr) > args.max_train + 3:
        tr_use = tr[:args.max_train]
        va = tr[args.max_train:args.max_train + args.n_val]
    else:
        tr_use, va = tr[:-args.n_val], tr[-args.n_val:]   # dùng hết train, giữ đuôi làm heldout
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
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | train_bank={len(tr_use)} heldout={len(va)}'
      f'{" [TRÙNG BANK]" if overlap_warn else ""}')

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

    r512 = max(1, round(args.aupro_res / G3))

    def all_maps(fused, pil, W, H):
        """dict variant -> (map native GPU, map numpy @aupro_res)."""
        nat = {v: make_map(fused[v], args.canvas, gk, (H, W), device) for v in fused}
        ap = {v: make_map(fused[v], args.canvas, gk, (args.aupro_res, args.aupro_res), device) for v in fused}
        r_nat = max(1, round(min(H, W) / G3))
        nat['gmaxz'] = guided1(nat['maxz'], load_gray(pil, device), r_nat)
        ap['gmaxz'] = guided1(ap['maxz'], load_gray(pil, device, (args.aupro_res, args.aupro_res)), r512)
        return nat, {v: ap[v].cpu().numpy() for v in ap}   # float32: adeval/histc không nhận f16

    # ---- heldout: ngưỡng per-variant theo RULE đóng băng ----
    tr_px = {v: [] for v in VARIANTS}
    for k, path in enumerate(tqdm(va, ncols=70, desc=f'    {cat} heldout thr', leave=False)):
        pil = Image.open(path)
        W, H = pil.size
        fused = fuse2(va_g[0][k], up_grid(va_g[1][k], G3, device), st)
        nat, _ = all_maps(fused, pil, W, H)
        for v in VARIANTS:
            tr_px[v].append(nat[v].cpu().numpy().ravel()[::4])
        del nat
    thr = {v: float(np.percentile(np.concatenate(tr_px[v]), RULE_P)) * RULE_G for v in VARIANTS}
    del tr_px, va_g

    # ---- test: fuse grid -> Hist(trần) + F1@rule + AUPRO@aupro_res ----
    te_var = [fuse2(te_g[0][k], up_grid(te_g[1][k], G3, device), st) for k in range(len(idx))]
    del te_g
    h = {}
    for v in ['base', 'maxz']:
        vmin = min(float(d[v].min()) for d in te_var)
        vmax = max(float(d[v].max()) for d in te_var)
        h[v] = Hist(vmin - 0.05, vmax + 0.05)
    h['gmaxz'] = Hist(h['maxz'].lo - 0.5, h['maxz'].hi + 0.5)   # guided có thể vượt biên grid nhẹ
    mst = {v: np.zeros(3, np.float64) for v in VARIANTS}
    ap_preds = {v: [] for v in VARIANTS}
    ap_gts = []
    for fused, i in zip(tqdm(te_var, ncols=70, desc=f'    {cat} maps', leave=False), idx):
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
        nat, ap = all_maps(fused, pil, W, H)
        g_t = torch.from_numpy(g_nat).to(device) > 0
        r = max(1, round(min(H, W) / G3))
        for v in VARIANTS:
            h[v].add(nat[v].cpu().numpy().reshape(-1), g_nat.reshape(-1))
            ap_preds[v].append(ap[v])
            pred = closing(nat[v] > thr[v], 2 * r + 1)
            mst[v] += ((pred & g_t).sum().item(), (pred & ~g_t).sum().item(), ((~pred) & g_t).sum().item())
            del pred
        del nat, g_t
    del te_var

    out = {}
    for v in VARIANTS:
        tp, fp, fn = mst[v]
        out[v] = {'aupro': aupro05(ap_preds[v], ap_gts, rng), 'f1_max': h[v].f1_max(),
                  'f1': float(2 * tp / (2 * tp + fp + fn + 1e-9))}
        p(f'    [{cat}] {v:6s}: AUPRO0.05={out[v]["aupro"]:.4f}  trần={out[v]["f1_max"]:.4f}  F1@rule={out[v]["f1"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_fullscale: full test_public, chain chốt (two-scale maxz) + nhánh png gmaxz')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL test_public (mặc định run này)')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./fullscale')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fullscale', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} FULL-SCALE: max_train={args.max_train} max_eval={args.max_eval or "FULL"} '
      f'scales={SCALES} bank={args.bank_size}x2 cand={args.cand_size} | RULE p={RULE_P} g={RULE_G} morph=CÓ '
      f'| guided eps={GUID_EPS} (đóng băng)')
    p('  variants: base(3:48) | maxz(two-scale) | gmaxz(guided trên maxz). FAIR, không per-cat, không GT.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN theo variant (FULL test_public) =====')
    mean = {}
    for v in VARIANTS:
        mean[v] = {k: float(np.mean([res[c][v][k] for c in res])) for k in ('aupro', 'f1_max', 'f1')}
        p(f'  {v:6s}: AUPRO0.05={mean[v]["aupro"]:.4f}  trần={mean[v]["f1_max"]:.4f}  F1@rule={mean[v]["f1"]:.4f}')
    dt = mean['maxz']['aupro'] - mean['base']['aupro']
    dpng = mean['gmaxz']['f1'] - mean['maxz']['f1']
    p(f'\n  Transfer Δ(maxz-base) AUPRO = {dt:+.4f}  (kỳ vọng >= +0.010; < +0.005 -> mong manh)')
    p(f'  Nhánh PNG: ΔF1@rule(gmaxz-maxz) = {dpng:+.4f} -> png = {"gmaxz" if dpng >= 0.005 else "maxz"} '
      f'(tiêu chí đóng băng >= +0.005)')
    p('\nĐỌC (pre-registered): tiff = maxz (luôn). png = gmaxz nếu ΔF1@rule >= +0.005, ngược lại maxz. '
      'Số MEAN của cặp thắng = số fair chính thức test_public -> sang script submit. KHÔNG tune trong run.')


if __name__ == '__main__':
    main()
