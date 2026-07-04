# diag21_pin_premise.py
# -----------------------------------------------------------------------------
# PREMISE TEST cho PIN (Per-Image self-Normalization) — thay TTNA (bank-adapt đã bị diag20 bác bỏ).
#
# CHẨN ĐOÁN MỚI (diag20 dùng shift ĐỒNG NHẤT nên KHÔNG chạm tới cơ chế thật):
#   AUPRO0.05 tính FPR *GLOBAL* qua toàn bộ pixel/ảnh. test_private_mixed shift KHÔNG đồng nhất
#   per-image (mỗi ảnh một điều kiện sáng). => ảnh bị shift MẠNH có điểm-normal dâng cao,
#   NGỐN HẾT budget FPR<=0.05 -> defect ở ảnh shift-nhẹ bị đè xuống -> AUPRO sụp.
#   Đây là "CROSS-IMAGE SCORE DISPARITY" — bank-adapt không đụng tới, diag20 (đồng nhất) đo không ra.
#
# METHOD (một họ cơ chế -> hai metric), KHÔNG nhãn, tham số-tự-do:
#   PIN = chuẩn hoá MAP KHOẢNG-CÁCH theo thống-kê-normal CỦA CHÍNH ẢNH (robust median/MAD):
#     z_i(x) = (d_i(x) - median(d_i)) / (1.4826*MAD(d_i))
#   -> trừ offset shift per-image (median) + chuẩn scale per-image (MAD)
#   -> gỡ cross-image disparity  -> mọi ảnh về chung baseline -> budget FPR chia CÔNG BẰNG
#      -> cứu AUPRO0.05 (ranking GLOBAL).   (defect thưa <1% -> median/MAD ~ mức normal, không bị kéo)
#   -> map về thang chung -> self-cal percentile threshold -> cứu SegF1@fixed.
#   Cùng self-normalization ở 2 mức: per-image (AUPRO) + per-set (SegF1).
#
# Đối chứng (distance-only, không head few-shot):
#   BASE        : KHÔNG shift, global-norm,  thr = val mean+3σ.
#   SHIFT       : shift DỊ-NHẤT per-image, global-norm (= pipeline hiện tại), val-thr -> tái hiện & KHUẾCH ĐẠI tụt.
#   SHIFT+SC    : maps SHIFT, self-cal percentile thr  -> chỉ đổi ngưỡng (SegF1); AUPRO = SHIFT.
#   SHIFT+PIN   : per-image norm maps SHIFT  -> AUPRO mới; + self-cal thr -> SegF1.
#   [SAFETY]    : PIN áp lên BASE (KHÔNG shift) -> phải KHÔNG hại AUPRO (method vô hại khi không có domain-shift).
#   [DISPARITY] : std_i(median map_i) — bằng chứng trực tiếp: SHIFT thổi phồng, PIN kéo về gần BASE.
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag21_pin_premise.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shift_lo 0.3 --shift_hi 1.2 --out_dir ./diag21
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageEnhance

from dataset import MVTecAD2Dataset
from utils import ader_evaluator, get_gaussian_kernel, get_logger
from backbones_ext import load_backbone

warnings.filterwarnings("ignore")

VALID = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ===== core: y hệt diag20 / train_softpro ====================================
def photometric_shift(pil, s):
    """Giả lập dark-field/back-light: giảm sáng + tăng contrast + gamma + bạc màu. s=0 -> no-op. Deterministic."""
    if s <= 0:
        return pil
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)
    return Image.fromarray((arr * 255.0).astype(np.uint8), 'RGB')


def to_tensor(pil, R):
    pil = pil.convert('RGB').resize((R, R), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.).permute(2, 0, 1)
    for c in range(3):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x


def tile_pils(pil, T):
    w, h = pil.size
    return [pil.crop((round(j * w / T), round(i * h / T), round((j + 1) * w / T), round((i + 1) * h / T)))
            for i in range(T) for j in range(T)]


def subsample(flat, n, seed=0):
    if flat.shape[0] <= n:
        return flat
    g = torch.Generator().manual_seed(seed)
    return flat[torch.randperm(flat.shape[0], generator=g)[:n]]


@torch.no_grad()
def img_featmap(bb, pil, T, R, gt, layers, enc_batch):
    tiles = tile_pils(pil, T)
    fl = []
    for s in range(0, len(tiles), enc_batch):
        b = torch.stack([to_tensor(t, R) for t in tiles[s:s + enc_batch]])
        fl.append(bb.extract(b, layers))
    f = torch.cat(fl, 0)
    C = f.shape[-1]
    grid = torch.zeros(T * gt, T * gt, C, device=f.device)
    for k in range(T * T):
        i, j = k // T, k % T
        grid[i * gt:(i + 1) * gt, j * gt:(j + 1) * gt] = f[k, :gt * gt].reshape(gt, gt, C)
    return grid


@torch.no_grad()
def nn_map_flat(qflat, bank, device, chunk=4096):
    out = torch.empty(qflat.shape[0], device=device)
    for s in range(0, qflat.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(qflat[s:s + chunk], bank).min(1)[0]
    return out


def gt_grid(gpath, label_, G):
    if label_ == 0 or not (isinstance(gpath, str) and os.path.exists(gpath)):
        return np.zeros((G, G), dtype=np.uint8)
    gi = Image.open(gpath).convert('L').resize((G, G), Image.NEAREST)
    return (np.asarray(gi) > 127).astype(np.uint8)


def upmap(arr2d, size, gk, device):
    t = torch.tensor(arr2d, device=device)[None, None].float()
    t = F.interpolate(t, size=size, mode='bilinear', align_corners=False)
    return gk(t)[0, 0].cpu().numpy()


def region_metrics(maps, gts, gk, device, resize=256):
    pr = np.stack([upmap(m, resize, gk, device) for m in maps], 0)
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST))
                   for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * 0.01))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    r = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return r[7]   # AUPRO0.05


def segf1_fixed(maps, gts, thr, gk, device, resize=256):
    """SegF1 tại NGƯỠNG CỐ ĐỊNH thr (mirror server, KHÁC oracle P-F1max)."""
    ss, yy = [], []
    for m, g in zip(maps, gts):
        ss.append(upmap(m, resize, gk, device).reshape(-1))
        yy.append((np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) > 0)
                  .reshape(-1).astype(np.uint8))
    s = np.concatenate(ss)
    y = np.concatenate(yy)
    if y.sum() == 0:
        return 0.0
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / (prec + rec + 1e-12)


def norm_maps(grids_d, lo, hi):
    """global-norm: MỘT phép affine dùng chung cho mọi ảnh (monotone -> KHÔNG đổi AUPRO)."""
    return [(d - lo) / (hi - lo + 1e-8) for d in grids_d]


def pin_maps(grids_d, mode='mad', botq=0.8, eps=1e-6):
    """PER-IMAGE self-normalization: mỗi ảnh một affine theo thống-kê-normal của CHÍNH ảnh
       (robust median/MAD hoặc bottom-q mean/std). ĐỔI ranking global -> đụng tới AUPRO."""
    out = []
    for d in grids_d:
        flat = d.reshape(-1)
        if mode == 'botq':
            k = max(1, int(len(flat) * botq))
            core = np.sort(flat)[:k]
            loc, scale = float(core.mean()), float(core.std()) + eps
        else:  # mad (robust, tham số-tự-do, chịu được tới ~50% outlier)
            loc = float(np.median(flat))
            scale = float(np.median(np.abs(flat - loc))) * 1.4826 + eps
        out.append((d - loc) / scale)
    return out


def disparity(maps):
    """Cross-image score disparity = độ lệch chuẩn của median-per-ảnh. Cao = ảnh lệch nhau (shift dị nhất)."""
    return float(np.std([float(np.median(m)) for m in maps]))


def main():
    ap = argparse.ArgumentParser('diag21 — PIN premise (heterogeneous shift; per-image norm recovers both metrics)')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--shift_lo', type=float, default=0.3, help='cận DƯỚI cường độ shift per-image (dị nhất)')
    ap.add_argument('--shift_hi', type=float, default=1.2, help='cận TRÊN cường độ shift per-image (dị nhất)')
    ap.add_argument('--pin_mode', type=str, default='mad', choices=['mad', 'botq'])
    ap.add_argument('--pin_botq', type=float, default=0.8, help='tỉ lệ bottom-score/ảnh làm normal-core (mode botq)')
    ap.add_argument('--selfcal_pct', type=float, default=98.0, help='percentile self-cal threshold (RoBiS-style)')
    ap.add_argument('--thr_sigma', type=float, default=3.0, help='ngưỡng fixed = mean+kσ trên val/good')
    ap.add_argument('--max_eval', type=int, default=80, help='giới hạn ảnh eval/category cho nhanh')
    ap.add_argument('--max_val', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag21')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag21', args.out_dir).info
    torch.manual_seed(args.seed)

    bb = load_backbone(args.model, device)
    R = args.grid_tile * bb.patch
    if args.layers_fixed or not bb.n_layers:
        layers = [ly for ly in args.layers if ly < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(ly / 12 * bb.n_layers))) for ly in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles
    gt = args.grid_tile
    rng = np.random.default_rng(args.seed)
    shift_rng = np.random.default_rng(args.seed + 12345)   # rng RIÊNG cho cường độ shift per-image

    p('=' * 112)
    p(f'DIAG21 PIN premise | model={args.model} eff_grid={T*gt} layers={layers} | '
      f'shift~U[{args.shift_lo},{args.shift_hi}] per-image | pin={args.pin_mode} selfcal_pct={args.selfcal_pct}')
    p('BASE -> SHIFT(het, global-norm) -> SHIFT+SC(self-cal thr) -> SHIFT+PIN(per-image norm + self-cal) | +SAFETY +DISPARITY')
    p('=' * 112)

    agg = {k: [] for k in ['base_au', 'base_f1', 'sh_au', 'sh_f1', 'sc_f1',
                           'pin_au', 'pin_f1', 'safe_au', 'disp_base', 'disp_sh', 'disp_pin']}

    for cat in args.categories:
        # ---- bank train (y hệt diag20) ----
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        acc = []
        keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            buf = []
            for pth in tr:
                buf.extend(tile_pils(Image.open(pth), T))
                while len(buf) >= args.enc_batch:
                    b = torch.stack([to_tensor(t, R) for t in buf[:args.enc_batch]])
                    buf = buf[args.enc_batch:]
                    f = bb.extract(b, layers)
                    acc.append(subsample(f.reshape(-1, f.shape[-1]), args.enc_batch * keep).cpu())
            if buf:
                f = bb.extract(torch.stack([to_tensor(t, R) for t in buf]), layers)
                acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
        bank = subsample(torch.cat(acc, 0), args.bank_size).to(device)
        Cdim = bank.shape[-1]

        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
        rng.shuffle(bad)
        rng.shuffle(good)
        eval_idx = bad[:args.max_eval] + good[:args.max_eval // 2]

        # ---- eval: BASE (no-shift) + SHIFT DỊ-NHẤT (mỗi ảnh 1 cường độ) ----
        base_grids, sh_grids, gts, s_list = [], [], [], []
        for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval'):
            pil = Image.open(ds.img_paths[i])
            s_i = float(shift_rng.uniform(args.shift_lo, args.shift_hi))   # <-- KEY: shift per-image
            s_list.append(s_i)
            gb = img_featmap(bb, pil, T, R, gt, layers, args.enc_batch)
            gs = img_featmap(bb, photometric_shift(pil, s_i), T, R, gt, layers, args.enc_batch)
            base_grids.append(gb.cpu().numpy())
            sh_grids.append(gs.cpu().numpy())
            gts.append(gt_grid(ds.gt_paths[i], ds.labels[i], gb.shape[0]))
        G = base_grids[0].shape[0]

        val_imgs = sorted(glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', '*.png')) +
                          glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', '*.jpg')))[:args.max_val]
        val_grids = [img_featmap(bb, Image.open(v), T, R, gt, layers, args.enc_batch).cpu().numpy() for v in val_imgs]

        def nn_all(grids):
            out = []
            for g in grids:
                d = nn_map_flat(torch.tensor(g.reshape(-1, Cdim), device=device), bank, device)
                out.append(d.reshape(G, G).cpu().numpy())
            return out

        d_base = nn_all(base_grids)
        d_sh = nn_all(sh_grids)
        d_val = nn_all(val_grids) if val_grids else []

        # ---------- BASE (no shift, global-norm) ----------
        lo_b, hi_b = np.percentile(np.stack(d_base), 1), np.percentile(np.stack(d_base), 99)
        m_base = norm_maps(d_base, lo_b, hi_b)
        au_base = region_metrics(m_base, gts, gk, device)
        if d_val:
            vs = np.concatenate([upmap((dv - lo_b) / (hi_b - lo_b + 1e-8), 256, gk, device).reshape(-1) for dv in d_val])
            thr_base = float(vs.mean() + args.thr_sigma * vs.std())
        else:
            thr_base = float(np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in m_base]).mean())
        f1_base = segf1_fixed(m_base, gts, thr_base, gk, device)

        # ---------- SHIFT (het, global-norm, val-thr = calib SAI) ----------
        lo_s, hi_s = np.percentile(np.stack(d_sh), 1), np.percentile(np.stack(d_sh), 99)
        m_sh = norm_maps(d_sh, lo_s, hi_s)
        au_sh = region_metrics(m_sh, gts, gk, device)
        if d_val:
            vs_s = np.concatenate([upmap((dv - lo_s) / (hi_s - lo_s + 1e-8), 256, gk, device).reshape(-1) for dv in d_val])
            thr_valonshift = float(vs_s.mean() + args.thr_sigma * vs_s.std())
        else:
            thr_valonshift = thr_base
        f1_sh = segf1_fixed(m_sh, gts, thr_valonshift, gk, device)

        # ---------- SHIFT+SC (self-cal percentile thr; AUPRO = SHIFT) ----------
        pooled_sh = np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in m_sh])
        thr_sc = float(np.percentile(pooled_sh, args.selfcal_pct))
        f1_sc = segf1_fixed(m_sh, gts, thr_sc, gk, device)

        # ---------- SHIFT+PIN (per-image norm -> AUPRO mới; + self-cal -> SegF1) ----------
        m_pin = pin_maps(d_sh, mode=args.pin_mode, botq=args.pin_botq)
        au_pin = region_metrics(m_pin, gts, gk, device)
        pooled_pin = np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in m_pin])
        thr_pin = float(np.percentile(pooled_pin, args.selfcal_pct))
        f1_pin = segf1_fixed(m_pin, gts, thr_pin, gk, device)

        # ---------- SAFETY: PIN áp lên BASE (no-shift) phải KHÔNG hại AUPRO ----------
        m_safe = pin_maps(d_base, mode=args.pin_mode, botq=args.pin_botq)
        au_safe = region_metrics(m_safe, gts, gk, device)

        # ---------- DISPARITY (bằng chứng cơ chế) ----------
        disp_base = disparity(m_base)
        disp_sh = disparity(m_sh)
        disp_pin = disparity(m_pin)

        for kk, vv in [('base_au', au_base), ('base_f1', f1_base), ('sh_au', au_sh), ('sh_f1', f1_sh),
                       ('sc_f1', f1_sc), ('pin_au', au_pin), ('pin_f1', f1_pin), ('safe_au', au_safe),
                       ('disp_base', disp_base), ('disp_sh', disp_sh), ('disp_pin', disp_pin)]:
            agg[kk].append(vv)

        p(f'  [{cat:<11}] AUPRO05  base={au_base:.3f} shift={au_sh:.3f} PIN={au_pin:.3f} ({au_pin-au_sh:+.3f}) '
          f'safe={au_safe:.3f} | SegF1 base={f1_base:.3f} shift={f1_sh:.3f} +SC={f1_sc:.3f} +PIN={f1_pin:.3f} '
          f'| disp b/s/p={disp_base:.3f}/{disp_sh:.3f}/{disp_pin:.3f} | s~[{min(s_list):.2f},{max(s_list):.2f}]')

    p('\n' + '=' * 112)
    M = {k: float(np.mean(v)) for k, v in agg.items()}
    p('{:<14}{:>10}{:>10}{:>10}{:>10}'.format('MEAN', 'AUPRO05', 'SegF1', 'dAU', 'dF1'))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10}{:>10}'.format('BASE(no-shift)', M['base_au'], M['base_f1'], '-', '-'))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}'.format('SHIFT(het)', M['sh_au'], M['sh_f1'],
                                                          M['sh_au'] - M['base_au'], M['sh_f1'] - M['base_f1']))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10}{:>10.4f}'.format('SHIFT+SC', M['sh_au'], M['sc_f1'], '(=shift)',
                                                       M['sc_f1'] - M['sh_f1']))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}'.format('SHIFT+PIN', M['pin_au'], M['pin_f1'],
                                                          M['pin_au'] - M['sh_au'], M['pin_f1'] - M['sh_f1']))
    p(f'\nDISPARITY (std median/ảnh)  BASE={M["disp_base"]:.4f}  SHIFT={M["disp_sh"]:.4f}  '
      f'PIN={M["disp_pin"]:.4f}   (SHIFT>BASE = shift dị nhất; PIN->gần BASE = gỡ disparity)')
    p(f'SAFETY  PIN@BASE AUPRO05={M["safe_au"]:.4f} vs BASE={M["base_au"]:.4f} '
      f'(drop={M["base_au"]-M["safe_au"]:+.4f}; ~0 = PIN vô hại khi KHÔNG shift)')

    drop_au = M['base_au'] - M['sh_au']
    drop_f1 = M['base_f1'] - M['sh_f1']
    rec_au = M['pin_au'] - M['sh_au']
    rec_f1 = max(M['sc_f1'], M['pin_f1']) - M['sh_f1']
    safe_drop = M['base_au'] - M['safe_au']
    disp_cut = M['disp_sh'] - M['disp_pin']

    v_fail = 'PASS' if (drop_au > 0.02 and drop_f1 > 0.02) else 'FAIL'
    v_au = 'PASS' if rec_au > 0.01 else 'FAIL'
    v_f1 = 'PASS' if rec_f1 > 0.01 else 'FAIL'
    v_safe = 'PASS' if safe_drop < 0.01 else 'FAIL'
    v_mech = 'PASS' if disp_cut > 0 else 'FAIL'
    p('=' * 112)
    p(f'PREMISE FAILURE  (shift-het tái hiện tụt private)   : {v_fail}  (dAU={-drop_au:+.3f} dF1={-drop_f1:+.3f})')
    p(f'PREMISE METHOD-AUPRO (per-image norm cứu ranking)   : {v_au}  ({rec_au:+.3f})')
    p(f'PREMISE METHOD-SegF1 (self-cal/PIN cứu ngưỡng)      : {v_f1}  ({rec_f1:+.3f})')
    p(f'PREMISE SAFETY  (PIN vô hại khi KHÔNG shift)        : {v_safe}  (drop={safe_drop:+.3f})')
    p(f'PREMISE MECHANISM (PIN gỡ cross-image disparity)    : {v_mech}  (Δdisp={disp_cut:+.4f})')
    p('=> BUILD PIN nếu FAILURE + METHOD-AUPRO + SAFETY PASS (một lever per-image cứu AUPRO, self-cal cứu SegF1).')
    p('   Nếu METHOD-AUPRO FAIL: disparity KHÔNG phải gốc AUPRO-drop -> shift làm defect==rare-normal (few-shot gánh).')

    with open(os.path.join(args.out_dir, 'diag21.csv'), 'w') as fcsv:
        fcsv.write('metric,base,shift,shift_sc,shift_pin\n')
        fcsv.write(f'AUPRO05,{M["base_au"]:.4f},{M["sh_au"]:.4f},{M["sh_au"]:.4f},{M["pin_au"]:.4f}\n')
        fcsv.write(f'SegF1,{M["base_f1"]:.4f},{M["sh_f1"]:.4f},{M["sc_f1"]:.4f},{M["pin_f1"]:.4f}\n')
        fcsv.write(f'disparity,{M["disp_base"]:.4f},{M["disp_sh"]:.4f},,{M["disp_pin"]:.4f}\n')
        fcsv.write(f'safety_pin_at_base_au,{M["safe_au"]:.4f},,,\n')
    p(f'Đã lưu: {os.path.join(args.out_dir, "diag21.csv")}')


if __name__ == '__main__':
    main()
