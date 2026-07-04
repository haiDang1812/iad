# diag20_ttna_premise.py
# -----------------------------------------------------------------------------
# PREMISE TEST cho TTNA (Test-Time Normal Adaptation) — novelty nhắm CÚ TỤT PRIVATE.
#
# FAILURE cần chứng minh (chưa diag nào của repo chạm tới — tất cả đều test_public KHÔNG shift):
#   Dưới light-shift (đặc trưng test_private của AD2), normal trôi khỏi manifold train:
#     -> distance normal tăng  -> rare-normal khuếch đại  -> AUPRO0.05 tụt   (nghẽn A)
#     -> phân bố điểm normal dâng -> ngưỡng mean+3σ (calib trên val sạch) quá thấp
#        -> precision sụp        -> SegF1 tụt                                (nghẽn B)
#   => MỘT gốc (normal drift) hại CẢ HAI metric.
#
# METHOD (một cơ chế -> hai metric), KHÔNG nhãn, contamination-robust:
#   1) score s0 bằng bank train.
#   2) chọn confidently-normal = bottom-q patch/ẢNH theo s0 (defect thưa -> bottom-q gần sạch).
#   3) bank' = bank ∪ subsample(confidently-normal test)  -> recompute distance
#        -> normal-shift có hàng xóm gần (distance tụt); defect vẫn xa mọi normal (vẫn cao)
#        -> ĐỔI RANKING -> nâng AUPRO0.05.
#   4) ngưỡng self-cal = mean+3σ trên normal-core test (thay val sạch) -> nâng SegF1@fixed.
#
# Thiết kế đối chứng (distance-only, cô lập cơ chế; không head few-shot):
#   BASE           : không shift, bank train, thr = val mean+3σ.
#   SHIFT          : shift eval, bank train, thr = val mean+3σ (calib SAI) -> tái hiện tụt private.
#   SHIFT+THR      : shift eval, bank train, thr = self-cal percentile (RoBiS-style) -> chỉ cứu SegF1.
#   SHIFT+TTNA     : shift eval, bank', thr = self-cal -> cứu CẢ AUPRO0.05 (bank) LẪN SegF1 (thr).
# In kèm contamination% của pool adapt (dùng GT chỉ để kiểm an toàn, KHÔNG dùng để chọn).
#
# Chạy:
#   HF_HUB_OFFLINE=1 python diag20_ttna_premise.py --data_path ../data --model v3_large \
#     --tiles 2 --grid_tile 28 --shift 1.0 --bottom_q 0.5 --out_dir ./diag20
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


# ===== core: giống train_softpro.py / diag16 =================================
def photometric_shift(pil, s):
    """Giả lập dark-field/back-light của test_private: giảm sáng + tăng contrast + gamma + lệch màu nhẹ.
    s=0 -> no-op. s=1 -> shift mạnh. Deterministic (tái lập được)."""
    if s <= 0:
        return pil
    pil = pil.convert('RGB')
    pil = ImageEnhance.Brightness(pil).enhance(1.0 - 0.45 * s)   # tối đi
    pil = ImageEnhance.Contrast(pil).enhance(1.0 + 0.5 * s)      # gắt hơn
    pil = ImageEnhance.Color(pil).enhance(1.0 - 0.3 * s)         # bạc màu
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.power(np.clip(arr, 0, 1), 1.0 + 0.6 * s)           # gamma > 1 -> tối vùng mid
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
    gt = np.stack([np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) for g in gts], 0).astype(np.uint8)
    sp = np.array([np.sort(m.reshape(-1))[::-1][:max(1, int(m.size * 0.01))].mean() for m in pr])
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gt])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    r = ader_evaluator(pr, sp, gt, gt_sp, use_metrics=METRIC_NAMES)
    return r[7], r[5]   # AUPRO0.05, P-F1max (oracle upper-bound)


def segf1_fixed(maps, gts, thr, gk, device, resize=256, max_neg=4000000, seed=0):
    """SegF1 tại NGƯỠNG CỐ ĐỊNH thr (mirror server, KHÁC oracle P-F1max)."""
    ss, yy = [], []
    for m, g in zip(maps, gts):
        ss.append(upmap(m, resize, gk, device).reshape(-1))
        yy.append((np.asarray(Image.fromarray(g).resize((resize, resize), Image.NEAREST)) > 0).reshape(-1).astype(np.uint8))
    s = np.concatenate(ss); y = np.concatenate(yy)
    if y.sum() == 0:
        return 0.0
    pred = s >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return 2 * prec * rec / (prec + rec + 1e-12)


def norm_maps(grids_d, lo, hi):
    return [(d - lo) / (hi - lo + 1e-8) for d in grids_d]


def main():
    ap = argparse.ArgumentParser('diag20 — TTNA premise (shift reproduces private drop; adapt recovers both metrics)')
    ap.add_argument('--data_path', type=str, default='/workspace/data')
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--layers_fixed', action='store_true')
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64)
    ap.add_argument('--shift', type=float, default=1.0, help='cường độ photometric shift (mô phỏng private)')
    ap.add_argument('--bottom_q', type=float, default=0.5, help='tỉ lệ bottom-score/ẢNH coi là confidently-normal')
    ap.add_argument('--adapt_size', type=int, default=30000, help='số patch normal-test thêm vào bank')
    ap.add_argument('--selfcal_pct', type=float, default=98.0, help='percentile self-cal threshold (RoBiS-style)')
    ap.add_argument('--thr_sigma', type=float, default=3.0, help='ngưỡng fixed = mean+kσ trên val/good')
    ap.add_argument('--max_eval', type=int, default=80, help='giới hạn ảnh eval/category cho nhanh')
    ap.add_argument('--max_val', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag20')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag20', args.out_dir).info
    torch.manual_seed(args.seed)

    bb = load_backbone(args.model, device)
    R = args.grid_tile * bb.patch
    if args.layers_fixed or not bb.n_layers:
        layers = [l for l in args.layers if l < (bb.n_layers or 1e9)]
    else:
        layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    T = args.tiles; gt = args.grid_tile; C_expect = None
    rng = np.random.default_rng(args.seed)

    p('=' * 104)
    p(f'DIAG20 TTNA premise | model={args.model} eff_grid={T*gt} layers={layers} | shift={args.shift} '
      f'bottom_q={args.bottom_q} adapt={args.adapt_size} selfcal_pct={args.selfcal_pct}')
    p('BASE(no-shift) -> SHIFT(bank train, val-thr) -> SHIFT+THR(self-cal) -> SHIFT+TTNA(bank\' + self-cal)')
    p('=' * 104)

    agg = {k: [] for k in ['base_au', 'base_f1', 'sh_au', 'sh_f1', 'thr_f1', 'ttna_au', 'ttna_f1', 'contam']}

    for cat in args.categories:
        # ---- bank train ----
        tr = sorted(glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.png')) +
                    glob.glob(os.path.join(args.data_path, cat, 'train', 'good', '*.jpg')))
        acc = []; keep = max(64, args.bank_size * 4 // max(1, len(tr) * T * T))
        with torch.no_grad():
            buf = []
            for pth in tr:
                buf.extend(tile_pils(Image.open(pth), T))
                while len(buf) >= args.enc_batch:
                    b = torch.stack([to_tensor(t, R) for t in buf[:args.enc_batch]]); buf = buf[args.enc_batch:]
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
        rng.shuffle(bad); rng.shuffle(good)
        eval_idx = bad[:args.max_eval] + good[:args.max_eval // 2]

        # ---- eval: extract feature BASE (no-shift) và SHIFT; giữ grid CPU để re-nn sau adapt ----
        base_grids, sh_grids, gts = [], [], []
        for i in tqdm(eval_idx, ncols=80, desc=f'  {cat}/eval'):
            pil = Image.open(ds.img_paths[i])
            gb = img_featmap(bb, pil, T, R, gt, layers, args.enc_batch)
            gs = img_featmap(bb, photometric_shift(pil, args.shift), T, R, gt, layers, args.enc_batch)
            base_grids.append(gb.cpu().numpy()); sh_grids.append(gs.cpu().numpy())
            gts.append(gt_grid(ds.gt_paths[i], ds.labels[i], gb.shape[0]))
        G = base_grids[0].shape[0]

        # ---- val/good (KHÔNG shift = calib sạch, đúng thực tế) cho ngưỡng fixed ----
        val_imgs = sorted(glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', '*.png')) +
                          glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', '*.jpg')))[:args.max_val]
        val_grids = [img_featmap(bb, Image.open(v), T, R, gt, layers, args.enc_batch).cpu().numpy() for v in val_imgs]

        def nn_all(grids, bnk):
            out = []
            for g in grids:
                d = nn_map_flat(torch.tensor(g.reshape(-1, Cdim), device=device), bnk, device)
                out.append(d.reshape(G, G).cpu().numpy())
            return out

        # ---------- BASE (no shift) ----------
        d_base = nn_all(base_grids, bank)
        lo_b, hi_b = np.percentile(np.stack(d_base), 1), np.percentile(np.stack(d_base), 99)
        m_base = norm_maps(d_base, lo_b, hi_b)
        au_base, _ = region_metrics(m_base, gts, gk, device)
        # thr fixed từ val (norm bằng lo/hi của base), upsample 256 gaussian, mean+kσ
        if val_grids:
            d_val = nn_all(val_grids, bank)
            vs = np.concatenate([upmap((dv - lo_b) / (hi_b - lo_b + 1e-8), 256, gk, device).reshape(-1) for dv in d_val])
            thr_base = float(vs.mean() + args.thr_sigma * vs.std())
        else:
            thr_base = float(np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in m_base]).mean())
        f1_base = segf1_fixed(m_base, gts, thr_base, gk, device)

        # ---------- SHIFT (bank train, val-thr = calib SAI) ----------
        d_sh = nn_all(sh_grids, bank)
        lo_s, hi_s = np.percentile(np.stack(d_sh), 1), np.percentile(np.stack(d_sh), 99)
        m_sh = norm_maps(d_sh, lo_s, hi_s)
        au_sh, _ = region_metrics(m_sh, gts, gk, device)
        # val-thr áp lên map shift (val KHÔNG shift -> ngưỡng lệch): dùng chính thr_base nhưng norm theo lo_s/hi_s
        if val_grids:
            vs_s = np.concatenate([upmap((dv - lo_s) / (hi_s - lo_s + 1e-8), 256, gk, device).reshape(-1) for dv in d_val])
            thr_valonshift = float(vs_s.mean() + args.thr_sigma * vs_s.std())
        else:
            thr_valonshift = thr_base
        f1_sh = segf1_fixed(m_sh, gts, thr_valonshift, gk, device)

        # ---------- self-cal threshold (RoBiS-style percentile trên map SHIFT) ----------
        pooled_sh = np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in m_sh])
        thr_selfcal = float(np.percentile(pooled_sh, args.selfcal_pct))
        f1_thr = segf1_fixed(m_sh, gts, thr_selfcal, gk, device)   # SHIFT+THR: chỉ đổi ngưỡng, bank vẫn train

        # ---------- TTNA: chọn confidently-normal (bottom-q/ẢNH theo s0) + adapt bank ----------
        sel_feats = []; contam_pos = 0; contam_tot = 0
        for g2d, d2d, gtg in zip(sh_grids, d_sh, gts):
            flat_d = d2d.reshape(-1)
            k = max(1, int(len(flat_d) * args.bottom_q))
            idx = np.argsort(flat_d)[:k]                      # patch điểm THẤP nhất của ảnh = normal đáng tin
            sel_feats.append(g2d.reshape(-1, Cdim)[idx])
            gflat = gtg.reshape(-1)
            contam_pos += int(gflat[idx].sum()); contam_tot += len(idx)
        sel = torch.tensor(np.concatenate(sel_feats), device=device)
        sel = subsample(sel, args.adapt_size)
        bank2 = torch.cat([bank, sel], 0)
        contam = 100.0 * contam_pos / max(1, contam_tot)

        d_ttna = nn_all(sh_grids, bank2)
        lo_t, hi_t = np.percentile(np.stack(d_ttna), 1), np.percentile(np.stack(d_ttna), 99)
        m_ttna = norm_maps(d_ttna, lo_t, hi_t)
        au_ttna, _ = region_metrics(m_ttna, gts, gk, device)
        pooled_t = np.concatenate([upmap(m, 256, gk, device).reshape(-1) for m in m_ttna])
        thr_t = float(np.percentile(pooled_t, args.selfcal_pct))
        f1_ttna = segf1_fixed(m_ttna, gts, thr_t, gk, device)

        for kk, vv in [('base_au', au_base), ('base_f1', f1_base), ('sh_au', au_sh), ('sh_f1', f1_sh),
                       ('thr_f1', f1_thr), ('ttna_au', au_ttna), ('ttna_f1', f1_ttna), ('contam', contam)]:
            agg[kk].append(vv)

        p(f'  [{cat:<11}] AUPRO05  base={au_base:.3f} shift={au_sh:.3f} '
          f'TTNA={au_ttna:.3f} ({au_ttna-au_sh:+.3f})  | SegF1  base={f1_base:.3f} shift={f1_sh:.3f} '
          f'+thr={f1_thr:.3f} +TTNA={f1_ttna:.3f}  | contam={contam:.1f}%')

    p('\n' + '=' * 104)
    M = {k: float(np.mean(v)) for k, v in agg.items()}
    p('{:<14}{:>10}{:>10}{:>10}{:>10}'.format('MEAN', 'AUPRO05', 'SegF1', 'dAU', 'dF1'))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10}{:>10}'.format('BASE(no-shift)', M['base_au'], M['base_f1'], '-', '-'))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}'.format('SHIFT', M['sh_au'], M['sh_f1'],
                                                          M['sh_au'] - M['base_au'], M['sh_f1'] - M['base_f1']))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10}{:>10.4f}'.format('SHIFT+THR', M['sh_au'], M['thr_f1'], '(=shift)',
                                                       M['thr_f1'] - M['sh_f1']))
    p('{:<14}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}'.format('SHIFT+TTNA', M['ttna_au'], M['ttna_f1'],
                                                          M['ttna_au'] - M['sh_au'], M['ttna_f1'] - M['sh_f1']))
    p(f'\ncontamination pool adapt (MEAN) = {M["contam"]:.2f}%  (thấp = defect KHÔNG bị bình-thường-hoá -> an toàn)')

    drop_au = M['base_au'] - M['sh_au']; drop_f1 = M['base_f1'] - M['sh_f1']
    rec_au = M['ttna_au'] - M['sh_au']; rec_f1 = M['ttna_f1'] - M['sh_f1']
    v_fail = 'PASS' if (drop_au > 0.02 and drop_f1 > 0.02) else 'FAIL'
    v_au = 'PASS' if rec_au > 0.01 else 'FAIL'
    v_f1 = 'PASS' if rec_f1 > 0.01 else 'FAIL'
    v_safe = 'PASS' if M['contam'] < 5.0 else 'FAIL'
    p('=' * 104)
    p(f'PREMISE FAILURE (shift tái hiện tụt private): {v_fail}  (dAUPRO05={-drop_au:+.3f} dSegF1={-drop_f1:+.3f})')
    p(f'PREMISE METHOD-AUPRO (bank-adapt cứu ranking): {v_au}  ({rec_au:+.3f})')
    p(f'PREMISE METHOD-SegF1 (self-cal cứu ngưỡng)   : {v_f1}  ({rec_f1:+.3f})')
    p(f'PREMISE SAFETY (contamination-robust)        : {v_safe}  (contam={M["contam"]:.2f}%)')
    p('=> BUILD TTNA nếu FAILURE + (METHOD-AUPRO hoặc METHOD-SegF1) PASS và SAFETY PASS.')
    p('   Nếu METHOD-AUPRO FAIL: normal-shift KHÔNG tách được defect ngay cả khi thêm normal-test')
    p('     -> shift làm defect==rare-normal (bất khả unsup), cần few-shot/SoftPRO gánh.')

    with open(os.path.join(args.out_dir, 'diag20.csv'), 'w') as fcsv:
        fcsv.write('metric,base,shift,shift_thr,shift_ttna\n')
        fcsv.write(f'AUPRO05,{M["base_au"]:.4f},{M["sh_au"]:.4f},{M["sh_au"]:.4f},{M["ttna_au"]:.4f}\n')
        fcsv.write(f'SegF1,{M["base_f1"]:.4f},{M["sh_f1"]:.4f},{M["thr_f1"]:.4f},{M["ttna_f1"]:.4f}\n')
        fcsv.write(f'contam_pct,{M["contam"]:.4f},,,\n')
    p(f'Đã lưu: {os.path.join(args.out_dir, "diag20.csv")}')


if __name__ == '__main__':
    main()
