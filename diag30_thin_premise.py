# diag30_thin_premise.py
# -----------------------------------------------------------------------------
# EVAL UNSUPERVISED CÔNG BẰNG (fair) — đo bằng ĐÚNG thước paper: AUPRO0.05 + SegF1(native).
#   AIM: AUPRO0.05 >= 0.80 (8x), SegF1 >= 0.60 (6x).
#
# FAIR — KHÔNG feed GT vào model:
#   * KHÔNG head, KHÔNG label, KHÔNG shot. Bank CHỈ từ train/good.
#   * GT test_public CHỈ dùng để CHẤM hai metric (không thể đo segmentation mà thiếu GT),
#     tuyệt đối không đi vào score/threshold/bank.
#   * Ngưỡng SegF1 = self-cal mean+kσ trên chính phân bố map (label-free, như production test_ksig).
#
# SO SÁNH (cô lập đòn bẩy), qua các eff_grid (--deep_configs):
#   deep : nn-distance thuần (PatchCore-style, unsup)              <- baseline công bằng
#   hf   : memory bank TẦN-SỐ-CAO thuần (high-pass residual, unsup) <- tín hiệu MỚI, orthogonal
#   max  : fuse tham-số-tự-do = max(normalize(deep), normalize(hf))
#   avg  : fuse = trung bình 2 map đã chuẩn hóa
#   (max/avg KHÔNG tinh chỉnh trọng số trên GT -> vẫn fair.)
#
# Thước KHỚP eval_native / eval_nrs_head để số so được với các bảng cũ:
#   SegF1@ksig(native) + SegF1_max(native) qua Hist; AUPRO0.05@512 qua ader_evaluator.
#
# ĐỌC:
#   - deep chạm 0.80/0.60 luôn -> baseline unsup đã đủ, khỏi cần HF.
#   - fuse(max/avg) > deep rõ  -> HF-bank là đòn bẩy unsup MỚI, sạch, đúng thể loại PatchCore
#                                  -> đây là novelty để xây method (fuse deep-bank + hf-bank).
#   - tăng eff_grid kéo cả 2 metric lên -> high-native distance là hướng phụ đáng tiền.
#   - HF ~ vô ích & deep plateau xa 0.80/0.60 -> unsup thuần chưa tới; báo cáo trung thực.
#
#   python diag30_thin_premise.py --data_path ../data --out_dir ./fair30 \
#       --deep_configs 3x48 --categories can sheet_metal fruit_jelly vial fabric rice wallplugs walnuts
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
from scipy.ndimage import gaussian_filter

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, img_featgrid, nn_map, VALID, IMG_EXT,
)
from eval_native import Hist, make_map                                  # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']
FUSIONS = ['deep', 'hf', 'max', 'avg']


def aupro05(preds, gts, rng):
    """AUPRO0.05 KHỚP eval_nrs_head: sp = max mỗi map; nhiễu nhỏ nếu phẳng để tránh chia 0."""
    sp = np.array([float(p.max()) for p in preds])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + rng.normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(preds), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


# ============================ tín hiệu HF (memory bank tần-số-cao, unsup) =====
def hf_desc_grid(pil, work_res, Ghf, sigma):
    """Mô tả tần-số-cao cục bộ, translation-invariant (PatchCore trên HF, KHÔNG deep, KHÔNG nhãn).
    high-pass = gray - gaussian(gray); pool về Ghf, mỗi ô -> [mean|hp|, std, max|hp|]."""
    g = np.asarray(pil.convert('L').resize((work_res, work_res), Image.BILINEAR), np.float32) / 255.0
    hp = g - gaussian_filter(g, sigma)
    b = work_res // Ghf
    hp = hp[:b * Ghf, :b * Ghf].reshape(Ghf, b, Ghf, b)
    a = np.abs(hp)
    d = np.stack([a.mean((1, 3)), hp.std((1, 3)), a.max((1, 3))], -1)
    return d.reshape(-1, 3)


def build_hf_bank(train_imgs, work_res, Ghf, sigma, bank_cap, rng):
    D = np.concatenate([hf_desc_grid(Image.open(p), work_res, Ghf, sigma) for p in train_imgs], 0)
    mu, sd = D.mean(0), D.std(0) + 1e-6
    Dn = (D - mu) / sd
    if len(Dn) > bank_cap:
        Dn = Dn[rng.choice(len(Dn), bank_cap, replace=False)]
    return torch.tensor(Dn, dtype=torch.float32), mu, sd


def hf_distgrid(pil, work_res, Ghf, sigma, bank, mu, sd, device, chunk=4096):
    d = (hf_desc_grid(pil, work_res, Ghf, sigma) - mu) / sd
    q = torch.tensor(d, dtype=torch.float32, device=device)
    bk = bank.to(device)
    out = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = torch.cdist(q[s:s + chunk], bk).min(1)[0]
    return out.reshape(Ghf, Ghf).cpu().numpy()


def norm01(grids):
    lo, hi = np.percentile(np.concatenate([g.reshape(-1) for g in grids]), [1, 99])
    return [((g - lo) / (hi - lo + 1e-8)).astype(np.float32) for g in grids], (lo, hi)


# ============================ metric từ tập s_grid (G,G) ======================
def eval_sgrids(s_grids, sizes, idx, ds, canvas, gk, aupro_res, thr_sigma, device, rng):
    gmin = min(float(s.min()) for s in s_grids)
    gmax = max(float(s.max()) for s in s_grids)
    h = Hist(gmin - 0.05, gmax + 0.05)
    ap_preds, ap_gts = [], []
    for s, (H, W), i in zip(s_grids, sizes, idx):
        m_nat = make_map(s, canvas, gk, (H, W), device).cpu().numpy()             # SegF1 @ native
        m_ap = make_map(s, canvas, gk, (aupro_res, aupro_res), device).cpu().numpy()  # AUPRO @512
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), np.uint8)
            g_ap = np.zeros((aupro_res, aupro_res), np.uint8)
        else:
            gpil = Image.open(ds.gt_paths[i]).convert('L')
            g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
            g_ap = (np.asarray(gpil.resize((aupro_res, aupro_res), Image.BOX)) > 0).astype(np.uint8)
        h.add(m_nat.reshape(-1), g_nat.reshape(-1))
        ap_preds.append(m_ap); ap_gts.append(g_ap)
    return {'segf1': h.f1_at(h.ksig(thr_sigma)), 'segf1_max': h.f1_max(),
            'aupro': aupro05(ap_preds, ap_gts, rng)}


# ============================ chạy một category ==============================
def run_cat(bb, cat, args, layers, gk, device, p, rng):
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không có train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    sizes = [(Image.open(ds.img_paths[i]).size[1], Image.open(ds.img_paths[i]).size[0]) for i in idx]
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | train bank={len(tr_use)}')

    # ---- HF grids (1 lần, độc lập deep config) ----
    hf_bank, hmu, hsd = build_hf_bank(tr_use, args.hf_res, args.hf_grid, args.hf_sigma, args.bank_size, rng)
    hf_raw = [hf_distgrid(Image.open(ds.img_paths[i]), args.hf_res, args.hf_grid, args.hf_sigma,
                          hf_bank, hmu, hsd, device) for i in tqdm(idx, ncols=70, desc=f'    {cat} hf', leave=False)]
    hf_n, _ = norm01(hf_raw)                                                   # (Ghf,Ghf) chuẩn hóa

    out = {}
    for (T, gt_) in args.deep_configs:
        tag = f'{T}x{gt_}={T * gt_}'
        R = gt_ * bb.patch
        bank = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
        G = T * gt_
        deep_raw = []
        with torch.no_grad():
            for i in tqdm(idx, ncols=70, desc=f'    {cat} deep {tag}', leave=False):
                g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
                deep_raw.append(np.asarray(nn_map(g, bank, device)))
        del bank
        deep_n, _ = norm01(deep_raw)
        # HF chuẩn hóa -> đưa về lưới G để fuse
        hf_G = [F.interpolate(torch.tensor(hn)[None, None], size=G, mode='bilinear',
                              align_corners=False)[0, 0].numpy().astype(np.float32) for hn in hf_n]

        out[tag] = {}
        for fu in args.fusions:
            if fu == 'deep':
                sg = deep_n
            elif fu == 'hf':
                sg = hf_G
            elif fu == 'max':
                sg = [np.maximum(d, h).astype(np.float32) for d, h in zip(deep_n, hf_G)]
            else:  # avg
                sg = [(0.5 * (d + h)).astype(np.float32) for d, h in zip(deep_n, hf_G)]
            m = eval_sgrids(sg, sizes, idx, ds, args.canvas, gk, args.aupro_res, args.thr_sigma, device, rng)
            out[tag][fu] = m
            p(f'    [{cat}] {tag:9s} {fu:4s}: AUPRO0.05={m["aupro"]:.4f}  '
              f'SegF1={m["segf1"]:.4f}  trần={m["segf1_max"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('diag30: eval UNSUP fair — AUPRO0.05 + SegF1(native), deep vs +HF-bank')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--deep_configs', type=str, default='3x48',
                    help='eff_grid sweep "T x gt" ngăn bởi phẩy, vd 3x48,4x48')
    ap.add_argument('--fusions', type=str, nargs='+', default=FUSIONS, choices=FUSIONS)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split (fair); >0 để chạy nhanh')
    ap.add_argument('--canvas', type=int, default=256, help='độ phân giải cổ chai + gaussian (256 = production)')
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--thr_sigma', type=float, default=4.5, help='k cho ngưỡng SegF1 self-cal (label-free)')
    # HF-bank
    ap.add_argument('--hf_res', type=int, default=1024)
    ap.add_argument('--hf_grid', type=int, default=256)
    ap.add_argument('--hf_sigma', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./fair30')
    args = ap.parse_args()

    args.deep_configs = [(int(a), int(b)) for a, b in (c.split('x') for c in args.deep_configs.split(','))]
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fair30', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=max(3, int(5 * args.canvas / 256) | 1),
                             sigma=4.0 * args.canvas / 256).to(device)
    p(f'device={device} model={args.model} layers={args.layers}->{layers} | deep_configs={args.deep_configs} '
      f'canvas={args.canvas} aupro_res={args.aupro_res} k={args.thr_sigma} | HF res={args.hf_res} grid={args.hf_grid}')
    p('  AIM: AUPRO0.05>=0.80 (8x), SegF1>=0.60 (6x)')
    p('  FAIR: KHÔNG head/label/shot; bank chỉ train/good; GT CHỈ để chấm metric, không vào model.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không có category nào chạy được.'); return

    tags = [f'{T}x{gt_}={T * gt_}' for T, gt_ in args.deep_configs]
    p('\n' + '=' * 92 + '\n===== MEAN qua các category (AUPRO0.05 / SegF1 / trần) =====')
    for tag in tags:
        for fu in args.fusions:
            au = float(np.mean([res[c][tag][fu]['aupro'] for c in res]))
            f1 = float(np.mean([res[c][tag][fu]['segf1'] for c in res]))
            fm = float(np.mean([res[c][tag][fu]['segf1_max'] for c in res]))
            hit = ('  <-- AIM' if (au >= 0.80 and f1 >= 0.60) else
                   ('  (AUPRO đạt)' if au >= 0.80 else ('  (SegF1 đạt)' if f1 >= 0.60 else '')))
            p(f'  [{tag:9s}] {fu:4s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}  trần={fm:.4f}{hit}')

    p('\nĐỌC:')
    p('  - deep đạt 0.80/0.60          -> baseline unsup đủ, HF không cần.')
    p('  - max/avg > deep rõ           -> HF-bank là đòn bẩy unsup MỚI (novelty: fuse deep+HF bank).')
    p('  - eff_grid cao kéo cả 2 lên   -> high-native distance là hướng phụ đáng tiền.')
    p('  - HF vô ích & deep xa AIM     -> unsup thuần chưa tới; báo cáo trung thực, cân nhắc NRS lại.')


if __name__ == '__main__':
    main()
