# eval_fairthr_gen.py
# -----------------------------------------------------------------------------
# GENERALIZE TEST cho lever fairthr: chạy Y NGUYÊN protocol của eval_fairthr.py
#   (raw NN map + threshold train-side quantile x gain + morph closing 1 ô grid)
#   trên MVTec-AD (cũ) và VisA, với RULE ĐÓNG BĂNG từ AD2:
#       p = 95, gain = 1.15, morph = CÓ
#   KHÔNG đổi bất kỳ số nào cho dataset mới. Sweep (p x gain) vẫn in ra nhưng chỉ
#   để ĐO KHOẢNG CÁCH rule-đóng-băng vs best-per-dataset — đó là thước generalize.
#
#   FAIR: bank + threshold CHỈ train/good của dataset đích; GT chỉ để chấm.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   - MEAN SegF1@rule-đóng-băng >= 0.8 x MEAN best-combo của dataset đó
#     VÀ >= f1@ksig4.5 transductive  -> rule GENERALIZE, claim được trong paper
#     ("một rule chung, tự suy ngưỡng từ train/good của bất kỳ dataset nào").
#   - rule-đóng-băng < 0.5 x best-combo (rơi vào cliff gain)  -> rule KHÔNG
#     generalize, độ nhạy gain giết nó; khai thẳng, không re-tune lén.
#   - khoảng giữa (0.5-0.8x)  -> generalize một phần; khai gap, phân tích cat nào vỡ.
#   MỌI KẾT QUẢ ĐỀU DÙNG ĐƯỢC. Không có nhánh "chạy lại cho đẹp".
#
#   python eval_fairthr_gen.py --dataset mvtec --data_path ../data_mvtec --out_dir ./fairthr_mvtec --max_eval 30
#   python eval_fairthr_gen.py --dataset visa  --data_path ../data_visa  --out_dir ./fairthr_visa  --max_eval 30
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
from infer_submit_mvtec_ad2 import build_bank, IMG_EXT                     # noqa: E402
from eval_fairthr import nn_grid, closing, PS, GAINS                       # noqa: E402
from eval_generalize import SimpleADDataset, CATS                          # noqa: E402
from eval_native import Hist, make_map                                     # noqa: E402
from diag30_thin_premise import aupro05                                    # noqa: E402
from utils import get_gaussian_kernel, get_logger                          # noqa: E402
from backbones_ext import load_backbone                                    # noqa: E402

warnings.filterwarnings('ignore')

RULE = (95.0, 1.15)          # ĐÓNG BĂNG từ AD2 (eval_fairthr 2026-08-19). KHÔNG ĐỔI.
RULE_MORPH = True


def run_cat(bb, cat, args, layers, gk, device, p, rng):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
    root = os.path.join(args.data_path, cat)
    tr = sorted(sum([glob.glob(os.path.join(root, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr
    va = tr[args.max_train:args.max_train + args.n_val] if args.max_train else []
    overlap = False
    if len(va) < 3:
        va, overlap = tr_use[-args.n_val:], True

    try:
        ds = SimpleADDataset(root)
    except FileNotFoundError as e:
        p(f'  [{cat}] {e} -> bỏ'); return None
    bad = [i for i in range(len(ds)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={G} heldout={len(va)}{" [TRÙNG BANK]" if overlap else ""}')

    bank = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    p(f'    [{cat}] bank={bank.shape[0]} C={bank.shape[1]}')

    # ---- ngưỡng train-side (y hệt eval_fairthr) ----
    tr_px = []
    for path in tqdm(va, ncols=70, desc=f'    {cat} heldout', leave=False):
        pil = Image.open(path)
        W, H = pil.size
        s = nn_grid(bb, pil, T, R, gt_, layers, args.enc_batch, bank, device)
        m = make_map(s, args.canvas, gk, (H, W), device).cpu().numpy()
        tr_px.append(m.ravel()[::4])
    tr_px = np.concatenate(tr_px)
    thrs = {(pp, gg): float(np.percentile(tr_px, pp)) * gg for pp in PS for gg in GAINS}
    del tr_px

    # ---- test ----
    s_grids = []
    for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
        s_grids.append(nn_grid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch, bank, device))
    del bank; torch.cuda.empty_cache()

    gmin = min(float(s.min()) for s in s_grids)
    gmax = max(float(s.max()) for s in s_grids)
    h = Hist(gmin - 0.05, gmax + 0.05)
    ap_preds, ap_gts = [], []
    mstat = {k: np.zeros(3, np.float64) for k in thrs}
    for s, i in zip(s_grids, idx):
        W, H = Image.open(ds.img_paths[i]).size
        m_t = make_map(s, args.canvas, gk, (H, W), device)
        m_nat = m_t.cpu().numpy()
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), np.uint8)
            g_ap = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
        else:
            gpil = Image.open(ds.gt_paths[i]).convert('L')
            if gpil.size != (W, H):                                        # vài GT VisA lệch size
                gpil = gpil.resize((W, H), Image.NEAREST)
            g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
            g_ap = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
        h.add(m_nat.reshape(-1), g_nat.reshape(-1))
        ap_preds.append(make_map(s, args.canvas, gk, (args.aupro_res, args.aupro_res), device).cpu().numpy())
        ap_gts.append(g_ap)
        r = max(1, round(min(H, W) / G))
        k = 2 * r + 1
        g_t = torch.from_numpy(g_nat).to(device) > 0
        for key, thr in thrs.items():
            pred = closing(m_t > thr, k)
            tp = (pred & g_t).sum().item(); fp = (pred & ~g_t).sum().item(); fn = ((~pred) & g_t).sum().item()
            mstat[key] += (tp, fp, fn)
        del m_t, g_t
    aupro = aupro05(ap_preds, ap_gts, rng)
    del s_grids, ap_preds, ap_gts

    out = {'aupro': aupro, 'f1_max': h.f1_max(), 'f1_ksig': h.f1_at(h.ksig(args.thr_sigma)),
           'f1': {}, 'f1m': {}}
    for key, thr in thrs.items():
        out['f1'][key] = h.f1_at(thr)
        tp, fp, fn = mstat[key]
        out['f1m'][key] = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    rf = out['f1m'][RULE] if RULE_MORPH else out['f1'][RULE]
    bestv = max(max(out['f1'].values()), max(out['f1m'].values()))
    p(f'    [{cat}] AUPRO0.05={aupro:.4f}  trần={out["f1_max"]:.4f}  f1@ksig{args.thr_sigma}={out["f1_ksig"]:.4f} '
      f'| RULE-AD2={rf:.4f}  best-combo={bestv:.4f}')
    for pp in PS:
        row = '  '.join(f'g{gg}:{out["f1"][(pp, gg)]:.3f}/{out["f1m"][(pp, gg)]:.3f}' for gg in GAINS)
        p(f'      p{pp:<5}: {row}   (f1/f1+morph)')
    return out


def main():
    ap = argparse.ArgumentParser('fairthr generalize: rule AD2 đóng băng trên MVTec-AD / VisA')
    ap.add_argument('--dataset', type=str, required=True, choices=['visa', 'mvtec'])
    ap.add_argument('--data_path', type=str, required=True)
    # ==== default = y nguyên eval_fairthr trên AD2. ĐỪNG ĐỔI. ====
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--thr_sigma', type=float, default=4.5, help='chỉ để in mốc transductive')
    ap.add_argument('--n_val', type=int, default=16)
    # =============================================================
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split. Thử nhanh: 30')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=None)
    ap.add_argument('--out_dir', type=str, default='./fairthr_gen')
    args = ap.parse_args()

    cats = args.categories or CATS[args.dataset]
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fairthr_' + args.dataset, args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'dataset={args.dataset} ({len(cats)} cat) | RULE ĐÓNG BĂNG TỪ AD2: p={RULE[0]} gain={RULE[1]} '
      f'morph={"CÓ" if RULE_MORPH else "KHÔNG"} | eff_grid={args.tiles * args.grid_tile} bank={args.bank_size}')
    p('  FAIR: bank + threshold CHỈ train/good dataset đích. Sweep chỉ để đo gap vs best, KHÔNG re-tune.')

    res = {}
    for cat in cats:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được — kiểm tra --data_path/layout.'); return

    p('\n' + '=' * 84 + f'\n===== TỔNG {args.dataset} ({len(res)} cat) =====')
    au = float(np.mean([res[c]['aupro'] for c in res]))
    fm = float(np.mean([res[c]['f1_max'] for c in res]))
    fk = float(np.mean([res[c]['f1_ksig'] for c in res]))
    rule_f = float(np.mean([(res[c]['f1m'] if RULE_MORPH else res[c]['f1'])[RULE] for c in res]))
    p(f'  AUPRO0.05(raw)={au:.4f}   trần={fm:.4f}   f1@ksig(transductive)={fk:.4f}')
    best = None
    for pp in PS:
        for gg in GAINS:
            f1 = float(np.mean([res[c]['f1'][(pp, gg)] for c in res]))
            f1m = float(np.mean([res[c]['f1m'][(pp, gg)] for c in res]))
            mk = ' <-- RULE AD2' if (pp, gg) == RULE else ''
            p(f'  p{pp:<5} g{gg:<4}: SegF1={f1:.4f}  +morph={f1m:.4f}{mk}')
            best = max(best or 0.0, f1, f1m)
    ratio = rule_f / (best + 1e-9)
    p(f'\n  RULE AD2 (p={RULE[0]} g={RULE[1]} morph={"CÓ" if RULE_MORPH else "KHÔNG"}): SegF1={rule_f:.4f}')
    p(f'  BEST combo dataset này: {best:.4f}  ->  rule/best = {ratio:.2f}   (rule/trần = {rule_f / (fm + 1e-9):.2f})')
    p('\nĐỌC (pre-registered): rule/best >= 0.8 & rule >= ksig -> GENERALIZE, claim được. '
      '< 0.5 -> cliff gain giết rule, khai thẳng. 0.5-0.8 -> một phần, phân tích cat vỡ.')


if __name__ == '__main__':
    main()
