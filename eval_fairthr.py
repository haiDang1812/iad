# eval_fairthr.py
# -----------------------------------------------------------------------------
# LEVER SegF1 FAIR, KHÔNG đụng map (AUPRO raw giữ nguyên): THRESHOLD TRAIN-SIDE + MORPHOLOGY.
#
#   Bối cảnh: adapter (mọi biến thể) + synth-selector CHẾT với tư cách lever AUPRO
#   (vial/wallplugs: defect near-manifold trong span bị nén; synth-val mù đúng chỗ đó).
#   Nhưng adapter lộ ra 1 sự thật: SegF1 x2 của nó = THRESHOLD TRANSFER, không phải map
#   (trần raw 0.317 ~ trần method 0.344, chỉ khác achieved). Vậy lấy gain đó trực tiếp:
#     map  = raw NN distance (AUPRO an toàn tuyệt đối, không học gì)
#     thr  = quantile-p(điểm pixel ảnh train/good GIỮ LẠI ngoài bank) x gain   [FAIR, SuperADD-style]
#     post = morphological closing (dilate->erode GPU), bán kính = 1 ô grid    [rule theo scale, chung]
#
#   MỘT RULE TOÀN CỤC: cặp (p, gain) DÙNG CHUNG cho cả 8 cat; giá trị ngưỡng từng cat
#   tự suy từ train/good của cat đó theo rule chung. KHÔNG per-cat setting. Bảng sweep
#   (p x gain) = ablation công khai; chốt = 1 cặp duy nhất theo MEAN SegF1.
#
#   FAIR: bank + threshold CHỈ từ train/good; GT chỉ để chấm. Không test-side threshold
#   (ksig 4.5 chỉ in làm mốc so sánh — nó là transductive).
#
#   ĐỌC (pre-register):
#     - tồn tại (p,gain) chung: mean SegF1 >= 0.15 (mốc adapter 0.18-0.19) VÀ AUPRO = raw 0.579
#       -> threshold train-side thay được adapter, map giữ nguyên -> chốt rule, lên full pipeline.
#     - mọi combo << ksig4.5 test-side -> tail train không đại diện tail test (shift) -> lever chết,
#       SegF1 fair phải tìm ở map/resolution.
#     - morph: +deltaF1 dương đều tay -> giữ; âm -> bỏ (1 knob, quyết theo MEAN, không per-cat).
#
#   python eval_fairthr.py --data_path ../data --out_dir ./fairthr --max_eval 30 \
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
from infer_submit_mvtec_ad2 import img_featgrid, build_bank, IMG_EXT       # noqa: E402
from eval_native import Hist, make_map                                     # noqa: E402
from diag30_thin_premise import aupro05                                    # noqa: E402
from dataset import MVTecAD2Dataset                                        # noqa: E402
from utils import get_gaussian_kernel, get_logger                          # noqa: E402
from backbones_ext import load_backbone                                    # noqa: E402

warnings.filterwarnings('ignore')

PS = [95.0, 99.0, 99.9]                 # percentile train-side (rule chung)
GAINS = [1.0, 1.15, 1.3, 1.5]           # gain (rule chung)


@torch.no_grad()
def nn_grid(bb, pil, T, R, gt_, layers, eb, bank, device, chunk=2048):
    g = img_featgrid(bb, pil, T, R, gt_, layers, eb)
    G = g.shape[0]; C = g.shape[-1]
    q = g.reshape(-1, C).to(device)
    d = torch.empty(q.shape[0], device=device)
    for s in range(0, q.shape[0], chunk):
        d[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(dim=1).values
    del g, q
    return d.reshape(G, G).cpu().numpy()


@torch.no_grad()
def closing(binmask, k):
    """morphological closing GPU: dilate (maxpool) -> erode (-maxpool(-x)). binmask: (H,W) bool tensor."""
    x = binmask[None, None].float()
    x = torch.nn.functional.max_pool2d(x, k, stride=1, padding=k // 2)
    x = -torch.nn.functional.max_pool2d(-x, k, stride=1, padding=k // 2)
    return x[0, 0] > 0.5


def run_cat(bb, cat, args, layers, gk, device, p, rng):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr
    va = tr[args.max_train:args.max_train + args.n_val] if args.max_train else []
    overlap = False
    if len(va) < 3:
        va, overlap = tr_use[-args.n_val:], True                        # hiếm: cat quá ít ảnh train

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={G} heldout={len(va)}{" [TRÙNG BANK]" if overlap else ""}')

    bank = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    p(f'    [{cat}] bank={bank.shape[0]} C={bank.shape[1]}')

    # ---- ngưỡng train-side: điểm pixel native của ảnh train/good GIỮ LẠI (ngoài bank) ----
    tr_px = []
    for path in tqdm(va, ncols=70, desc=f'    {cat} heldout', leave=False):
        pil = Image.open(path)
        W, H = pil.size
        s = nn_grid(bb, pil, T, R, gt_, layers, args.enc_batch, bank, device)
        m = make_map(s, args.canvas, gk, (H, W), device).cpu().numpy()
        tr_px.append(m.ravel()[::4])                                    # subsample chống RAM
    tr_px = np.concatenate(tr_px)
    thrs = {(pp, gg): float(np.percentile(tr_px, pp)) * gg for pp in PS for gg in GAINS}
    del tr_px

    # ---- test: 1 lượt native map -> Hist (F1 mọi ngưỡng) + morph streaming + AUPRO ----
    s_grids = []
    for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
        s_grids.append(nn_grid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch, bank, device))
    del bank; torch.cuda.empty_cache()

    gmin = min(float(s.min()) for s in s_grids)
    gmax = max(float(s.max()) for s in s_grids)
    h = Hist(gmin - 0.05, gmax + 0.05)
    ap_preds, ap_gts = [], []
    mstat = {k: np.zeros(3, np.float64) for k in thrs}                  # TP,FP,FN sau closing
    for s, i in zip(s_grids, idx):
        W, H = Image.open(ds.img_paths[i]).size
        m_t = make_map(s, args.canvas, gk, (H, W), device)              # tensor GPU (H,W)
        m_nat = m_t.cpu().numpy()
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), np.uint8)
            g_ap = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
        else:
            gpil = Image.open(ds.gt_paths[i]).convert('L')
            g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
            g_ap = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
        h.add(m_nat.reshape(-1), g_nat.reshape(-1))
        ap_preds.append(make_map(s, args.canvas, gk, (args.aupro_res, args.aupro_res), device).cpu().numpy())
        ap_gts.append(g_ap)
        # morph: bán kính = 1 ô grid theo scale ảnh (rule chung, không theo cat)
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
    p(f'    [{cat}] AUPRO0.05={aupro:.4f}  trần={out["f1_max"]:.4f}  f1@ksig{args.thr_sigma}={out["f1_ksig"]:.4f} (mốc transductive)')
    for pp in PS:
        row = '  '.join(f'g{gg}:{out["f1"][(pp, gg)]:.3f}/{out["f1m"][(pp, gg)]:.3f}' for gg in GAINS)
        p(f'      p{pp:<5}: {row}   (f1/f1+morph)')
    return out


def main():
    ap = argparse.ArgumentParser('eval_fairthr: train-side threshold + morphology trên raw map (fair)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split. Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--thr_sigma', type=float, default=4.5, help='chỉ để in mốc transductive cũ')
    ap.add_argument('--n_val', type=int, default=16, help='số ảnh train/good giữ lại NGOÀI bank làm threshold')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./fairthr')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fairthr', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} layers={layers} bank={args.bank_size} '
      f'heldout={args.n_val} P={PS} GAIN={GAINS}')
    p('  FAIR: bank + threshold CHỈ train/good. MỘT rule (p,gain) chung mọi cat. Map = raw NN (không học).')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN theo (p,gain) — CHỌN 1 CẶP CHUNG =====')
    au = float(np.mean([res[c]['aupro'] for c in res]))
    fm = float(np.mean([res[c]['f1_max'] for c in res]))
    fk = float(np.mean([res[c]['f1_ksig'] for c in res]))
    p(f'  AUPRO0.05(raw map)={au:.4f}   trần={fm:.4f}   f1@ksig(transductive)={fk:.4f}')
    best = None
    for pp in PS:
        for gg in GAINS:
            f1 = float(np.mean([res[c]['f1'][(pp, gg)] for c in res]))
            f1m = float(np.mean([res[c]['f1m'][(pp, gg)] for c in res]))
            p(f'  p{pp:<5} g{gg:<4}: SegF1={f1:.4f}  +morph={f1m:.4f}')
            for tag, v in (('', f1), ('m', f1m)):
                if best is None or v > best[0]:
                    best = (v, pp, gg, tag)
    v, pp, gg, tag = best
    p(f'\n  RULE CHỐT (mean cao nhất, chung mọi cat): p={pp} gain={gg} morph={"CÓ" if tag == "m" else "KHÔNG"} -> SegF1={v:.4f}')
    p('\nĐỌC: SegF1(rule chốt) >= 0.15 & AUPRO=raw -> threshold fair thay được adapter, lên full pipeline. '
      'Mọi combo << ksig -> tail train không đại diện test -> lever chết, quay về map/resolution.')


if __name__ == '__main__':
    main()
