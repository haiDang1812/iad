# eval_overlapmap.py
# -----------------------------------------------------------------------------
# LEVER MAP tiếp theo (sau bankmap: CORESET VÀO nền +0.030 AUPRO/+0.030 trần; knn3 chết):
#   NỀN MỚI = 3:48 + bank coreset 50k (từ cand 200k) + 1-NN + RULE ĐÓNG BĂNG p95 g1.15 morph.
#   MỘT thay đổi duy nhất: SƠ ĐỒ TILING non-overlap  ->  CỬA SỔ CHỒNG NỬA + TRUNG BÌNH SCORE
#   (cơ chế SuperADD 640/128: mỗi vùng được nhìn trong nhiều ngữ cảnh cửa sổ, score = mean).
#   Sơ đồ áp dụng ĐỒNG NHẤT cho bank + heldout + test (bank phải thấy cùng phân bố ngữ cảnh).
#
#   base: 3x3 cửa sổ không chồng (tái lập core1 của bankmap trong-run)
#   ov  : (2T-1)^2 = 5x5 cửa sổ cùng cỡ, stride nửa tile; score 1-NN per-window dán vào
#         canvas G=144 với sum/count -> trung bình vùng chồng lấn.
#
#   FAIR: bank + threshold CHỈ train/good; GT chỉ để chấm. Không setting per-cat.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   - Δ(ov - base) mean: AUPRO >= +0.010 hoặc trần >= +0.020 -> OVERLAP VÀO config nền.
#   - ~0 / âm -> overlap chết ở quick harness -> lever map còn lại = guided native-res
#     upsampling (mặt trận novelty) + full-res khi submit.
#
#   python eval_overlapmap.py --data_path ../data --out_dir ./overlapmap --max_eval 30 \
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
from infer_submit_mvtec_ad2 import img_featgrid, build_bank, to_tensor, nn_map, IMG_EXT  # noqa: E402
from eval_bankmap import coreset                                                   # noqa: E402
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from diag30_thin_premise import aupro05                                            # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

RULE_P, RULE_G = 95.0, 1.15          # ĐÓNG BĂNG (fairthr, argmax 3 lần). KHÔNG ĐỔI.
VARIANTS = ['base', 'ov']


def win_pils(pil, T, ov2):
    """Cửa sổ cỡ 1 tile, stride tile/ov2. ov2=1 -> đúng tile_pils; ov2=2 -> chồng nửa, (2T-1)^2."""
    w, h = pil.size
    n = ov2 * (T - 1) + 1
    out = []
    for i in range(n):
        for j in range(n):
            x0 = round(j * w / (T * ov2)); y0 = round(i * h / (T * ov2))
            out.append(pil.crop((x0, y0, min(w, x0 + round(w / T)), min(h, y0 + round(h / T)))))
    return out


@torch.no_grad()
def win_feats(bb, pil, T, R, gt, layers, eb, ov2):
    """(n*n, gt*gt, C) feature các cửa sổ chồng lấn (cùng tiền xử lý to_tensor như tile)."""
    wins = win_pils(pil, T, ov2)
    parts = []
    for s in range(0, len(wins), eb):
        b = torch.stack([to_tensor(t, R) for t in wins[s:s + eb]])
        parts.append(bb.extract(b, layers))
    return torch.cat(parts, 0)[:, :gt * gt]


@torch.no_grad()
def overlap_score(bb, pil, T, R, gt, layers, eb, bank, device, ov2=2):
    """Score 1-NN per-window dán vào canvas G=T*gt với sum/count -> trung bình overlap."""
    f = win_feats(bb, pil, T, R, gt, layers, eb, ov2)
    n = ov2 * (T - 1) + 1
    st = gt // ov2
    G = T * gt
    acc = np.zeros((G, G), np.float64); cnt = np.zeros((G, G), np.float64)
    for k in range(n * n):
        d = nn_map(f[k].reshape(gt, gt, -1), bank, device)
        i, j = k // n, k % n
        acc[i * st:i * st + gt, j * st:j * st + gt] += d
        cnt[i * st:i * st + gt, j * st:j * st + gt] += 1
    del f
    return (acc / cnt).astype(np.float32)


@torch.no_grad()
def build_cand_overlap(bb, tr, T, R, gt, layers, eb, cand_size, device, ov2=2):
    """Pool ứng viên từ cửa sổ CHỒNG LẤN trên train/good (mirror build_bank, đổi tiling)."""
    from infer_submit_mvtec_ad2 import subsample
    n = ov2 * (T - 1) + 1
    keep = max(64, cand_size * 4 // max(1, len(tr) * n * n))
    acc = []; buf = []
    for pth in tr:
        buf.extend(win_pils(Image.open(pth), T, ov2))
        while len(buf) >= eb:
            b = torch.stack([to_tensor(t, R) for t in buf[:eb]]); buf = buf[eb:]
            f = bb.extract(b, layers)
            acc.append(subsample(f.reshape(-1, f.shape[-1]), eb * keep).cpu())
    if buf:
        f = bb.extract(torch.stack([to_tensor(t, R) for t in buf]), layers)
        acc.append(subsample(f.reshape(-1, f.shape[-1]), len(buf) * keep).cpu())
    return subsample(torch.cat(acc, 0), cand_size)


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
    assert gt_ % 2 == 0, 'grid_tile phải chẵn cho stride nửa tile'
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

    # ---- 2 bank coreset, mỗi variant từ tiling của chính nó ----
    banks = {}
    cand = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.cand_size, device)
    banks['base'] = coreset(cand, args.bank_size, device, seed=args.seed)
    del cand; torch.cuda.empty_cache()
    cand = build_cand_overlap(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.cand_size, device).to(device)
    banks['ov'] = coreset(cand, args.bank_size, device, seed=args.seed)
    del cand; torch.cuda.empty_cache()
    p(f'    [{cat}] bank base={banks["base"].shape[0]} ov={banks["ov"].shape[0]} (coreset tu cand={args.cand_size})')

    def score_grid(pil, v):
        if v == 'base':
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            d = nn_map(g, banks['base'], device); del g
            return d
        return overlap_score(bb, pil, T, R, gt_, layers, args.enc_batch, banks['ov'], device)

    # ---- heldout -> ngưỡng per-variant theo RULE đóng băng ----
    tr_px = {v: [] for v in VARIANTS}
    for path in tqdm(va, ncols=70, desc=f'    {cat} heldout', leave=False):
        pil = Image.open(path)
        W, H = pil.size
        for v in VARIANTS:
            m = make_map(score_grid(pil, v), args.canvas, gk, (H, W), device).cpu().numpy()
            tr_px[v].append(m.ravel()[::4])
    thr = {v: float(np.percentile(np.concatenate(tr_px[v]), RULE_P)) * RULE_G for v in VARIANTS}
    del tr_px

    # ---- test ----
    s_grids = {v: [] for v in VARIANTS}
    for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
        pil = Image.open(ds.img_paths[i])
        for v in VARIANTS:
            s_grids[v].append(score_grid(pil, v))
    del banks; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        gmin = min(float(s.min()) for s in s_grids[v])
        gmax = max(float(s.max()) for s in s_grids[v])
        h = Hist(gmin - 0.05, gmax + 0.05)
        ap_preds, ap_gts = [], []
        mst = np.zeros(3, np.float64)
        for s, i in zip(s_grids[v], idx):
            W, H = Image.open(ds.img_paths[i]).size
            m_t = make_map(s, args.canvas, gk, (H, W), device)
            if ds.labels[i] == 0:
                g_nat = np.zeros((H, W), np.uint8)
                g_ap = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
            else:
                gpil = Image.open(ds.gt_paths[i]).convert('L')
                g_nat = (np.asarray(gpil) > 127).astype(np.uint8)
                g_ap = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
            h.add(m_t.cpu().numpy().reshape(-1), g_nat.reshape(-1))
            ap_preds.append(make_map(s, args.canvas, gk, (args.aupro_res, args.aupro_res), device).cpu().numpy())
            ap_gts.append(g_ap)
            r = max(1, round(min(H, W) / G))
            g_t = torch.from_numpy(g_nat).to(device) > 0
            pred = closing(m_t > thr[v], 2 * r + 1)
            mst += ((pred & g_t).sum().item(), (pred & ~g_t).sum().item(), ((~pred) & g_t).sum().item())
            del m_t, g_t
        tp, fp, fn = mst
        out[v] = {'aupro': aupro05(ap_preds, ap_gts, rng), 'f1_max': h.f1_max(),
                  'f1': float(2 * tp / (2 * tp + fp + fn + 1e-9))}
        del ap_preds, ap_gts
        p(f'    [{cat}] {v:4s}: AUPRO0.05={out[v]["aupro"]:.4f}  trần={out[v]["f1_max"]:.4f}  F1@rule={out[v]["f1"]:.4f}')
    del s_grids
    return out


def main():
    ap = argparse.ArgumentParser('eval_overlapmap: non-overlap vs cửa sổ chồng nửa + mean score (nền coreset+1NN, rule đóng băng)')
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
    ap.add_argument('--out_dir', type=str, default='./overlapmap')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('overlapmap', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} bank={args.bank_size}(coreset) cand={args.cand_size} '
      f'heldout={args.n_val} | RULE ĐÓNG BĂNG p={RULE_P} gain={RULE_G} morph=CÓ')
    p('  variants: base (=core1 bankmap, non-overlap 3x3) | ov (5x5 chồng nửa + mean score). FAIR, không per-cat.')

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
        p(f'  {v:4s}: AUPRO0.05={mean[v]["aupro"]:.4f}  trần={mean[v]["f1_max"]:.4f}  F1@rule={mean[v]["f1"]:.4f}')
    d = {k: mean['ov'][k] - mean['base'][k] for k in mean['base']}
    p(f'\n  Δ OVERLAP (ov - base): AUPRO{d["aupro"]:+.4f}  trần{d["f1_max"]:+.4f}  F1{d["f1"]:+.4f}')
    p('\nĐỌC (pre-registered): ΔAUPRO>=+0.010 hoặc Δtrần>=+0.020 -> overlap VÀO config nền. '
      '~0/âm -> overlap chết ở quick harness -> còn guided native-res upsampling (novelty front).')


if __name__ == '__main__':
    main()
