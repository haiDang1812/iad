# eval_synthdefect.py
# -----------------------------------------------------------------------------
# LEVER (2): head few-shot là cú nhảy LỚN NHẤT lịch sử dự án (dist 0.44 -> fused 0.72),
#   NHƯNG học từ 10 shot -> overfit -> transfer kém (router đa-scale vừa chết vì đúng lý do này).
#   NÚT THẮT = DỮ LIỆU, không phải model.
#
# Ý tưởng: SYNTHETIC-DEFECT augmentation. Sinh defect nhân tạo (cut-paste / noise / intensity)
#   trên ảnh TRAIN NORMAL (vô hạn) -> hàng nghìn cặp (defect-pixel, normal-pixel) đa dạng hình
#   dạng + độ sáng -> head GENERALIZE thay vì thuộc 10 shot -> nâng trần in-domain (tới oracle)
#   + transfer tốt hơn (supervision transfer > mọi unsupervised trick đã chết).
#
# So — CẢ 2 metric trên test_public — head trained on:
#   real   : chỉ 10 shot thật (hiện tại)
#   synth  : chỉ synthetic
#   real+synth : gộp (đề xuất)
#   dist-only : head_w=0 (mốc không head)
# PASS = real+synth > real trên CẢ HAI => synthetic supervision nâng head => build vào infer,
#        rồi submit đo transfer thật. FAIL = synth ~ real hoặc hại => synthetic OOD, bỏ.
#
#   python eval_synthdefect.py --data_path ../data --out_dir ./synth --tiles 3 --grid_tile 24
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
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, img_featgrid, nn_map, gt_grid, up_to, cc_label, subsample,
    Head, train_softpro, train_bce, VALID, IMG_EXT, SMOOTH_RES,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def aupro05(maps, gts):
    sp = np.array([float(m.max()) for m in maps])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(maps), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def segf1_ksig(maps, gts, k):
    P = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    thr = float(P.mean() + k * P.std())
    TP = FP = FN = 0.0
    for m, g in zip(maps, gts):
        pred = m >= thr; gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
    return 2 * TP / (2 * TP + FP + FN + 1e-9)


def synth_defect(pil, rng):
    """Sinh 1 anomaly nhân tạo trên ảnh normal: cut-paste / noise / intensity-jitter."""
    im = np.array(pil.convert('RGB')); H, W = im.shape[:2]
    rw = int(rng.uniform(0.05, 0.25) * W); rh = int(rng.uniform(0.05, 0.25) * H)
    rw, rh = max(rw, 4), max(rh, 4)
    x0 = int(rng.integers(0, max(1, W - rw))); y0 = int(rng.integers(0, max(1, H - rh)))
    mode = int(rng.integers(0, 3))
    if mode == 0:                                                # cut-paste (structural)
        sx = int(rng.integers(0, max(1, W - rw))); sy = int(rng.integers(0, max(1, H - rh)))
        im[y0:y0 + rh, x0:x0 + rw] = im[sy:sy + rh, sx:sx + rw]
    elif mode == 1:                                              # noise (textural)
        im[y0:y0 + rh, x0:x0 + rw] = rng.integers(0, 256, (rh, rw, 3), dtype=np.uint8)
    else:                                                        # intensity/color jitter (mô phỏng shift cục bộ)
        patch = im[y0:y0 + rh, x0:x0 + rw].astype(np.float32)
        patch = np.clip(patch * rng.uniform(0.3, 1.7) + rng.uniform(-40, 40), 0, 255)
        im[y0:y0 + rh, x0:x0 + rw] = patch.astype(np.uint8)
    mask = np.zeros((H, W), np.uint8); mask[y0:y0 + rh, x0:x0 + rw] = 1
    return Image.fromarray(im), mask


def gather_pos(g, mask_G, Cdim, rbase):
    """g: feature grid [G,G,C] (numpy); mask_G: [G,G] {0,1} -> (pos_list, rid_list, rbase)."""
    lab, n = cc_label(mask_G); flat = g.reshape(-1, Cdim)
    pos_list, rid_list = [], []
    for rid in range(1, n + 1):
        idxs = (lab.reshape(-1) == rid)
        if idxs.sum() > 0:
            pos_list.append(flat[idxs]); rid_list.append(np.full(int(idxs.sum()), rbase)); rbase += 1
    return pos_list, rid_list, rbase


def make_head(pos_list, rid_list, bank, args, device):
    Cdim = bank.shape[-1]
    if sum(len(x) for x in pos_list) < 3 or len(rid_list) < 1:
        return None
    Xpos = torch.tensor(np.concatenate(pos_list), device=device)
    region_ids = np.concatenate(rid_list)
    Xneg = subsample(bank, args.n_neg).detach()
    mu = bank.mean(0, keepdim=True); sd = bank.std(0, keepdim=True) + 1e-6
    if args.loss == 'softpro':
        return train_softpro(Head(Cdim, mu, sd).to(device), Xpos, region_ids, Xneg,
                             args.steps, args.lr, args.q, args.temp, args.w_bce, args.w_fp, device).eval()
    return train_bce(Head(Cdim, mu, sd).to(device), Xpos, Xneg, args.steps, args.lr, device).eval()


def run_cat(bb, cat, args, layers, gk, device):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch; hw = args.head_w; k = args.thr_sigma; Cdim = None
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    tr_bank = tr[:args.max_train] if args.max_train else tr
    bank = build_bank(bb, tr_bank, T, R, gt, layers, args.enc_batch, args.bank_size, device)
    Cdim = bank.shape[-1]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shot_idx = bad[:args.shots]

    # ---- positives thật (10 shot) ----
    rpos, rrid, rb = [], [], 0
    for i in shot_idx:
        g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch).cpu().numpy()
        G = g.shape[0]; m = gt_grid(ds.gt_paths[i], 1, G)
        pl, rl, rb = gather_pos(g, m, Cdim, rb)
        rpos += pl; rrid += rl

    # ---- positives synthetic (trên ảnh train normal) ----
    spos, srid, sb = [], [], 0
    synth_srcs = list(tr); rng.shuffle(synth_srcs); synth_srcs = synth_srcs[:args.n_synth]
    for pth in synth_srcs:
        cim, cmask = synth_defect(Image.open(pth), rng)
        g = img_featgrid(bb, cim, T, R, gt, layers, args.enc_batch).cpu().numpy()
        G = g.shape[0]
        mG = np.asarray(Image.fromarray(cmask * 255).resize((G, G), Image.NEAREST)) > 127
        pl, rl, sb = gather_pos(g, mG.astype(np.uint8), Cdim, sb)
        spos += pl; srid += rl

    rs_rid = list(rrid) + [s + rb for s in srid]                 # synth region-id offset qua real -> không đụng
    heads = {
        'real':  make_head(rpos, rrid, bank, args, device),
        'synth': make_head(spos, srid, bank, args, device),
        'real+synth': make_head(rpos + spos, rs_rid, bank, args, device),
    }
    if heads['real'] is None:
        return None

    # ---- eval trên test_public (loại shot) ----
    idx = [i for i in bad if i not in set(shot_idx)][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    gts = [gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8) for i in idx]
    grids, dmaps = [], []
    for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
        g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch)
        grids.append(g); dmaps.append(np.asarray(nn_map(g, bank, device)))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d in dmaps]), [1, 99])

    def eval_head(head, w):
        maps = []
        for g, d in zip(grids, dmaps):
            if head is not None and w > 0:
                with torch.no_grad():
                    pr = torch.sigmoid(head(g.reshape(-1, Cdim))).reshape(g.shape[0], g.shape[0]).cpu().numpy()
            else:
                pr = 0.0
            fused = (1 - w) * ((d - lo) / (hi - lo + 1e-8)) + w * pr
            maps.append(up_to(fused, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32))
        return aupro05(maps, gts), segf1_ksig(maps, gts, k)

    out = {'dist': eval_head(None, 0.0)}
    for name, h in heads.items():
        out[name] = eval_head(h, hw) if h is not None else (float('nan'), float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser('eval_synthdefect: synthetic-defect augmentation cho head few-shot')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--n_synth', type=int, default=60, help='số ảnh synthetic-defect sinh từ train/good')
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=80)
    ap.add_argument('--max_eval', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./synth')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('synth', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles*args.grid_tile} layers={layers} head_w={args.head_w} '
      f'n_synth={args.n_synth} k={args.thr_sigma}')

    variants = ['dist', 'real', 'synth', 'real+synth']
    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        s = '  '.join(f'{v}=({r[v][0]:.3f}/{r[v][1]:.3f})' for v in variants)
        p(f'  [{cat:11s}] {s}')
    if not res:
        return

    p('\n' + '=' * 78 + '\n===== MEAN (AUPRO0.05 / SegF1@test_ksig) =====')
    m = {v: (float(np.nanmean([res[c][v][0] for c in res])), float(np.nanmean([res[c][v][1] for c in res])))
         for v in variants}
    for v in variants:
        d = '' if v == 'real' else (f'   Δ={m[v][0]-m["real"][0]:+.4f}/{m[v][1]-m["real"][1]:+.4f} vs real'
                                    if v != 'dist' else '')
        tag = {'real': '  <- head hiện tại', 'real+synth': '  <- ĐỀ XUẤT'}.get(v, '')
        p(f'  {v:11s}: AUPRO0.05={m[v][0]:.4f}  SegF1={m[v][1]:.4f}{d}{tag}')

    p('\nĐỌC (Δ thật): real+synth > real trên CẢ HAI => synthetic supervision nâng head =>')
    p('  build vào infer (build_head + synthetic positives) rồi SUBMIT đo transfer thật (kỳ vọng')
    p('  đóng khe public->private vì head hết overfit 10 shot). Nếu synth ~ real hoặc hại => OOD, bỏ.')


if __name__ == '__main__':
    main()
