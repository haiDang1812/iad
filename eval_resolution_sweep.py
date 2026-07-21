# eval_resolution_sweep.py
# -----------------------------------------------------------------------------
# LEVER B (nâng trần MAP, không đụng shift): sweep ĐỘ PHÂN GIẢI / granularity của
#   lưới feature rồi đo THẲNG chất lượng map trên test_public (có GT).
#
# Vì sao res: AUPRO0.05 + SegF1 đều thưởng ĐỊNH VỊ BIÊN SẮC ở FPR thấp. Lưới hiện
#   tại 56x56 (tiles=2, grid_tile=28) upsample lên 256 -> biên tù, defect nhỏ nhòe.
#   Lưới mịn hơn (nhiều tile / grid_tile lớn) -> biên sắc -> cả 2 metric lên. Đây là
#   đòn bẩy "nâng cả đường cong", KHÁC projection (chỉ shift-specific, đã bác bỏ).
#
# Đo trên nhánh DISTANCE (bank NN), KHÔNG head/fuse/threshold -> cô lập chất lượng map:
#   - AUPRO0.05 : threshold-free (metric chính).
#   - best-SegF1: F1 pixel tại ngưỡng tối ưu (TRẦN operating-point của map đó).
#   Config nào nâng CẢ HAI so với 2:28 (production) -> đáng nấu vào submission.
#
# Config = "tiles:grid_tile", eff-grid = tiles*grid_tile. Nặng dần theo eff-grid.
#   python eval_resolution_sweep.py --data_path ../data --configs 2:28 2:36 3:24 3:28
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
from infer_submit_mvtec_ad2 import (                      # noqa: E402
    build_bank, img_featgrid, nn_map, up_to, VALID, IMG_EXT,
)
from dataset import MVTecAD2Dataset                        # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator  # noqa: E402
from backbones_ext import load_backbone                    # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def aupro05(preds, gts):
    sp = np.array([float(p.max()) for p in preds])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(preds), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def best_segf1(preds, gts):
    """F1 pixel tại ngưỡng tối ưu (pooled) = trần operating-point của map.

    Quét ĐỦ mọi ngưỡng bằng cumulative-TP trên score đã sort giảm dần. Lưới quantile cố
    định (linspace tới 0.9995) hụt ngưỡng tối ưu khi defect chiếm <0.05% pixel — can /
    wallplugs — và trả F1 ~0 dù map tốt.
    """
    P = np.concatenate([p.reshape(-1) for p in preds]).astype(np.float32)
    G = np.concatenate([g.reshape(-1) for g in gts]).astype(bool)
    npos = int(G.sum())
    if npos == 0:
        return float('nan')
    g = G[np.argsort(-P, kind='stable')]
    tp = np.cumsum(g, dtype=np.int64)                 # TP khi lấy top-k pixel
    k = np.arange(1, g.size + 1, dtype=np.int64)
    return float((2.0 * tp / (k + npos)).max())       # 2tp/(2tp+fp+fn) = 2tp/(k+npos)


def run_cat(bb, cat, tiles, grid_tile, args, layers, gk, device, p):
    T, gt = tiles, grid_tile
    R, L = grid_tile * bb.patch, layers
    er = args.eval_res
    rng = np.random.default_rng(args.seed)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    p(f'  [{cat}] {T}:{gt}: build bank từ {len(tr)} ảnh train...')
    bank = build_bank(bb, tr, T, R, gt, L, args.enc_batch, args.bank_size, device)

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    eval_idx = bad + good

    preds = []; gts = []
    for i in tqdm(eval_idx, ncols=70, desc=f'    {cat} {tiles}:{grid_tile}', leave=False):
        grid = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt, L, args.enc_batch)
        G = grid.shape[0]
        # GT: native -> eval_res TRỰC TIẾP (BOX rồi >0 giữ mọi defect chạm pixel đích).
        # KHÔNG đi qua lưới G — gt_grid nuốt defect nhỏ (cơ chế giết can) và làm ruler
        # thay đổi theo config -> sweep sẽ tự khen lưới mịn một cách giả tạo.
        if ds.labels[i] == 1 and ds.gt_paths[i]:
            gm = Image.open(ds.gt_paths[i]).convert('L').resize((er, er), Image.BOX)
            gg = (np.asarray(gm) > 0).astype(np.uint8)
        else:
            gg = np.zeros((er, er), np.uint8)
        gts.append(gg)
        preds.append(up_to(nn_map(grid, bank, device), (er, er), gk, device))
    return {'aupro05': aupro05(preds, gts), 'segf1': best_segf1(preds, gts), 'grid': G}


def main():
    ap = argparse.ArgumentParser('sweep resolution/granularity -> map quality (AUPRO0.05 + best-SegF1)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--configs', type=str, nargs='+', default=['3:24', '3:32', '3:40', '3:48'],
                    help='tiles:grid_tile ; production = 3:24 (eff 72); giữ tiles=3, tăng res '
                         'mỗi tile -> eff 72/96/120/144, số token/ảnh ~ nhau về chi phí')
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--max_train', type=int, default=60, help='cap ảnh train cho bank (0 = full); '
                    '60 ảnh × grid 72² = 311k patch >> bank_size, đủ dư')
    ap.add_argument('--eval_res', type=int, default=1024)
    ap.add_argument('--max_eval', type=int, default=40, help='số ảnh defect & good mỗi bên')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./ressweep')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('ressweep', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bb = load_backbone(args.model, device)
    # remap layer 1-12 -> độ sâu thật của backbone (GIỐNG production/diag29 — thiếu dòng này
    # là đo feature nông 2..9/24, số sập toàn tập)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    cfgs = [(int(c.split(':')[0]), int(c.split(':')[1])) for c in args.configs]
    p(f'device={device} model={args.model} configs={args.configs} '
      f'(eff-grid={[t*g for t, g in cfgs]}) max_eval={args.max_eval} '
      f'layers={args.layers}->{layers} max_train={args.max_train}')

    # results[cfg][cat] = dict
    results = {c: {} for c in args.configs}
    for cat in args.categories:
        for cstr, (T, gt) in zip(args.configs, cfgs):
            r = run_cat(bb, cat, T, gt, args, layers, gk, device, p)
            if r is None:
                continue
            results[cstr][cat] = r
            p(f'  [{cat}] {cstr} (grid={r["grid"]}x): AUPRO0.05={r["aupro05"]:.4f}  best-SegF1={r["segf1"]:.4f}')

    p('\n===== MEAN qua category =====')
    base = args.configs[0]
    base_au = np.mean([results[base][c]['aupro05'] for c in results[base]]) if results[base] else float('nan')
    base_f1 = np.mean([results[base][c]['segf1'] for c in results[base]]) if results[base] else float('nan')
    for cstr in args.configs:
        cats = results[cstr]
        if not cats:
            p(f'  {cstr}: (không có data)'); continue
        au = float(np.mean([cats[c]['aupro05'] for c in cats]))
        f1 = float(np.mean([cats[c]['segf1'] for c in cats]))
        tag = '  <- production' if cstr == base else \
              f'   ΔAUPRO={au-base_au:+.4f}  ΔSegF1={f1-base_f1:+.4f}'
        p(f'  {cstr:>6} (eff-grid={cstr.split(":")[0]}x{cstr.split(":")[1]}): '
          f'AUPRO0.05={au:.4f}  best-SegF1={f1:.4f}{tag}')
    p('\nĐỌC: config nào nâng CẢ AUPRO0.05 LẪN best-SegF1 so với production (2:28) -> nấu vào '
      'submission (đổi --tiles/--grid_tile). Nếu res cao mà 2 số bão hòa/giảm -> trần map thật '
      'sự là feature, không phải res -> quay lại đổi backbone/layer.')


if __name__ == '__main__':
    main()
