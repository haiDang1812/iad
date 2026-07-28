# eval_nrs_ablation.py
# -----------------------------------------------------------------------------
# ABLATION CHỦ LỰC của paper (chỉ trên AD2 — dataset đích). Bọc eval_nrs_head.run_cat
# lại và quét theo (grid_tile) x (seed) x (category) để ra HAI bảng paper-ready:
#
#   A1  MULTI-SEED  (cố định lưới production 3:48 = eff_grid 144, quét nhiều seed):
#       ->  ΔAUPRO0.05 = mean ± std qua seed, và số cat NRS thắng (k/8) MỖI seed.
#       Đây là cái giết phản biện "single-seed fluke": biến +0.069 một seed thành
#       0.069 ± σ có khoảng tin cậy.
#
#   A2  GRID-SWEEP  (cố định seed, quét grid_tile -> eff_grid 48..192):
#       ->  ΔAUPRO0.05 theo eff_grid, kèm %defect dưới-ô (p_sub) tại mỗi lưới.
#       Cơ chế sub-cell annihilation dự đoán: lưới CÀNG THÔ -> p_sub CÀNG CAO ->
#       ΔNRS CÀNG LỚN. Nếu Δ tăng đơn điệu khi lưới thô -> +0.069 thành MỘT QUY LUẬT
#       có kiểm soát, không còn là con số lẻ. Tương quan Δ vs p_sub phải dương rõ.
#
#   (A3 số "method của ta" cho bảng so SOTA = lát cắt từ dump JSON này: mỗi cat lấy
#    nrs['aupro'] cho tiff/AUPRO và grid['f1'] cho png/SegF1 — per-metric head.)
#
# CÙNG pipeline / CÙNG bank / CÙNG shots với production; biến DUY NHẤT đổi giữa hai
# nhánh mỗi lần gọi run_cat vẫn là supervision (grid NEAREST vs NRS native). Ở đây ta
# chỉ quét THÊM hai trục ngoài: seed và grid_tile.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy):
#   A1: mean Δ>0 với std < mean và NRS thắng >=6/8 ở ĐA SỐ seed  -> ablation vững, báo 0.069±σ.
#       mean Δ>0 nhưng std ~ mean / thắng lung lay 4-6           -> yếu, phải thêm seed / xem lại.
#   A2: r(Δ, p_sub) >= +0.5 và Δ tăng khi lưới thô               -> cơ chế XÁC NHẬN, dùng làm hình chủ lực.
#       Δ phẳng theo lưới                                        -> lợi ích NRS không đến từ dưới-ô; khai lại.
#
#   # A1 (mặc định lưới 144, 5 seed, 8 cat):
#   python eval_nrs_ablation.py --data_path ../data --out_dir ./abl_seed \
#       --grids 48 --seeds 0 1 2 3 4
#   # A2 (1 seed, quét eff_grid 48/72/96/144/192):
#   python eval_nrs_ablation.py --data_path ../data --out_dir ./abl_grid \
#       --grids 16 24 32 48 64 --seeds 0
# -----------------------------------------------------------------------------
import os
import sys
import json
import copy
import argparse
import warnings

import numpy as np
import torch

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import VALID                                 # noqa: E402  (8 cat AD2)
from eval_nrs_head import run_cat                                        # noqa: E402  (lõi ablation 1-biến)
from eval_generalize import subcell_stats                               # noqa: E402  (biến giải thích cơ chế)
from dataset import MVTecAD2Dataset                                      # noqa: E402
from utils import get_gaussian_kernel, get_logger                        # noqa: E402
from backbones_ext import load_backbone                                  # noqa: E402

warnings.filterwarnings('ignore')


def psub_at(data_path, cat, eff_grid):
    """%region defect dưới-ô tại lưới hiệu dụng eff_grid — đo trên GT gốc, không GPU,
    độc lập seed (quét TẤT CẢ ảnh bad để khỏi phụ thuộc thứ tự shuffle)."""
    try:
        ds = MVTecAD2Dataset(root=os.path.join(data_path, cat), transform=None,
                             gt_transform=None, phase='test')
    except Exception:
        return None
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    st = subcell_stats(ds, bad, eff_grid, max_img=10 ** 9)
    return st


def main():
    ap = argparse.ArgumentParser('NRS ablation chủ lực (AD2): multi-seed + grid-sweep')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='./abl')
    # ==== HAI TRỤC QUÉT ====
    ap.add_argument('--grids', type=int, nargs='+', default=[48],
                    help='các grid_tile (eff_grid = tiles*grid_tile). A1: [48]; A2: [16 24 32 48 64]')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                    help='các seed. A1: 5 seed; A2: 1 seed đủ')
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    # ==== config uniform AD2 (ĐỪNG ĐỔI — phải trùng production) ====
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--eval_reserve', type=int, default=0)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--n_neg_img', type=int, default=8000)
    ap.add_argument('--pos_per_region', type=int, default=1500)
    ap.add_argument('--max_pos', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split')
    ap.add_argument('--aupro_max', type=int, default=120)
    ap.add_argument('--aupro_res', type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('nrs_abl', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    n_runs = len(args.grids) * len(args.seeds) * len(args.categories)
    p('=' * 100)
    p(f'NRS ABLATION (AD2) | grids(grid_tile)={args.grids} -> eff={[args.tiles * g for g in args.grids]} '
      f'| seeds={args.seeds} | {len(args.categories)} cat | tổng {n_runs} run_cat')
    p(f'  model={args.model} layers={args.layers}->{layers} loss={args.loss} shots={args.shots} '
      f'head_w={args.head_w} k={args.thr_sigma}')
    p('=' * 100)

    # rec[(gt_, seed, cat)] = {'grid': {...}, 'nrs': {...}}  ;  psub[(gt_, cat)] = float
    rec, psub = {}, {}
    done = 0
    for gt_ in args.grids:
        eff = args.tiles * gt_
        for cat in args.categories:
            if (gt_, cat) not in psub:
                st = psub_at(args.data_path, cat, eff)
                psub[(gt_, cat)] = None if st is None else st['p_sub']
        for seed in args.seeds:
            for cat in args.categories:
                done += 1
                a = copy.deepcopy(args)
                a.grid_tile = gt_
                a.seed = seed
                p(f'\n----- [{done}/{n_runs}] eff_grid={eff} (gt={gt_}) seed={seed} cat={cat} '
                  f'p_sub={psub[(gt_, cat)]} -----')
                try:
                    r = run_cat(bb, cat, a, layers, gk, device, p)
                except Exception as e:                                    # 1 cat lỗi không giết cả quét
                    p(f'    !! LỖI {cat} eff={eff} seed={seed}: {e}')
                    r = None
                if r is not None:
                    rec[(gt_, seed, cat)] = {'grid': r['grid'], 'nrs': r['nrs']}
                # dump tăng dần: rớt giữa chừng vẫn còn số
                dump = {'meta': {'grids': args.grids, 'seeds': args.seeds, 'tiles': args.tiles,
                                 'categories': args.categories},
                        'psub': {f'{g}|{c}': v for (g, c), v in psub.items()},
                        'rec': {f'{g}|{s}|{c}': v for (g, s, c), v in rec.items()}}
                with open(os.path.join(args.out_dir, 'ablation.json'), 'w') as f:
                    json.dump(dump, f, indent=2)

    def d_ap(gt_, seed, cat):
        v = rec.get((gt_, seed, cat))
        return None if v is None else v['nrs']['aupro'] - v['grid']['aupro']

    # ============================= A1: MULTI-SEED (mỗi lưới) =============================
    for gt_ in args.grids:
        eff = args.tiles * gt_
        p('\n' + '=' * 100)
        p(f'===== A1 MULTI-SEED | eff_grid={eff} (gt={gt_}) | ΔAUPRO0.05 = nrs - grid =====')
        # bảng: mỗi seed 1 dòng, mean Δ qua 8 cat + số cat thắng
        p(f'  {"seed":>4s} | {"meanΔ":>8s} {"NRS thắng":>9s} | per-cat Δ')
        for seed in args.seeds:
            ds_ = [d_ap(gt_, seed, c) for c in args.categories]
            vv = [x for x in ds_ if x is not None]
            if not vv:
                continue
            win = sum(1 for x in vv if x > 0)
            cells = ' '.join(f'{x:+.3f}' if x is not None else '  NA ' for x in ds_)
            p(f'  {seed:>4d} | {np.mean(vv):+8.4f} {win:>6d}/{len(vv)} | {cells}')
        # per-cat mean±std qua seed
        p(f'  {"-"*40}')
        p(f'  {"cat":13s} {"meanΔ":>8s} {"std":>7s} {"n":>3s}  (qua seed)')
        cat_means = []
        for c in args.categories:
            vv = [d_ap(gt_, s, c) for s in args.seeds]
            vv = [x for x in vv if x is not None]
            if not vv:
                continue
            m, sd = float(np.mean(vv)), float(np.std(vv))
            cat_means.append(m)
            p(f'  {c:13s} {m:+8.4f} {sd:7.4f} {len(vv):>3d}')
        # tổng: gộp mọi (seed,cat)
        allv = [d_ap(gt_, s, c) for s in args.seeds for c in args.categories]
        allv = [x for x in allv if x is not None]
        if allv:
            # mean-của-mean-theo-cat (mỗi cat trọng số bằng nhau) + std-giữa-seed của meanΔ-8cat
            seed_means = []
            for s in args.seeds:
                w = [d_ap(gt_, s, c) for c in args.categories]
                w = [x for x in w if x is not None]
                if w:
                    seed_means.append(np.mean(w))
            p(f'  {"-"*40}')
            p(f'  TỔNG eff={eff}: ΔAUPRO0.05 mean(per-cat)={np.mean(cat_means):+.4f} '
              f'| meanΔ-8cat qua seed = {np.mean(seed_means):+.4f} ± {np.std(seed_means):.4f} '
              f'(n_seed={len(seed_means)})')

    # ============================= A2: GRID-SWEEP =============================
    if len(args.grids) >= 2:
        p('\n' + '=' * 100)
        p('===== A2 GRID-SWEEP | ΔAUPRO0.05 (trung bình qua seed+cat) theo eff_grid =====')
        p(f'  {"eff_grid":>8s} {"gt":>4s} {"meanΔ":>8s} {"meanΔ±":>7s} {"p_sub":>7s}  (p_sub = %defect dưới-ô)')
        sweep = []
        for gt_ in sorted(args.grids):
            eff = args.tiles * gt_
            vv = [d_ap(gt_, s, c) for s in args.seeds for c in args.categories]
            vv = [x for x in vv if x is not None]
            ps = [psub[(gt_, c)] for c in args.categories if psub.get((gt_, c)) is not None]
            if not vv:
                continue
            mΔ, sdΔ = float(np.mean(vv)), float(np.std(vv))
            mps = float(np.mean(ps)) if ps else float('nan')
            sweep.append((eff, mΔ, mps))
            p(f'  {eff:>8d} {gt_:>4d} {mΔ:+8.4f} {sdΔ:7.4f} {mps:7.1%}')
        # tương quan cơ chế: Δ nên TĂNG khi lưới THÔ -> Δ vs eff_grid ÂM, Δ vs p_sub DƯƠNG
        if len(sweep) >= 3:
            effs = np.array([s[0] for s in sweep], float)
            dds = np.array([s[1] for s in sweep], float)
            pss = np.array([s[2] for s in sweep], float)
            if dds.std() > 1e-9 and effs.std() > 1e-9:
                p(f'  r(Δ, eff_grid)  = {np.corrcoef(dds, effs)[0, 1]:+.3f}  (kỳ vọng ÂM: lưới mịn -> Δ nhỏ)')
            if dds.std() > 1e-9 and np.isfinite(pss).all() and pss.std() > 1e-9:
                p(f'  r(Δ, p_sub)     = {np.corrcoef(dds, pss)[0, 1]:+.3f}  (kỳ vọng DƯƠNG rõ: nhiều dưới-ô -> Δ lớn)')

    # ============================= A3: lát cắt số cho bảng SOTA (per-metric head) =============================
    # tiff/AUPRO0.05 = NRS head ; png/SegF1 = GRID head. Lấy tại eff_grid production (max) & seed đầu.
    prod_gt = max(args.grids)
    s0 = args.seeds[0]
    p('\n' + '=' * 100)
    p(f'===== A3 số method (per-metric) tại eff_grid={args.tiles * prod_gt} seed={s0} — dán vào bảng so SOTA =====')
    p(f'  {"cat":13s} {"AUPRO0.05(nrs)":>14s} {"SegF1(grid)":>12s}')
    ap_list, f1_list = [], []
    for c in args.categories:
        v = rec.get((prod_gt, s0, c))
        if v is None:
            p(f'  {c:13s} {"NA":>14s} {"NA":>12s}')
            continue
        au, f1 = v['nrs']['aupro'], v['grid']['f1']
        ap_list.append(au)
        f1_list.append(f1)
        p(f'  {c:13s} {au:14.4f} {f1:12.4f}')
    if ap_list:
        p(f'  {"MEAN":13s} {np.mean(ap_list):14.4f} {np.mean(f1_list):12.4f}  '
          f'(số public full-split, KHÔNG phải private server 79.69/50.39)')

    p(f'\nDump: {os.path.join(args.out_dir, "ablation.json")}  (đủ số cho cả 3 bảng + vẽ scatter A2)')


if __name__ == '__main__':
    main()
