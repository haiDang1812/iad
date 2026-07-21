# diag25_threshold_transfer.py
# -----------------------------------------------------------------------------
# CÂU HỎI: cú tụt SegF1 private (0.33 vs winner ~0.60) là do MAP xấu, hay do NGƯỠNG?
#
# Fact đã verify:
#   - Server chấm SegF1 trên mask nhị phân .png DO TA CẮT (check_and_prepare_data_for_upload.py).
#   - infer_submit dòng 359: lo/hi lấy percentile trên PRIVATE (đã transductive).
#     infer_submit dòng 370: thr = mean+3σ trên validation/good (domain NGUỒN).
#   -> map chuẩn hóa theo đích, nhưng dao cắt đo ở nguồn.
#
# Diag này tách ĐÔI nguyên nhân:
#   (A) IN-DOMAIN: trên test_public (CÓ GT), quy tắc mean+3σ mất bao nhiêu SegF1 so với
#       ngưỡng oracle? -> quy tắc tự nó tốt hay tệ, chưa dính shift.
#   (B) UNDER-SHIFT: predicted-positive-rate (PPR = tỉ lệ pixel vượt thr) trên
#       validation/good vs test_private vs test_private_mixed. thr cố định, phân bố đổi.
#       PPR_priv >> PPR_val  =>  dao cắt quá THẤP ở đích => precision sập => SegF1 sập.
#
# Và thử luôn 1 lời giải rẻ:
#   THR-TGT (target-quantile): chọn thr sao cho PPR ở đích KHỚP PPR quan sát ở validation.
#   Bất biến với mọi phép dịch/co thang điểm. Đo nó trên test_public (có GT) để biết
#   nó có tốt hơn mean+3σ ngay cả khi KHÔNG có shift hay không.
#
# Attribution: in cả nhánh DIST-only (head_w=0) lẫn FUSE (head_w=0.6). Nếu PPR của FUSE
#   trôi mạnh hơn DIST -> thủ phạm là HEAD (khớp D9: head đọc chiều phương sai thấp).
#
#   python diag25_threshold_transfer.py --data_path ../data --out_dir ./diag25
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
    build_bank, build_head, score_grid, gt_grid, up_to, list_split_files,
    VALID, SPLITS, IMG_EXT, SMOOTH_RES,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')


def segf1(score, gt_bool, thr):
    """F1 pixel tại một ngưỡng cụ thể."""
    pred = score >= thr
    tp = float(np.logical_and(pred, gt_bool).sum())
    fp = float(np.logical_and(pred, ~gt_bool).sum())
    fn = float(np.logical_and(~pred, gt_bool).sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def best_segf1(score, gt_bool):
    """F1-max CHÍNH XÁC: quét mọi ngưỡng bằng cumulative-TP trên score đã sort giảm dần.
    (Lưới quantile cố định sẽ hụt ngưỡng tối ưu khi defect < 0.05% pixel -> F1 giả ~0.)"""
    npos = int(gt_bool.sum())
    if npos == 0:
        return float('nan'), float('nan')
    order = np.argsort(-score, kind='stable')
    g = gt_bool[order]
    tp = np.cumsum(g, dtype=np.int64)
    k = np.arange(1, g.size + 1, dtype=np.int64)
    f1 = 2.0 * tp / (k + npos)
    j = int(f1.argmax())
    return float(f1[j]), float(score[order][j])       # (F1max, thr đạt F1max)


def sub_pixels(a, n, rng):
    """Lấy mẫu pixel để ước lượng phân bố (private nhiều ảnh, không cần giữ hết)."""
    f = a.reshape(-1)
    return f if f.size <= n else f[rng.integers(0, f.size, n)]


def run_cat(bb, cat, args, layers, gk, device):
    # KHÔNG @torch.no_grad() ở đây: build_head phải train (loss.backward()).
    # img_featgrid / nn_map / build_bank / score_grid đều đã tự bọc no_grad.
    T, gt = args.tiles, args.grid_tile
    hw = args.head_w
    rng_np = np.random.default_rng(args.seed)

    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr, T, gt * bb.patch, gt, layers, args.enc_batch, args.bank_size, device)

    # ---- shots: ĐÚNG như infer_submit (rng default_rng(seed), 10 ảnh test_public/bad) ----
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng_np.shuffle(bad)
    shot_pool = bad[:args.shots]
    head = build_head(bb, ds, shot_pool, bank, args, layers, device)
    if head is None:
        return None

    # ---- PASS 1: private -> pooled dist -> lo/hi (y hệt infer_submit) ----
    priv_raw = {}
    for split in SPLITS:
        files, _ = list_split_files(args.data_path, cat, split)
        if not files:
            continue
        if args.max_priv:
            files = files[:args.max_priv]
        priv_raw[split] = [score_grid(bb, Image.open(f), bank, head, args, layers, device)
                           for f in tqdm(files, ncols=70, desc=f'    {cat}/{split}', leave=False)]
    if not priv_raw:
        return None
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for v in priv_raw.values() for d, _ in v]), [1, 99])

    def fuse(d, pr, use_head):
        dr = (d - lo) / (hi - lo + 1e-8)
        return (1 - hw) * dr + hw * pr if use_head else dr

    def up(a):
        return up_to(a, (SMOOTH_RES, SMOOTH_RES), gk, device)

    # ---- validation/good: nguồn của ngưỡng production ----
    val_imgs = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e)) for e in IMG_EXT], []))
    if args.max_val:
        val_imgs = val_imgs[:args.max_val]
    val_pairs = [score_grid(bb, Image.open(v), bank, head, args, layers, device)
                 for v in tqdm(val_imgs, ncols=70, desc=f'    {cat}/val', leave=False)]

    # ---- test_public eval split: LOẠI shot (chống leak), có GT ----
    shots = set(shot_pool)
    pub_idx = [i for i in bad if i not in shots][:args.max_eval]
    pub_idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    pub_pairs, pub_gts = [], []
    for i in tqdm(pub_idx, ncols=70, desc=f'    {cat}/pub', leave=False):
        pub_pairs.append(score_grid(bb, Image.open(ds.img_paths[i]), bank, head, args, layers, device))
        pub_gts.append(gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES))
    G = np.concatenate([g.reshape(-1) for g in pub_gts]).astype(bool)

    out = {'prevalence': float(G.mean())}
    for use_head, tag in [(True, 'FUSE'), (False, 'DIST')]:
        vs = np.concatenate([sub_pixels(up(fuse(d, pr, use_head)), args.px_per_img, rng_np) for d, pr in val_pairs])
        ps = np.concatenate([up(fuse(d, pr, use_head)).reshape(-1) for d, pr in pub_pairs]).astype(np.float32)
        pv = {s: np.concatenate([sub_pixels(up(fuse(d, pr, use_head)), args.px_per_img, rng_np) for d, pr in v])
              for s, v in priv_raw.items()}

        thr_prod = float(vs.mean() + args.thr_sigma * vs.std())     # quy tắc production
        ppr_val = float((vs >= thr_prod).mean())                    # PPR ở NGUỒN

        # (A) in-domain: quy tắc mean+3σ vs oracle, trên test_public (có GT)
        f1_prod = segf1(ps, G, thr_prod)
        f1_best, _ = best_segf1(ps, G)
        # THR-TGT: cắt ở quantile của CHÍNH tập đích sao cho PPR khớp ppr_val
        thr_tgt_pub = float(np.quantile(ps, 1.0 - ppr_val)) if 0 < ppr_val < 1 else thr_prod
        f1_tgt = segf1(ps, G, thr_tgt_pub)

        # (B) under-shift: PPR ở ĐÍCH với CÙNG thr_prod
        ppr = {s: float((v >= thr_prod).mean()) for s, v in pv.items()}
        # thr mà target-quantile SẼ chọn trên private (để xem nó lệch thr_prod bao nhiêu σ_val)
        allp = np.concatenate(list(pv.values()))
        thr_tgt_priv = float(np.quantile(allp, 1.0 - ppr_val)) if 0 < ppr_val < 1 else thr_prod

        out[tag] = {
            'thr_prod': thr_prod, 'ppr_val': ppr_val,
            'f1_prod': f1_prod, 'f1_tgt': f1_tgt, 'f1_best': f1_best,
            'ppr_priv': ppr, 'thr_tgt_priv': thr_tgt_priv,
            'drift_sigma': (thr_tgt_priv - thr_prod) / (vs.std() + 1e-9),
        }
    return out


def main():
    ap = argparse.ArgumentParser('diag25: ngưỡng có transfer sang private không?')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--grid_tile', type=int, default=28)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=64, help='PHẢI trùng submit (64) để bank giống hệt')
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=3.0)
    ap.add_argument('--max_eval', type=int, default=30, help='số ảnh bad & good test_public mỗi bên')
    ap.add_argument('--max_val', type=int, default=30)
    ap.add_argument('--max_priv', type=int, default=60, help='số ảnh mỗi private split (chỉ cần phân bố)')
    ap.add_argument('--px_per_img', type=int, default=20000, help='pixel lấy mẫu/ảnh khi ước lượng phân bố')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./diag25')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag25', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} layers={layers} '
      f'head_w={args.head_w} thr=mean+{args.thr_sigma}std')
    p('PPR = tỉ lệ pixel vượt ngưỡng. PPR_priv >> PPR_val => dao cắt quá thấp ở đích.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ (thiếu data / thiếu defect region ở shots)'); continue
        res[cat] = r
        p(f'\n  [{cat}] defect prevalence (test_public) = {r["prevalence"]*100:.3f}% pixel')
        for tag in ['FUSE', 'DIST']:
            d = r[tag]
            pp = '  '.join(f'{s.replace("test_", "")}={v*100:.2f}%' for s, v in d['ppr_priv'].items())
            p(f'    {tag}: PPR_val={d["ppr_val"]*100:.3f}%  |  {pp}   (thr trôi {d["drift_sigma"]:+.2f}σ_val)')
            p(f'    {tag}: SegF1_pub  mean+3σ={d["f1_prod"]:.4f}   thr-tgt={d["f1_tgt"]:.4f}   oracle={d["f1_best"]:.4f}')

    if not res:
        return
    p('\n' + '=' * 78 + '\n===== MEAN qua category =====')
    for tag in ['FUSE', 'DIST']:
        g = lambda k: float(np.mean([res[c][tag][k] for c in res]))                       # noqa: E731
        ppr_p = float(np.mean([np.mean(list(res[c][tag]['ppr_priv'].values())) for c in res]))
        p(f'  {tag}: PPR_val={g("ppr_val")*100:.3f}%  PPR_priv={ppr_p*100:.3f}%  '
          f'(x{ppr_p/max(g("ppr_val"),1e-9):.1f})  thr trôi {g("drift_sigma"):+.2f}σ_val')
        p(f'  {tag}: SegF1_pub  mean+3σ={g("f1_prod"):.4f}  thr-tgt={g("f1_tgt"):.4f}  oracle={g("f1_best"):.4f}')
    p(f'  defect prevalence trung bình = {float(np.mean([res[c]["prevalence"] for c in res]))*100:.3f}%')

    p('\nĐỌC:')
    p(' (A) IN-DOMAIN: thr-tgt > mean+3σ trên test_public => quy tắc production tự nó đã tệ,')
    p('     sửa ngưỡng ăn điểm ngay cả khi không có shift. Khoảng cách tới oracle = trần còn lại.')
    p(' (B) UNDER-SHIFT: PPR_priv/PPR_val >> 1 => thr quá thấp ở đích => mask phình => precision sập.')
    p('     So PPR_priv của FUSE vs DIST: FUSE trôi mạnh hơn => thủ phạm là HEAD (chiều phương sai thấp,')
    p('     xem D9), không phải distance (shift-gap~1.0 đã cho thấy distance đứng yên).')


if __name__ == '__main__':
    main()
