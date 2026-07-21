# eval_lnc.py  (v3)
# -----------------------------------------------------------------------------
# LNC — Local Normal-score Calibration: ngưỡng/hiệu chỉnh per-pixel theo manifold normal cục bộ.
#   θ(x) = mức score của các pixel NORMAL gần x nhất trong feature space. "x có cao hơn cả
#   những thứ normal trông giống nó không?" => tự nâng θ ở vùng texture, hạ ở vùng mượt.
#
# LỊCH SỬ (mỗi bản sửa đúng 1 lỗi đã ĐO được):
#   v1  ref=validation/good  -> SegF1 -0.203. Ngưỡng lại tính trên SOURCE domain = đúng cái bug
#       mà test_ksig đã sửa. Thang val != thang test => θ lệch bậc => vial/sheet_metal sập ~0.
#   v2  ref=test (transductive) + rebase θ_λ = thr_global + λ·σ_s·z(θ)  [λ=0 == global test_ksig,
#       baseline lồng bên trong]. rice +0.159, fabric +0.022 — nhưng vial/fruit_jelly vẫn sập.
#       NGUYÊN NHÂN ĐO ĐƯỢC: reference TỰ NHIỄM DEFECT. vial gt_rate 0.98%, fruit_jelly 0.81%:
#       defect to & đặc trong feature space => k hàng xóm của 1 pixel defect PHẦN LỚN cũng là
#       defect => θ ở đó bị đẩy cao => defect tự che chính nó. rice/sheet_metal defect thưa
#       (0.16%/0.10%) nên ít nhiễm. median+MAD chịu được vài neighbor bẩn, không chịu được đa số.
#   v3  (a) PURIFY: chỉ giữ pixel score <= percentile P làm reference => hàng xóm của defect BUỘC
#           phải là NORMAL giống nó nhất. Sweep --purifies (100 = v2, không lọc).
#       (b) BỎ AUROC. Probe phải nằm trong CHÍNH hệ metric của ta (AUROC cân đều toàn dải FPR,
#           còn SegF1/AUPRO0.05 chỉ sống ở đuôi cực cao => AUROC nói dối được cả 2 chiều):
#             SegF1_max  = TRẦN F1 trên toàn dải ngưỡng của map s vs map s/θ  (bỏ biến quy tắc cắt)
#             AUPRO0.05  = metric thật thứ 2, trên cùng 2 map => LNC có là đòn bẩy CẢ HAI không
#             SegF1@ksig = kết quả thực tế theo λ
#
#   python eval_lnc.py --data_path ../data --out_dir ./lnc --tiles 3 --grid_tile 24 --ref test
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, gt_grid, up_to, VALID, IMG_EXT, SMOOTH_RES,
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


def segf1_max(maps, gts):
    """TRẦN của map: F1 tốt nhất trên toàn dải ngưỡng (pooled pixel). Không phụ thuộc quy tắc cắt,
    nhưng đo bằng ĐÚNG hình học F1 (khác AUROC - cân đều toàn dải FPR, không phải vùng ta vận hành)."""
    y = np.concatenate([g.reshape(-1) for g in gts]).astype(np.uint8)
    s = np.concatenate([m.reshape(-1) for m in maps]).astype(np.float32)
    if y.max() == 0:
        return float('nan')
    pr, rc, _ = precision_recall_curve(y, s)
    return float(np.nanmax(2 * pr * rc / (pr + rc + 1e-12)))


def f1_ppr(maps, gts, thr_maps):
    """thr_maps: scalar hoặc list map ngưỡng per-pixel. Trả (pooled F1, predicted-positive-rate)."""
    TP = FP = FN = TN = 0.0
    for i, (m, g) in enumerate(zip(maps, gts)):
        t = thr_maps if np.isscalar(thr_maps) else thr_maps[i]
        pred = m >= t; gb = g.astype(bool)
        TP += float(np.logical_and(pred, gb).sum())
        FP += float(np.logical_and(pred, ~gb).sum())
        FN += float(np.logical_and(~pred, gb).sum())
        TN += float(np.logical_and(~pred, ~gb).sum())
    n = TP + FP + FN + TN
    return 2 * TP / (2 * TP + FP + FN + 1e-9), (TP + FP) / (n + 1e-9)


@torch.no_grad()
def lnc_theta(feat, ref_feat, ref_score, k, c, robust, chunk=1024):
    """θ(x) = loc + c*scale của score các NORMAL neighbor gần x nhất trong feature space.
    robust=True -> median + c*1.4826*MAD (chịu được neighbor anomaly khi ref=test).
    feat:[P,C] -> [P]"""
    P = feat.shape[0]
    kk = min(k, ref_feat.shape[0])
    out = torch.empty(P, device=feat.device)
    for i in range(0, P, chunk):
        d = torch.cdist(feat[i:i + chunk], ref_feat)          # [chunk, R]
        idx = d.topk(kk, dim=1, largest=False).indices
        sc = ref_score[idx]                                   # [chunk, k]
        if robust:
            med = sc.median(1).values
            mad = (sc - med[:, None]).abs().median(1).values
            out[i:i + chunk] = med + c * 1.4826 * mad
        else:
            out[i:i + chunk] = sc.mean(1) + c * sc.std(1)
    return out


def run_cat(bb, cat, args, layers, gk, device):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch; hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr[:args.max_train] if args.max_train else tr,
                      T, R, gt, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None

    @torch.no_grad()
    def score_feat(pil):
        g = img_featgrid(bb, pil, T, R, gt, layers, args.enc_batch)
        d = np.asarray(nn_map(g, bank, device))
        pr = torch.sigmoid(head(g.reshape(-1, C))).reshape(g.shape[0], g.shape[0]).cpu().numpy()
        return g.reshape(-1, C).half().cpu(), d, pr

    # ---- test: feature + d + pr ----
    idx = [i for i in bad if i not in set(bad[:args.shots])][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]
    tf, td, tp = [], [], []
    for i in tqdm(idx, ncols=70, desc=f'    {cat}/test', leave=False):
        f, d, pr = score_feat(Image.open(ds.img_paths[i]))
        tf.append(f); td.append(d); tp.append(pr)
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d in td]), [1, 99])

    def fuse(d, pr):
        return (1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr

    G = td[0].shape[0]
    s_grid = [fuse(d, pr).astype(np.float32) for d, pr in zip(td, tp)]

    # ---- reference pool (feature, score) ----
    if args.ref == 'val':
        val = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'validation', 'good', e))
                          for e in IMG_EXT], []))[:args.max_val]
        rf, rs = [], []
        for v in tqdm(val, ncols=70, desc=f'    {cat}/ref', leave=False):
            f, d, pr = score_feat(Image.open(v))
            rf.append(f); rs.append(fuse(d, pr).reshape(-1))
        RF0 = torch.cat(rf, 0); RS0 = np.concatenate(rs).astype(np.float32)
    else:                                                     # transductive: chính pixel test
        RF0 = torch.cat(tf, 0); RS0 = np.concatenate([s.reshape(-1) for s in s_grid]).astype(np.float32)

    # ---- ground truth + map baseline (s) ----
    gts = [gt_grid(ds.gt_paths[i], ds.labels[i], SMOOTH_RES).astype(np.uint8) for i in idx]
    smaps = [up_to(s, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32) for s in s_grid]
    P = np.concatenate([m.reshape(-1) for m in smaps])
    thr_g, s_std = float(P.mean() + args.thr_sigma * P.std()), float(P.std())
    base = {'f1max': segf1_max(smaps, gts), 'aupro': aupro05(smaps, gts)}   # TRẦN + AUPRO của map gốc
    robust = (args.ref == 'test') or args.robust

    out = {}
    for pu in args.purifies:
        # PURIFY: bỏ pixel score cao khỏi reference => hàng xóm của defect buộc phải là NORMAL
        #   giống nó nhất, thay vì defect khác (v2 sập vì reference tự nhiễm defect: vial/fruit_jelly).
        keep = np.where(RS0 <= np.percentile(RS0, pu))[0] if pu < 100 else np.arange(RS0.shape[0])
        sel = rng.choice(keep, min(args.ref_size, keep.shape[0]), replace=False)
        ref_feat = RF0[sel].float().to(device); ref_score = torch.tensor(RS0[sel], device=device)

        th_grid = [lnc_theta(f.float().to(device), ref_feat, ref_score, args.k, args.c, robust)
                   .reshape(G, G).cpu().numpy().astype(np.float32) for f in tf]
        tmaps = [up_to(t, (SMOOTH_RES, SMOOTH_RES), gk, device).astype(np.float32) for t in th_grid]

        # (A) PREMISE trong ĐÚNG hệ metric của ta: map hiệu chỉnh cục bộ  r = s/θ.
        #     f1max = trần F1 trên toàn dải ngưỡng (bỏ biến "quy tắc cắt"); aupro = metric thật thứ 2.
        rmaps = [(s / (np.abs(t) + 1e-6)).astype(np.float32) for s, t in zip(smaps, tmaps)]
        prem = {'f1max': segf1_max(rmaps, gts), 'aupro': aupro05(rmaps, gts)}

        # (B) SegF1 thực tế @ test_ksig. λ=0 == global; θ_λ = thr_g + λ·σ_s·z(θ) (rebase về test domain)
        Q = np.concatenate([m.reshape(-1) for m in tmaps])
        t_mu, t_sd = float(Q.mean()), float(Q.std()) + 1e-9
        f1 = {lam: f1_ppr(smaps, gts, [thr_g + lam * s_std * ((t - t_mu) / t_sd) for t in tmaps])
              for lam in args.lams}
        out[pu] = {'prem': prem, 'f1': f1}

    return {'base': base, 'pu': out, 'gt_rate': float(np.mean([g.mean() for g in gts]))}


def main():
    ap = argparse.ArgumentParser('eval_lnc v2: premise test + rebase λ + ref transductive')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
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
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--ref', type=str, default='test', choices=['val', 'test'])
    ap.add_argument('--robust', action='store_true', help='median+MAD (bật mặc định khi ref=test)')
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--c', type=float, default=4.5)
    ap.add_argument('--ref_size', type=int, default=20000)
    ap.add_argument('--lams', type=float, nargs='+', default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument('--purifies', type=float, nargs='+', default=[100, 98, 95, 90],
                    help='giữ pixel <= percentile P làm reference (100 = không lọc = v2)')
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_val', type=int, default=30)
    ap.add_argument('--max_eval', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./lnc')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('lnc', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles*args.grid_tile} ref={args.ref} k={args.k} c={args.c} '
      f'ref_size={args.ref_size} purifies={args.purifies} lams={args.lams} | λ=0 == global test_ksig '
      f'k={args.thr_sigma}')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        b = r['base']
        p(f'  [{cat:11s}] BASE(map s): SegF1_max={b["f1max"]:.4f}  AUPRO0.05={b["aupro"]:.4f}   '
          f'(gt_rate={r["gt_rate"]:.4f})')
        for pu in args.purifies:
            q = r['pu'][pu]
            p(f'      purify{pu:>5g}: TRẦN s/θ  SegF1_max={q["prem"]["f1max"]:.4f}'
              f'({q["prem"]["f1max"]-b["f1max"]:+.4f})  AUPRO0.05={q["prem"]["aupro"]:.4f}'
              f'({q["prem"]["aupro"]-b["aupro"]:+.4f})  |  SegF1@ksig  '
              + '  '.join(f'λ{lm:g}={q["f1"][lm][0]:.4f}' for lm in args.lams))
    if not res:
        return

    lam0 = args.lams[0]
    p('\n' + '=' * 96 + '\n===== MEAN (metric thật: SegF1 & AUPRO0.05 — không dùng AUROC) =====')
    p(f'  BASE map s     : SegF1_max={np.mean([res[c]["base"]["f1max"] for c in res]):.4f}   '
      f'AUPRO0.05={np.mean([res[c]["base"]["aupro"] for c in res]):.4f}   '
      f'SegF1@ksig={np.mean([res[c]["pu"][args.purifies[0]]["f1"][lam0][0] for c in res]):.4f}')
    for pu in args.purifies:
        fm = float(np.mean([res[c]['pu'][pu]['prem']['f1max'] for c in res]))
        ap_ = float(np.mean([res[c]['pu'][pu]['prem']['aupro'] for c in res]))
        best = max(args.lams, key=lambda lm: np.mean([res[c]['pu'][pu]['f1'][lm][0] for c in res]))
        bf = float(np.mean([res[c]['pu'][pu]['f1'][best][0] for c in res]))
        p(f'  purify={pu:>5g}   : SegF1_max={fm:.4f}   AUPRO0.05={ap_:.4f}   '
          f'SegF1@ksig best λ={best:g} -> {bf:.4f}')

    p('\nĐỌC (dừng ở bước FAIL đầu tiên):')
    p('  1) TRẦN: SegF1_max(s/θ) > SegF1_max(s)?  => hiệu chỉnh cục bộ làm MAP tốt hơn thật,')
    p('     đo bằng đúng hình học F1, đã bỏ biến "quy tắc cắt". Nếu không => LNC phá map => GIẾT.')
    p('  2) AUPRO0.05(s/θ) > AUPRO0.05(s)?  => LNC là đòn bẩy CẢ HAI metric (thay luôn map .tiff).')
    p('  3) PURIFY: v2 (purify=100) sập ở vial/fruit_jelly vì reference tự nhiễm defect. Nếu purify')
    p('     thấp hơn cứu được 2 cat đó mà vẫn giữ lãi rice/fabric => cơ chế đã đúng.')
    p('  4) λ: nếu TRẦN lên mà SegF1@ksig không lên => lỗi ở quy tắc cắt, không ở θ.')


if __name__ == '__main__':
    main()
