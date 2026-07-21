# eval_thr_rules.py
# -----------------------------------------------------------------------------
# SHOOTOUT quy tắc ngưỡng png — sửa cú vỡ SegF1 private (server per-cat 2026-07-18):
#   fabric AUPRO 95.16 / F1 46.36 (public 86.29): map gần hoàn hảo, NGƯỠNG phá.
#   rice/sheet_metal: rate rule THẮNG (71.6/62.6 > public@ksig). => rate rule chỉ vỡ
#   khi prevalence (tỉ lệ pixel defect) private lệch public — đúng rủi ro đã gắn cờ.
#
# 4 rule (đều dùng được trên private, không cần GT đích):
#   ksig  : mean + 4.5σ pooled trên phân bố ĐÍCH (test_ksig cũ, sub 0.4584).
#   rate  : exceedance = r_pub (tỉ lệ pixel defect public). VỠ khi prevalence lệch.
#   medq  : MEDIAN qua ảnh của quantile-từng-ảnh tại (1-FPR*), FPR* = FPR của ngưỡng
#           oracle trên public. Median giết ảnh lỗi thiểu số -> MIỄN NHIỄM prevalence;
#           tự trượt theo shift như ksig nhưng không bị pixel defect làm bẩn mean/σ.
#   fbrate: rate với r = (share pixel của ảnh lỗi trong split) x a_cat(public).
#           Ở đây dùng f_bad THẬT (trần của estimator); nếu ~ medq thì medq thắng
#           (không cần estimator).
#
# Mô phỏng lệch prevalence: giữ nguyên ảnh good, subsample ảnh bad x{1.0, 0.5, 0.25}
# (tái tạo kiểu vỡ fabric). Mỗi rule chấm F1 trên pooled hist của subset, so ORACLE
# của chính subset. Per-image histogram (pos/neg) lưu 1 lần -> mọi subset/rule tính
# lại tức thì, chỉ MỘT pass encode.
#
# Setup = ĐÚNG nhánh png của submission (PER_CAT config + f1_head + seed order).
#
# ĐỌC (pre-register): chọn rule có WORST-CASE F1 (qua các mức prevalence) cao nhất,
#   tính MEAN qua cat. Kỳ vọng: medq ~ oracle ở mọi mức; rate sập ở x0.25.
#
#   python eval_thr_rules.py --data_path ../data --out_dir ./thrrules
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

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, IMG_EXT,
)
from infer_submit_nrs import PER_CAT                                    # noqa: E402
from eval_nrs_head import build_nrs_head                                # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
NB = 8192
FRACS = [1.0, 0.5, 0.25]                     # tỉ lệ ảnh bad giữ lại (mô phỏng prevalence private)
RULES = ['ksig', 'rate', 'medq', 'fbrate']


def f1_at_bin(hp, hn, b):
    tp = float(hp[b:].sum())
    fp = float(hn[b:].sum())
    fn = float(hp[:b].sum())
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def f1_max_bin(hp, hn):
    tp = np.cumsum(hp[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(hn[::-1])[::-1].astype(np.float64)
    fn = float(hp.sum()) - tp
    return float(np.max(2 * tp / (2 * tp + fp + fn + 1e-9))), int(np.argmax(2 * tp / (2 * tp + fp + fn + 1e-9)))


def exceed_bin(h, k):
    """bin nhỏ nhất b sao cho số đếm ở [b:] <= k (ngưỡng đạt exceedance k)."""
    cum = np.cumsum(h[::-1])[::-1]
    return int(np.searchsorted(-cum, -float(k)))


def run_cat(bb, cat, args, layers, gk, device, p):
    cfg = PER_CAT[cat]
    args.tiles, args.grid_tile = cfg['tiles'], cfg['grid_tile']
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    p(f'  [{cat}] cfg={T}:{gt_} f1_head={cfg["f1_head"]} | bank từ {len(tr)} ảnh...')
    bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shots = bad[:args.shots]
    torch.manual_seed(args.seed)                       # ĐÚNG thứ tự seed của infer_submit_nrs
    head_grid = build_head(bb, ds, shots, bank, args, layers, device)
    head_nrs = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)
    if head_nrs is None:
        return None
    head_f1 = head_nrs if (cfg['f1_head'] == 'nrs' or head_grid is None) else head_grid

    idx_bad = [i for i in bad if i not in set(shots)]
    idx_good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    if args.max_eval:
        idx_bad = idx_bad[:args.max_eval]
        idx_good = idx_good[:args.max_eval]
    idx = idx_bad + idx_good

    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]
            d = np.asarray(nn_map(g, bank, device))
            pr = torch.sigmoid(head_f1(g.reshape(-1, C))).reshape(G, G).cpu().numpy()
            grids.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])
    s_grids = [((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32) for d, pr in grids]
    glo = min(float(s.min()) for s in s_grids) - 0.05
    ghi = max(float(s.max()) for s in s_grids) + 0.05

    # per-image histogram (pos/neg) + moments -> mọi rule/subset tính offline
    stats = []                                          # (hp, hn, n, s1, s2, is_bad)
    for s, (H, W), i in zip(s_grids, sizes, idx):
        with torch.no_grad():
            t = torch.tensor(s, device=device)[None, None].float()
            t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
            t = gk(t)
            m = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy().reshape(-1)
        if ds.labels[i] == 0:
            gm = np.zeros(m.size, bool)
        else:
            gm = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).reshape(-1)
        b = np.clip(((m - glo) / (ghi - glo) * NB).astype(np.int32), 0, NB - 1)
        stats.append((np.bincount(b[gm], minlength=NB).astype(np.int64),
                      np.bincount(b[~gm], minlength=NB).astype(np.int64),
                      m.size, float(m.sum(dtype=np.float64)), float(np.square(m, dtype=np.float64).sum()),
                      ds.labels[i] == 1))

    # fit PUBLIC-full (đóng vai "nguồn"): r_pub, FPR*, a_cat
    HP = sum(st[0] for st in stats)
    HN = sum(st[1] for st in stats)
    n_all = sum(st[2] for st in stats)
    pos_all = float(HP.sum())
    r_pub = pos_all / n_all
    _, ob = f1_max_bin(HP, HN)
    fpr_star = max(float(HN[ob:].sum()) / max(float(HN.sum()), 1.0), 1e-9)
    pix_bad = sum(st[2] for st in stats if st[5])
    a_cat = pos_all / max(pix_bad, 1)
    p(f'    [{cat}] r_pub={r_pub:.3e} FPR*={fpr_star:.3e} a_cat={a_cat:.3e} '
      f'(bad {sum(1 for st in stats if st[5])} / good {sum(1 for st in stats if not st[5])} ảnh)')

    # per-image quantile bin tại (1-FPR*) — cho medq
    qbins = []
    for hp, hn, n, _, _, _ in stats:
        qbins.append(exceed_bin(hp + hn, fpr_star * n))

    out = {}
    bad_pos = [j for j, st in enumerate(stats) if st[5]]
    good_pos = [j for j, st in enumerate(stats) if not st[5]]
    for f in FRACS:
        S = good_pos + bad_pos[:max(1, round(f * len(bad_pos)))]
        hp = sum(stats[j][0] for j in S)
        hn = sum(stats[j][1] for j in S)
        n_s = sum(stats[j][2] for j in S)
        s1 = sum(stats[j][3] for j in S)
        s2 = sum(stats[j][4] for j in S)
        orc, _ = f1_max_bin(hp, hn)
        row = {'oracle': orc}
        # ksig
        mu = s1 / n_s
        sd = np.sqrt(max(s2 / n_s - mu * mu, 0.0))
        bk = int(np.clip((mu + args.thr_sigma * sd - glo) / (ghi - glo) * NB, 0, NB))
        row['ksig'] = f1_at_bin(hp, hn, bk)
        # rate (fit full-public, áp lên subset — đúng kiểu vỡ private)
        row['rate'] = f1_at_bin(hp, hn, exceed_bin(hp + hn, r_pub * n_s))
        # medq
        row['medq'] = f1_at_bin(hp, hn, int(np.median([qbins[j] for j in S])))
        # fbrate: r = share-pixel-ảnh-bad(subset) x a_cat(full fit)
        pb = sum(stats[j][2] for j in S if stats[j][5])
        row['fbrate'] = f1_at_bin(hp, hn, exceed_bin(hp + hn, (pb / n_s) * a_cat * n_s))
        out[f] = row
        p(f'    [{cat}] bad x{f:<4}: oracle={orc:.4f}  ' +
          '  '.join(f'{r}={row[r]:.4f}' for r in RULES))
    return out


def main():
    ap = argparse.ArgumentParser('shootout quy tắc ngưỡng png dưới mô phỏng lệch prevalence')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
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
    ap.add_argument('--max_eval', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=list(PER_CAT.keys()))
    ap.add_argument('--out_dir', type=str, default='./thrrules')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('thrrules', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(v / 12 * bb.n_layers))) for v in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} model={args.model} rules={RULES} fracs={FRACS} k={args.thr_sigma} '
      f'| per-cat config + f1_head THEO SUBMISSION (PER_CAT)')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        return

    p('\n' + '=' * 96 + '\n===== TỔNG =====')
    for f in FRACS:
        orc = float(np.mean([res[c][f]['oracle'] for c in res]))
        row = '  '.join(f'{r}={np.mean([res[c][f][r] for c in res]):.4f}' for r in RULES)
        p(f'  bad x{f:<4}: oracle={orc:.4f}  {row}')
    p('\n  WORST-CASE qua các mức prevalence (mean qua cat):')
    for r in RULES:
        wc = float(np.mean([min(res[c][f][r] for f in FRACS) for c in res]))
        gap = float(np.mean([min(res[c][f][r] - res[c][f]['oracle'] for f in FRACS) for c in res]))
        p(f'    {r:6s}: worst-F1={wc:.4f}  worst-gap-to-oracle={gap:+.4f}')

    p('\nĐỌC (pre-registered): chọn rule worst-case cao nhất làm ngưỡng png cho submission sau.')
    p('  Kỳ vọng: medq bám oracle ở mọi mức (miễn nhiễm prevalence); rate sập ở x0.25 (= fabric')
    p('  private); ksig ổn nhưng dưới medq vì mean/σ bị pixel defect kéo. fbrate ~ medq =>')
    p('  không cần estimator f_bad, dùng medq cho đơn giản.')


if __name__ == '__main__':
    main()
