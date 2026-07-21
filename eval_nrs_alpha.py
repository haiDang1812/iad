# eval_nrs_alpha.py
# -----------------------------------------------------------------------------
# NRS v2 — hai núm vặn sửa cú sập F1 của NRS trên cat defect to (fabric trần 0.32
# dù AUPRO 0.96), đo từ nrs48b. Chẩn đoán: rec-term cân 51 region BẰNG NHAU + pos
# biên là feature trộn ~90% normal -> đẩy region nhỏ toàn-biên vượt tau buộc head
# bắn cả hướng gần-normal -> loang: region vẫn rank tốt (AUPRO cao) nhưng
# precision pixel chết (F1 sập).
#
#   ALPHA — cân region theo kích thước: W_r ∝ n_r^alpha (n_r = pixel native THẬT
#     của region, trước cap). alpha=0 = NRS hiện tại (mọi region ngang nhau = hình
#     dáng AUPRO); alpha=1 = pixel-weighted (region to thống trị = hình dáng
#     pooled-F1). MỘT tham số quét liên tục giữa hai operating point của benchmark.
#   BETA — cân pos theo độ phủ: w_i = cov_i^beta, cov = GT BOX-downsample về lưới
#     (phân số defect trong ô) nội suy tại pixel. Dập pos biên/halo. KHÔNG giết
#     can: mọi pixel can đều low-cov -> sau chuẩn hóa NỘI region trọng số lại đều.
#
# Biến thể: (0,0)=NRS gốc, (1,0), (0,1), (1,1) + grid head (production baseline).
# Cùng bank/shots/encode 1 lần; train head tuyến tính rẻ -> 5 head, 5 map, 1 pass.
#
# ĐỌC (pre-register):
#   - Có biến thể đưa fabric F1 về >= grid-0.03 (>=0.83) mà can F1 >= 0.28
#     -> MỘT head NRS-ab thay được per-cat selection cho nhánh png; nếu AUPRO vẫn
#     >= grid -> có thể MỘT map cho cả hai artifact (novelty mạnh hơn nữa).
#   - Không biến thể nào cứu fabric -> dual-map như đã nộp là thiết kế đúng,
#     đóng hướng này trong 1 buổi.
#
#   python eval_nrs_alpha.py --data_path ../data --out_dir ./nrsab48 \
#       --tiles 3 --grid_tile 48 --categories fabric can sheet_metal
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
from scipy.ndimage import label as cc_label

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, gt_grid, Head, subsample, IMG_EXT,
)
from eval_nrs_head import feats_at                                      # noqa: E402
from eval_native import Hist                                            # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def aupro05(preds, gts):
    sp = np.array([float(p.max()) for p in preds])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(preds), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def collect_nrs_data(bb, ds, shot_pool, args, layers, device, p):
    """Thu data NRS MỘT LẦN (dùng chung cho mọi (alpha,beta)): pos = feature nội suy
    tại pixel defect native + cov (độ phủ ô, BOX) + n_r thật; neg in-image."""
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    G = T * gt_
    rng = np.random.default_rng(args.seed + 1)
    per_img_neg = max(1, args.n_neg_img // max(1, len(shot_pool)))
    pos_f, pos_r, pos_c, nreg, neg_f = [], [], [], [], []
    rbase = 0
    tot_px = 0
    for i in shot_pool:
        if not (isinstance(ds.gt_paths[i], str) and os.path.exists(ds.gt_paths[i])):
            continue
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        gpil = Image.open(ds.gt_paths[i]).convert('L')
        gm = np.asarray(gpil) > 127
        # cov grid: phân số defect trong mỗi ô (BOX = trung bình) -> (1,1,G,G) để nội suy
        cov = torch.tensor(np.asarray(gpil.resize((G, G), Image.BOX), dtype=np.float32) / 255.0,
                           device=device)[None, None]
        g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
        inp = g.permute(2, 0, 1).contiguous()[None]
        del g
        lab, n = cc_label(gm)
        for rid in range(1, n + 1):
            ys, xs = np.nonzero(lab == rid)
            n_true = len(ys)
            if n_true > args.pos_per_region:
                s = rng.choice(n_true, args.pos_per_region, replace=False)
                ys, xs = ys[s], xs[s]
            with torch.no_grad():
                pos_f.append(feats_at(inp, ys, xs, H, W).cpu())
                pos_c.append(feats_at(cov, ys, xs, H, W)[:, 0].clamp(1e-3, 1.0).cpu().numpy())
            pos_r.append(np.full(len(ys), rbase))
            nreg.append(n_true)
            rbase += 1
            tot_px += n_true
        nys, nxs = np.nonzero(~gm)
        s = rng.choice(len(nys), min(per_img_neg, len(nys)), replace=False)
        with torch.no_grad():
            neg_f.append(feats_at(inp, nys[s], nxs[s], H, W).cpu())
        del inp, cov
    if rbase < 1:
        return None
    Xpos = torch.cat(pos_f, 0)
    rids = np.concatenate(pos_r)
    covs = np.concatenate(pos_c)
    # cap tổng pos, giữ >=5 px/region (như eval_nrs_head)
    if len(Xpos) > args.max_pos:
        keep = []
        for r in np.unique(rids):
            ii = np.nonzero(rids == r)[0]
            keep.append(ii if len(ii) <= 5 else rng.choice(ii, 5, replace=False))
        keep = np.unique(np.concatenate(keep))
        rest = np.setdiff1d(np.arange(len(Xpos)), keep)
        extra = rng.choice(rest, min(len(rest), max(0, args.max_pos - len(keep))), replace=False)
        sel = np.sort(np.concatenate([keep, extra]).astype(np.int64))
        Xpos, rids, covs = Xpos[sel], rids[sel], covs[sel]
    p(f'      NRS-data: {rbase} region native / {tot_px} px gốc -> {len(Xpos)} pos, '
      f'cov q10/50/90 = {np.percentile(covs, [10, 50, 90]).round(3)}')
    return Xpos, rids, covs, np.array(nreg, np.float64), torch.cat(neg_f, 0)


def train_softpro_ab(head, Xpos, rids, covs, nreg, Xneg, alpha, beta, args, device):
    """train_softpro với 2 trọng số: W_r = n_r^alpha (giữa region), w_i = cov_i^beta
    (trong region + BCE). alpha=beta=0 == NRS gốc (khớp train_softpro production)."""
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    rt = torch.tensor(rids, device=device)
    uniq = torch.unique(rt)
    wp = torch.tensor(covs, dtype=torch.float32, device=device) ** beta          # (Npos,)
    wr = torch.tensor(nreg, dtype=torch.float32, device=device)[uniq] ** alpha   # (R,)
    y = torch.cat([torch.ones(len(Xpos)), torch.zeros(len(Xneg))]).to(device)
    wbce = torch.cat([len(Xneg) * wp / wp.sum(), torch.ones(len(Xneg), device=device)])
    Xall = torch.cat([Xpos, Xneg], 0)
    for _ in range(args.steps):
        opt.zero_grad()
        sp = head(Xpos)
        sn = head(Xneg)
        tau = torch.quantile(sn.detach(), args.q)
        scale = args.temp * (sn.detach().std() + 1e-6)
        s = torch.sigmoid((sp - tau) / scale)
        rec_r = torch.stack([(s[rt == r] * wp[rt == r]).sum() / wp[rt == r].sum() for r in uniq])
        rec = (rec_r * wr).sum() / wr.sum()
        fp_excess = torch.sigmoid((sn - tau) / scale).mean()
        bce = F.binary_cross_entropy_with_logits(head(Xall), y, weight=wbce)
        loss = (1 - rec) + args.w_fp * fp_excess + args.w_bce * bce
        loss.backward()
        opt.step()
    head.eval()
    return head


def run_cat(bb, cat, args, layers, gk, device, p, variants):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    p(f'  [{cat}] build bank từ {len(tr)} ảnh train...')
    bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shots = bad[:args.shots]

    torch.manual_seed(args.seed)
    head_grid = build_head(bb, ds, shots, bank, args, layers, device)
    data = collect_nrs_data(bb, ds, shots, args, layers, device, p)
    if data is None:
        p(f'  [{cat}] không có data NRS -> bỏ')
        return None
    Xpos_c, rids, covs, nreg, neg_img = data
    Xpos = Xpos_c.to(device)
    Xneg = torch.cat([subsample(bank, args.n_neg).detach().cpu(), neg_img], 0).to(device)
    mu = bank.mean(0, keepdim=True)
    sd = bank.std(0, keepdim=True) + 1e-6

    heads = [('grid', head_grid)]
    for a, b in variants:
        torch.manual_seed(args.seed)                       # init trùng nhau giữa các biến thể
        h = train_softpro_ab(Head(C, mu, sd).to(device), Xpos, rids, covs, nreg, Xneg,
                             a, b, args, device)
        heads.append((f'a{a:g}b{b:g}', h))
    p(f'    heads: grid={"None->dist-only" if head_grid is None else "OK"} + {len(variants)} biến thể NRS-ab')

    idx_bad = [i for i in bad if i not in set(shots)]
    idx_good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    if args.max_eval:
        idx_bad = idx_bad[:args.max_eval]
        idx_good = idx_good[:args.max_eval]
    idx = idx_bad + idx_good
    m = max(2, args.aupro_max) // 2
    ap_ids = set(idx_bad[:m] + idx_good[:m]) if args.aupro_max else set(idx)

    # PASS 1: encode 1 lần, áp MỌI head
    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]
            d = np.asarray(nn_map(g, bank, device))
            flat = g.reshape(-1, C)
            prs = [torch.sigmoid(h(flat)).reshape(G, G).cpu().numpy() if h is not None else None
                   for _, h in heads]
            grids.append((d, prs))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])

    def fuse(d, pr):
        dr = (d - lo) / (hi - lo + 1e-8)
        return dr.astype(np.float32) if pr is None else ((1 - hw) * dr + hw * pr).astype(np.float32)

    # PASS 2: mỗi head -> Hist native + AUPRO@aupro_res
    out = {}
    for k, (name, h) in enumerate(heads):
        s_grids = [fuse(d, prs[k]) for d, prs in grids]
        gmin = min(float(s.min()) for s in s_grids)
        gmax = max(float(s.max()) for s in s_grids)
        hst = Hist(gmin - 0.05, gmax + 0.05)
        ap_preds, ap_gts = [], []
        for s, (H, W), i in zip(s_grids, sizes, idx):
            with torch.no_grad():
                t = torch.tensor(s, device=device)[None, None].float()
                t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
                t = gk(t)
                mnat = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                if i in ap_ids:
                    m512 = F.interpolate(t, size=(args.aupro_res, args.aupro_res),
                                         mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            if ds.labels[i] == 0:
                gnat = np.zeros(mnat.size, np.uint8)
                g512 = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
            else:
                gpil = Image.open(ds.gt_paths[i]).convert('L')
                gnat = (np.asarray(gpil) > 127).astype(np.uint8).reshape(-1)
                g512 = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
            hst.add(mnat.reshape(-1), gnat)
            if i in ap_ids:
                ap_preds.append(m512)
                ap_gts.append(g512)
        tag = ' (dist-only)' if (name == 'grid' and head_grid is None) else ''
        out[name] = {'f1': hst.f1_at(hst.ksig(args.thr_sigma)), 'f1max': hst.f1_max(),
                     'aupro': aupro05(ap_preds, ap_gts), 'tag': tag}
        p(f'    [{cat}] {name:6s}{tag}: F1@ksig={out[name]["f1"]:.4f} trần={out[name]["f1max"]:.4f} '
          f'AUPRO0.05@{args.aupro_res}={out[name]["aupro"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('NRS-ab: alpha (cân region theo size) + beta (cân pos theo coverage)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--variants', type=str, nargs='+', default=['0:0', '1:0', '0:1', '1:1'],
                    help='alpha:beta ; 0:0 = NRS gốc (đối chứng nội bộ)')
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
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split')
    ap.add_argument('--aupro_max', type=int, default=120)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=['fabric', 'can', 'sheet_metal'])
    ap.add_argument('--out_dir', type=str, default='./nrsab')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('nrsab', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(v / 12 * bb.n_layers))) for v in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    variants = [(float(v.split(':')[0]), float(v.split(':')[1])) for v in args.variants]
    p(f'device={device} model={args.model} eff_grid={args.tiles * args.grid_tile} '
      f'layers={args.layers}->{layers} variants(a:b)={args.variants} k={args.thr_sigma} '
      f'| grid = production head; a0b0 = NRS gốc')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, variants)
        if r is not None:
            res[cat] = r
    if not res:
        return

    p('\n' + '=' * 96 + '\n===== TỔNG (thước native, full split trừ shots) =====')
    names = ['grid'] + [f'a{a:g}b{b:g}' for a, b in variants]
    for cat, r in res.items():
        for name in names:
            p(f"  [{cat:11s}] {name:6s}{r[name]['tag']}: F1={r[name]['f1']:.4f} "
              f"trần={r[name]['f1max']:.4f} AUPRO={r[name]['aupro']:.4f}")
    for key, lab in [('f1', 'F1@ksig'), ('f1max', 'trần'), ('aupro', 'AUPRO0.05')]:
        row = '  '.join(f'{name}={np.mean([res[c][name][key] for c in res]):.4f}' for name in names)
        p(f'  MEAN {lab:9s}: {row}')

    p('\nĐỌC (pre-registered):')
    p('  - Biến thể nào fabric F1 >= 0.83 VÀ can F1 >= 0.28 -> MỘT head NRS-ab thay per-cat')
    p('    selection cho png; nếu AUPRO của nó vẫn >= grid -> cân nhắc MỘT map cả 2 artifact.')
    p('  - alpha cứu fabric mà beta không -> vấn đề là cân region; beta cứu mà alpha không ->')
    p('    vấn đề là halo biên. Cả hai cùng cần -> giữ cả 2 núm, alpha là dial operating-point.')
    p('  - Không biến thể nào cứu fabric -> dual-map (đã nộp) là thiết kế đúng, đóng hướng này.')


if __name__ == '__main__':
    main()
